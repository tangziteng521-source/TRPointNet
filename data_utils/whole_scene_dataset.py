"""Whole-scene loader for the released TRPointNet evaluation script."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class WholeSceneDataset:
    """Split preprocessed scenes into overlapping fixed-size point blocks."""

    def __init__(
        self,
        root: Path,
        test_area: int = 3,
        block_points: int = 4096,
        stride: float = 0.5,
        block_size: float = 1.0,
        padding: float = 0.001,
        num_classes: int = 3,
    ) -> None:
        self.root = Path(root)
        self.block_points = block_points
        self.stride = stride
        self.block_size = block_size
        self.padding = padding
        self.num_classes = num_classes

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {self.root}. See DATA_FORMAT.md."
            )

        area_pattern = re.compile(rf"(?:^|_)Area_{test_area}(?:_|\.|$)")
        scene_files = sorted(
            path
            for path in self.root.glob("*.npy")
            if area_pattern.search(path.name)
        )
        if not scene_files:
            raise FileNotFoundError(
                f"No Area_{test_area} .npy files found in {self.root}. "
                "See DATA_FORMAT.md for naming and array format."
            )

        self.scene_names: list[str] = []
        self.scene_points: list[np.ndarray] = []
        self.semantic_labels: list[np.ndarray] = []
        for scene_file in scene_files:
            array = np.load(scene_file, allow_pickle=False)
            if array.ndim != 2 or array.shape[1] < 7:
                raise ValueError(
                    f"{scene_file} must have shape (N, >=7); got {array.shape}."
                )
            if array.shape[0] == 0:
                raise ValueError(f"{scene_file} contains no points.")
            if not np.isfinite(array[:, :7]).all():
                raise ValueError(f"{scene_file} contains NaN or infinite values.")

            labels_float = array[:, 6]
            labels = labels_float.astype(np.int64)
            if not np.array_equal(labels_float, labels):
                raise ValueError(f"{scene_file} contains non-integer labels.")
            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(
                    f"{scene_file} labels must be in [0, {num_classes - 1}]."
                )

            self.scene_names.append(scene_file.stem)
            self.scene_points.append(array[:, :6].astype(np.float32, copy=True))
            self.semantic_labels.append(labels)

    def __len__(self) -> int:
        return len(self.scene_points)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        points = self.scene_points[index]
        coord_min = points[:, :3].min(axis=0)
        coord_max = points[:, :3].max(axis=0)
        grid_x = max(
            1,
            int(np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1),
        )
        grid_y = max(
            1,
            int(np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1),
        )

        block_list: list[np.ndarray] = []
        index_list: list[np.ndarray] = []
        coordinate_scale = coord_max.copy()
        coordinate_scale[np.abs(coordinate_scale) < 1e-12] = 1.0

        for index_y in range(grid_y):
            for index_x in range(grid_x):
                start_x = coord_min[0] + index_x * self.stride
                end_x = min(start_x + self.block_size, coord_max[0])
                start_x = end_x - self.block_size
                start_y = coord_min[1] + index_y * self.stride
                end_y = min(start_y + self.block_size, coord_max[1])
                start_y = end_y - self.block_size

                point_indices = np.where(
                    (points[:, 0] >= start_x - self.padding)
                    & (points[:, 0] <= end_x + self.padding)
                    & (points[:, 1] >= start_y - self.padding)
                    & (points[:, 1] <= end_y + self.padding)
                )[0]
                if point_indices.size == 0:
                    continue

                block_count = int(np.ceil(point_indices.size / self.block_points))
                target_size = block_count * self.block_points
                missing = target_size - point_indices.size
                if missing:
                    repeated = np.random.choice(
                        point_indices,
                        missing,
                        replace=missing > point_indices.size,
                    )
                    point_indices = np.concatenate((point_indices, repeated))
                np.random.shuffle(point_indices)

                block = points[point_indices].copy()
                normalized_xyz = block[:, :3] / coordinate_scale
                block[:, 0] -= start_x + self.block_size / 2.0
                block[:, 1] -= start_y + self.block_size / 2.0
                block[:, 3:6] /= 255.0
                block = np.concatenate((block, normalized_xyz), axis=1)

                block_list.append(block.reshape(-1, self.block_points, 9))
                index_list.append(point_indices.reshape(-1, self.block_points))

        if not block_list:
            raise ValueError(f"Scene {self.scene_names[index]} produced no evaluation blocks.")
        return (
            np.concatenate(block_list, axis=0).astype(np.float32, copy=False),
            np.concatenate(index_list, axis=0).astype(np.int64, copy=False),
        )