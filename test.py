"""Reviewer-friendly evaluation entry point for TRPointNet.

Run with the default repository layout:
    python test.py

The script evaluates every .npy scene whose filename contains the selected
test-area token. Each scene must contain N rows and at least seven columns:
x, y, z, r, g, b, label.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.whole_scene_dataset import WholeSceneDataset
from models.trpointnet import get_model


CLASS_NAMES = ("trunk", "leaf", "ground")
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the released TRPointNet checkpoint on whole scenes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stanford_indoor3d"),
        help="Directory containing preprocessed N x 7 .npy scene files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/trpointnet_best.pth"),
        help="Path to the released model state_dict.",
    )
    parser.add_argument("--test-area", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-votes", type=int, default=3)
    parser.add_argument("--block-size", type=float, default=1.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="'auto' uses CUDA when available and otherwise CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics.json"),
        help="JSON file for the final metrics.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_points < 1024:
        parser.error("--num-points must be at least 1024 for this architecture")
    if args.num_votes < 1:
        parser.error("--num-votes must be at least 1")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # The released checkpoint contains tensors only, so weights_only=True avoids
    # executing arbitrary pickled Python objects during loading.
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint is not a non-empty model state_dict.")

    model = get_model(NUM_CLASSES, use_normals=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def add_votes(vote_pool: np.ndarray, point_indices: np.ndarray, labels: np.ndarray) -> None:
    """Accumulate one vote for every sampled point prediction."""
    np.add.at(vote_pool, (point_indices.reshape(-1), labels.reshape(-1)), 1)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = WholeSceneDataset(
        root=args.data_dir,
        test_area=args.test_area,
        block_points=args.num_points,
        block_size=args.block_size,
        stride=args.stride,
        num_classes=NUM_CLASSES,
    )
    model = load_model(args.checkpoint, device)

    total_intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_seen = np.zeros(NUM_CLASSES, dtype=np.int64)
    scene_results: list[dict[str, object]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test scenes: {len(dataset)} (Area_{args.test_area})")

    with torch.inference_mode():
        for scene_index in range(len(dataset)):
            scene_name = dataset.scene_names[scene_index]
            ground_truth = dataset.semantic_labels[scene_index]
            votes = np.zeros((ground_truth.shape[0], NUM_CLASSES), dtype=np.int32)

            vote_bar = tqdm(
                range(args.num_votes),
                desc=f"{scene_index + 1}/{len(dataset)} {scene_name}",
                leave=False,
            )
            for _ in vote_bar:
                blocks, point_indices = dataset[scene_index]
                for start in range(0, blocks.shape[0], args.batch_size):
                    end = min(start + args.batch_size, blocks.shape[0])
                    batch = torch.from_numpy(blocks[start:end]).float()
                    batch = batch.to(device, non_blocking=True).transpose(1, 2)
                    logits, _ = model(batch)
                    predictions = logits.argmax(dim=2).cpu().numpy()
                    add_votes(votes, point_indices[start:end], predictions)

            predictions = votes.argmax(axis=1)
            intersection = np.array(
                [
                    np.count_nonzero((predictions == label) & (ground_truth == label))
                    for label in range(NUM_CLASSES)
                ],
                dtype=np.int64,
            )
            union = np.array(
                [
                    np.count_nonzero((predictions == label) | (ground_truth == label))
                    for label in range(NUM_CLASSES)
                ],
                dtype=np.int64,
            )
            seen = np.array(
                [np.count_nonzero(ground_truth == label) for label in range(NUM_CLASSES)],
                dtype=np.int64,
            )

            total_intersection += intersection
            total_union += union
            total_seen += seen
            scene_iou = safe_ratio(intersection, union)
            scene_miou = float(np.nanmean(scene_iou))
            scene_oa = float(intersection.sum() / seen.sum())
            scene_results.append(
                {
                    "scene": scene_name,
                    "IoU": {
                        name: None if np.isnan(value) else float(value)
                        for name, value in zip(CLASS_NAMES, scene_iou)
                    },
                    "mIoU": scene_miou,
                    "OA": scene_oa,
                }
            )
            print(f"{scene_name}: mIoU={scene_miou:.6f}, OA={scene_oa:.6f}")

    class_iou = safe_ratio(total_intersection, total_union)
    miou = float(np.nanmean(class_iou))
    oa = float(total_intersection.sum() / total_seen.sum())
    metrics: dict[str, object] = {
        "classes": list(CLASS_NAMES),
        "test_area": args.test_area,
        "num_scenes": len(dataset),
        "num_votes": args.num_votes,
        "seed": args.seed,
        "IoU": {
            name: None if np.isnan(value) else float(value)
            for name, value in zip(CLASS_NAMES, class_iou)
        },
        "mIoU": miou,
        "OA": oa,
        "scenes": scene_results,
    }

    print("\n========== Final results ==========")
    for name, value in zip(CLASS_NAMES, class_iou):
        rendered = "N/A" if np.isnan(value) else f"{value:.6f}"
        print(f"IoU ({name:>6s}): {rendered}")
    print(f"mIoU: {miou:.6f}")
    print(f"OA:   {oa:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {args.output}")
    return metrics


def main() -> None:
    args = parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()