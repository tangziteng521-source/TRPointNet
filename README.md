# TRPointNet evaluation release

This repository contains the trained TRPointNet weights and the minimal code
needed to reproduce whole-scene semantic-segmentation evaluation. The released
model predicts three point classes: trunk (0), leaf (1), and ground (2).

## 1. Environment

Python 3.10 or 3.11 is recommended.

~~~bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

For a specific CUDA build, install PyTorch using the command generated at
https://pytorch.org/get-started/locally/ and then install the other packages.
CPU execution is supported but substantially slower.

## 2. Test data

Place the preprocessed test scenes in:

~~~text
data/stanford_indoor3d/
~~~

The default test split selects .npy files containing Area_3 in the filename.
See [DATA_FORMAT.md](DATA_FORMAT.md) for the exact array format and naming rules.
The dataset itself is not bundled with this release.

## 3. One-command evaluation

With the default layout, run:

~~~bash
python test.py
~~~

The command prints per-class IoU, mean IoU (mIoU), and overall point accuracy
(OA), then saves the same results to results/metrics.json.

To use a different data location:

~~~bash
python test.py --data-dir /path/to/stanford_indoor3d
~~~

Useful options:

~~~text
--test-area 3       Test split identifier
--batch-size 4      Inference batch size; lower this if GPU memory is limited
--num-points 4096   Points per block (must be at least 1024)
--num-votes 3       Repeated block samplings aggregated by majority vote
--device auto       auto, cuda, or cpu
--seed 0            Reproducible sampling seed
~~~

Run python test.py --help for the complete interface.

## Repository contents

~~~text
checkpoints/trpointnet_best.pth  Released tensor-only model weights
models/trpointnet.py             TRPointNet architecture
models/pointnet2_utils_knn.py    PointNet++/KNN layers
data_utils/whole_scene_dataset.py
test.py                          Standalone evaluation entry point
DATA_FORMAT.md                   Input contract
MODEL_CARD.md                    Model and metric details
SHA256SUMS.txt                   Integrity hashes
~~~

The evaluator has no dependency on training logs, log_dir, dynamic model-name
discovery, or the training-only provider module.

## Reproducibility notes

- The published checkpoint is a tensor-only state_dict and is loaded with
  PyTorch's safe weights_only=True mode when supported.
- The checkpoint architecture uses geometric normal estimation and must be
  instantiated with use_normals=True; the evaluator fixes this setting.
- Voting samples repeated boundary points stochastically. Keep --seed and
  --num-votes unchanged when comparing reported results.
- IoU is computed as TP / (TP + FP + FN) per class. mIoU is the unweighted
  mean over classes present in the aggregate union. OA is the number of correct
  point predictions divided by the total number of evaluated points.

## Citation and license

Before making the repository public, the authors should add the final paper
citation and a license approved by all rights holders. These items cannot be
inferred safely from the supplied experimental files; see
[PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md).