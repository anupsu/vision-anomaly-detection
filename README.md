# White Cap anomaly detection

Finds defective white bottle caps, photographed from above by a fixed mono camera, using
[PatchCore](https://arxiv.org/abs/2106.08265): a memory bank of patch features from good caps.
A cap is flagged when any of its patches is far from everything in that bank. Only good caps
are needed for training, and a heatmap shows where each cap differs.

Defects it picks up include missing foil liners, punctured foil, and torn or crumpled foil.

## How it works

| Step | What happens |
|---|---|
| Find the cap | `DiscLocator`: at 1/8 size, Otsu threshold, largest bright region, equal-area circle. Square crop with a 4% margin; everything outside the circle is masked. |
| Rotation | The cap turns freely, so its printed liner can face any way. Each good crop is also learned at 8 rotations, 45° apart (`--rotate-train 8`). |
| Features | Frozen Wide-ResNet-50-2 (ImageNet). Layer 2 + layer 3, each averaged over a 3 × 3 neighbourhood, at 1/8 of the 512 × 512 crop: a 64 × 64 grid of 1,536-dim patch features. One patch is about 14 × 14 px of the original photo. |
| Memory bank | Patches of the good caps (all rotations), reduced by greedy k-centre coreset to 10%. |
| Score | For each patch, the distance to its nearest bank patch. The image score is the worst patch. |
| Threshold | Leave-one-out on the good caps: each is scored against a bank built from the others (all their rotations). The highest such score is the threshold. |

There is no separate validation set; the leave-one-out threshold uses good caps only. Label the
scored images in the review UI to measure missed defects and false alarms.

## Data

Images are not in the repo. Put them here (or pass `--data`):

```
data/WhiteCap/good/   good caps, used for training
data/WhiteCap/all/    every cap to score (may include the good ones; they get their leave-one-out score)
```

Grayscale or RGB images (`.bmp`, `.png`, `.jpg`, `.tif`), one cap per image, bright cap on a
dark background.

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU. The ImageNet weights download on first use.

## Usage

```bash
# check the cap crops first (writes results/WhiteCap/part_crops.png and stops)
python patchcore_score.py --data data/WhiteCap --region part --part-locator disc \
    --part-margin 0.04 --out results/WhiteCap --preview-crops

# train on good/, score all/ (about 14 minutes for 9 good + 51 images on an M-series Mac)
python patchcore_score.py --data data/WhiteCap --region part --part-locator disc \
    --part-margin 0.04 --rotate-train 8 --coreset 0.1 --out results/WhiteCap

# progress of a running job
python patchcore_score.py --status results/WhiteCap

# review UI: scores, heatmaps, good/bad labels (saved to results/WhiteCap/labels.csv)
python review_app.py --data data/WhiteCap --results results/WhiteCap     # http://127.0.0.1:8765
```

The run writes `results/WhiteCap/`:
- `scores.csv`: every image's score and prediction
- `heatmaps/`: one heatmap per image
- `score_hist.png`: the score distribution
- `run_info.json`: threshold and settings
- `maps.npz`: patch maps, used by the review UI's area rules
- `crops.csv` and `part_crops.png`: where the cap was found

Extra scripts:

```bash
# one outside photo (e.g. a phone photo): grayscale, find the rim, crop, match grey levels, score
python scripts/score_image.py --image photo.jpg

# per-image inference time, by step
python scripts/time_inference.py
```

`score_image.py` scores a photo against the same model. A photo taken with another camera,
lighting, background or angle can score high just for looking different from the training
images, so for such photos train on a few good caps photographed the same way.

## Speed

About 0.67 s per image on an Apple M-series GPU, after warm-up. Three-quarters of that is the
nearest-neighbour search (4,096 patches against about 25,000 in the bank). Building the bank
is a one-off of about 100 s.

## Files

- `patchcore_score.py`: locate, train, threshold, score and heatmaps. It also supports other part
  views (`--region full|ring|part|unwrap`, `--part-locator corner|rim|color|disc`); White Cap uses
  `part` + `disc`.
- `ring_unwrap.py`: ring unwrapping, imported by both main scripts (not used for White Cap).
- `review_app.py`: local review and labelling UI.
- `scripts/`: single-photo scoring and timing.

`--mask-text` (masking engraved codes) needs an `engraving_mask.py` module that is not included;
White Cap does not use it.
