# Crop-Species-Detection

A hierarchical crop species classifier built on a **DINOv2 ViT-B/14** backbone,
packaged as an installable Python library (converted from the original
[`crop-species-detection-demo`](https://github.com/diegocaparrosvaquer/crop-species-detection-demo)
notebook demo).

Two-stage pipeline:

1. **Stage 1** — is this image cropland, or not?
2. **Stage 2** — if cropland, which of 9 species is it?
   (`banana`, `maize`, `millets`, `rapeseed`, `soya`, `sorghum`, `sunflower`,
   `vineyard`, `wheat type crop`)

## Installation

## From GitHub

Install the latest development version directly from the GitHub repository:

```bash
pip install git+https://github.com/diegocaparrosvaquer/CropSpeciesDetector-Package.git
```

For the latest stable release, install from PyPI instead:

```bash
pip install crop-species-detection
```

Matplotlib is optional and only needed if you want to reproduce the demo's probability bar charts:

```bash
pip install "crop-species-detection[viz]"
```

## Model checkpoints

The trained model checkpoints are hosted on Hugging Face Hub and are **downloaded automatically** when using `from_pretrained()`.

You do **not** need to clone the original demo repository, install Git LFS, or manually download the checkpoints.

```python
from crop_species_detection import CropSpeciesDetector

detector = CropSpeciesDetector.from_pretrained()
```

On the first use, the package automatically downloads:

* `stage1_best.pt` — Stage 1 cropland classifier
* `stage2_best.pt` — Stage 2 crop species classifier

The checkpoints are cached locally by `huggingface_hub`, so subsequent uses do not need to download them again.

You can also provide local checkpoints manually if desired:

```python
detector = CropSpeciesDetector.from_checkpoints(
    "path/to/stage1_best.pt",
    "path/to/stage2_best.pt",
)
```


## Usage

```python
from crop_species_detection import CropSpeciesDetector

detector = CropSpeciesDetector.from_checkpoints(
    "models/stage1_best.pt",
    "models/stage2_best.pt",
)

result = detector.predict("field.jpg")
print(result["final_prediction"], result["final_confidence"])

# Batch:
results = detector.predict_batch(["field1.jpg", "field2.jpg"])
```

`predict()` returns a dict with `stage1_prediction`, `stage1_confidence`,
`stage2_prediction`, `stage2_confidence`, `final_prediction`,
`final_confidence`, `stage2_used`, `stage1_probs`, and `stage2_probs`.

## Command line

```bash
crop-species-detect field1.jpg field2.jpg \
    --stage1 models/stage1_best.pt \
    --stage2 models/stage2_best.pt
```

Add `--json` for machine-readable output, `--device cpu|cuda` to pin the device.

## Package layout

```
crop_species_detection/
├── pyproject.toml
├── README.md
└── src/
    └── crop_species_detection/
        ├── __init__.py       # public API
        ├── model.py          # DINOv2Classifier architecture
        ├── pipeline.py       # CropSpeciesDetector (loading + inference)
        ├── cli.py            # `crop-species-detect` entry point
        └── constants.py      # class labels, preprocessing constants
```

## Performance (from the original demo)

| Metric            | Score  |
| ------------------ | ------ |
| Accuracy           | 93.72% |
| Balanced Accuracy   | 93.34% |
| Macro F1           | 93.18% |
| Weighted F1        | 93.70% |

See the upstream repo's README for full per-stage and per-class metrics.

## License

Code here is provided as a packaging of the original demo. Check the
licensing terms of the underlying DINOv2 model and training datasets before
commercial use or redistribution.
