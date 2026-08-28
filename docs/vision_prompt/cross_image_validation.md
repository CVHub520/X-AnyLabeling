# YOLOE cross-image validation

## Result

**Cross-Image Visual Prompt: PASS**

This validation used real YOLOE inference on two different files. No detection,
VPE, or output image was mocked.

| Field | Value |
| --- | --- |
| Model | YOLOE-11-S segmentation |
| Reference | THU-MIG/yoloe `ultralytics/assets/bus.jpg` |
| Reference bbox (xyxy) | `[221.52, 405.8, 344.98, 857.54]` (person) |
| Target | THU-MIG/yoloe `ultralytics/assets/zidane.jpg` |
| Reference equals target | No |
| Device | `cuda:0` |
| VPE shape | `[1, 1, 512]` |
| VPE dtype | `torch.float32` |
| Detection count | 2 |
| Confidences | `0.490147`, `0.209724` |
| VPE generation | 8.5782 s (includes first predictor setup/warm-up) |
| Target inference | 0.1750 s |
| Model load | 0.1451 s |

Target detections (xyxy):

1. `[750.727, 42.665, 1148.404, 716.746]`
2. `[123.550, 203.331, 1116.964, 716.951]`

Both rendered detections cover people in the target image. The output is
`tests/output/cross_image_result.jpg`; machine-readable metrics are beside it
as `cross_image_result.json`. `tests/output/` is ignored so generated evidence
is not accidentally committed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe tests\manual\yoloe_cross_image_demo.py
```

The script validates all paths, rejects identical reference and target paths,
checks bbox shape/finiteness/bounds, calls the current official `return_vpe`
API, records the actual predictor VPE, resets `model.predictor`, and performs
ordinary inference on the target. It returns a non-zero exit status when there
are no target detections.
