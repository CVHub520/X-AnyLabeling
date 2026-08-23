# Phase 2 backend validation

## Result

**Cross-Image backend state and model isolation: PASS**

The production `YOLOE` service was tested with the real YOLOE-11-S model on
CUDA. One reference image used two same-class person boxes. A single generated
VPE was then applied consecutively to Target B and Target C.

| Field | Value |
| --- | --- |
| Reference | `bus.jpg` |
| Reference boxes | 2, unified class `object` |
| VPE | `[1,1,512]`, float32, cuda:0 |
| Warm VPE generation | 0.2811 s |
| Target B | `zidane.jpg`, 2 Shapes, 0.9025 / 0.7868 |
| Target B inference | 0.1740 s |
| Target C | `bus_composite.jpg`, 4 Shapes, max 0.9885 |
| Target C inference | 0.0343 s |
| Target predictor | `SegmentationPredictor` |
| VPE regenerated between B/C | No |
| VPE tensor/model replaced between B/C | No |

Validated mode transitions:

```text
Text → Cross-Image → Text: PASS
Prompt-Free → Cross-Image → Prompt-Free: PASS
```

In both transitions, the original mode-specific model object was reused and
produced the same labels, boxes, and confidence values after returning from
Cross-Image mode. The dedicated cross-image model was cleared without deleting
or changing the other mode instances.

Generated evidence:

- `tests/output/phase2/target_b_result.jpg`
- `tests/output/phase2/target_c_result.jpg`
- `tests/output/phase2/phase2_report.json`

The generated output directory remains ignored by Git.
