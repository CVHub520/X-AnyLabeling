# Phase 4 Current Visual Prompt batch validation

## Result

**Cached VPE Batch Processing: PASS**

The production `BatchProcessingThread`, `ModelManager`, YOLOE backend, and
`save_auto_labeling_result()` pipeline were tested with a real CUDA model. The
VPE was built and set before the worker started, then reused without mutation
for all batch targets.

| Field | Value |
| --- | --- |
| Reference | `bus.jpg`, two person boxes |
| VPE | `[1,1,512]`, float32, cuda:0 |
| VPE build calls | 1 total |
| VPE set calls | 1 total |
| Valid image 1 | `01_bus.jpg`, 4 object annotations |
| Valid image 2 | `02_zidane.jpg`, 2 object annotations |
| Valid image 3 | `03_bus_composite.jpg`, 4 object annotations |
| Damaged input | `04_broken.jpg`, reported once and skipped |
| Progress | 1/4, 2/4, 3/4, 4/4 |
| Batch inference time | 0.4163 s |
| Target predictor | ordinary segmentation predictor |

The VPE tensor address and dedicated model identity stayed constant. No
`build_visual_prompt()` or `set_visual_prompt()` call occurred inside the
batch. A pre-existing `manual_keep` annotation on the second image remained
and automatic shapes were appended according to the existing preservation
policy.

The damaged image had a pre-existing sentinel JSON file. That file remained
byte-equivalent in meaning, proving a failed inference does not save an empty
result or overwrite existing annotations. Processing continued to completion.

Generated evidence (ignored by Git):

- `tests/output/phase4_batch/phase4_batch_report.json`
- `tests/output/phase4_batch/labels/01_bus.json`
- `tests/output/phase4_batch/labels/02_zidane.json`
- `tests/output/phase4_batch/labels/03_bus_composite.json`
- `tests/output/phase4_batch/labels/04_broken.json` (preserved sentinel)
