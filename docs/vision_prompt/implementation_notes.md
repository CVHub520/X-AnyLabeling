# Implementation notes

## Phase 0-3 changes

| File | Reason | Upstream behavior changed? |
| --- | --- | --- |
| `.gitignore` | Exclude generated test output and supplied reference media | No |
| `tests/manual/yoloe_cross_image_demo.py` | Reproducible real reference-to-target VPE validation | No |
| `tests/manual/yoloe_xanylabeling_baseline.py` | Reproducible X-AnyLabeling Text and intra-image Visual baseline | No |
| `tests/manual/yoloe_phase2_integration.py` | Target B/C cache and mode-isolation integration test | No |
| `docs/vision_prompt/*` | Record environment, architecture, validation, plan, and tests | No |
| `anylabeling/services/auto_labeling/yoloe.py` | Add the production cross-image VPE lifecycle | Yes, adds an opt-in cached backend mode |
| `tests/test_models/test_yoloe.py` | Validate state, bbox/VPE compatibility, predictor reset, and clearing | No |
| `anylabeling/services/auto_labeling/model_manager.py` | Threaded VPE generation, status signal, guarded clearing | Yes, YOLOE-only additive API |
| `anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.py` | Connect existing panel and canvas marks to the new manager API | Yes, YOLOE-only controls |
| `anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui` | Add Generate, compact state, and Clear widgets | Yes, localized row additions |
| `tests/test_models/test_yoloe_model_manager.py` | Validate manager state/status/build/clear dispatch | No |
| `tests/manual/yoloe_phase3_gui_integration.py` | Real Qt worker and Target B/C button flow | No |
| `anylabeling/views/labeling/utils/batch.py` | Select Current Visual Prompt and reuse it through existing batch worker/save flow | Yes, localized YOLOE mode and per-image error isolation |
| `tests/test_auto_labeling/test_yoloe_batch.py` | Validate selection, progress, cancellation, error continuation, and folder start | No |
| `tests/manual/yoloe_phase4_batch_integration.py` | Real CUDA VPE batch over three valid and one damaged image | No |
| `anylabeling/services/auto_labeling/visual_prompt_profile.py` | Safe JSON/NPZ schema, integrity and compatibility validation | Additive public module |
| `tests/test_models/test_visual_prompt_profile.py` | Profile round-trip, corruption, version, dtype/shape, and compatibility tests | No |
| `tests/manual/yoloe_phase5_profile_integration.py` | Real CUDA save/clear/load and Target B/C validation | No |

The original annotation serialization and merge/replace rules remain in place.
`save_auto_labeling_result()` now returns success/failure so the existing batch
worker can avoid treating a failed write as a successful image.

## Implemented public model methods

- `build_visual_prompt(reference_image, boxes, classes)`
- `set_visual_prompt(vpe, classes, reference_metadata)`
- `clear_visual_prompt()`
- `has_visual_prompt()`
- `predict_with_visual_prompt(target_image)`
- `save_visual_prompt_profile(name, directory)`
- `load_visual_prompt_profile(path)`

State includes the live VPE, class names, reference metadata, readiness state,
and a model signature. The live VPE remains ready after target inference.
Temporary canvas marks are cleared only after successful generation.

## Profile persistence decision

Profiles use schema version 1 with `metadata.json` and `embedding.npz`.
Metadata includes name/time, configured model name, model signature, reference
path/image size/boxes/classes, VPE shape/dtype, fixed embedding filename, and
SHA-256. Files are written through same-directory temporary files and atomic
replacement, with metadata replaced last. Loading uses
`numpy.load(..., allow_pickle=False)` and a 64 MiB archive limit.

The panel adds only Save Prompt and Load Prompt actions. Save defaults below
the existing X-AnyLabeling work directory; Load uses the existing model worker
thread because restoring a cleared prompt may lazily reload the YOLOE model.
There is intentionally no profile database, rename/delete manager, multi-image
reference aggregation, or automatic VPE averaging in the MVP.

## Upstream synchronization risks

- `yoloe.py` is an active integration point and likely conflict area if
  upstream changes the THU-MIG fork/API.
- `auto_labeling.py/.ui` and `batch.py` are broad shared surfaces; additions
  must remain localized.
- `ModelManager` has a large explicit model loader and shared execution thread;
  new methods should avoid unrelated restructuring.

## Predictor and model isolation decision

The validated encoder sequence ends with a `YOLOEVPSegPredictor`; target
inference requires `model.predictor = None`. Setting a VPE also mutates model
classes. Phase 2 therefore implements a lazily loaded
`_cross_image_visual_model`. `_visual_model` continues serving legacy
intra-image prompts unchanged. Real integration verified that text and
prompt-free model object identities and results survive cross-image round trips.

Profiles serialize `vpe.detach().cpu().float().numpy()` to NPZ and plain
metadata to JSON. Loading restores a tensor on the current compatible model's
device. Pickle is not used for profiles. Reference path, boxes, and classes are
retained so an incompatible VPE can be regenerated.
