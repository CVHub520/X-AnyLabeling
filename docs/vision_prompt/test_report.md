# Phase 0/1 test report

Validated on 2026-08-24.

| Test | Result | Evidence |
| --- | --- | --- |
| Git/source baseline | PASS | branch tracks `upstream/main` at `8eb3fdcf` |
| X-AnyLabeling startup | PASS | Qt main window/event loop remained active in offscreen mode |
| CUDA runtime | PASS | real CUDA tensor calculation on RTX 5060 |
| Existing YOLOE unit/layout tests | PASS | 9 tests passed |
| Dependency consistency | PASS | `pip check`: no broken requirements |
| X-AnyLabeling Text Prompt | PASS | 4 `person` rectangle Shapes, scores 0.9276/0.9259/0.9127/0.6366 |
| X-AnyLabeling intra-image Visual Prompt | PASS | 3 `object0` rectangle Shapes, scores 0.9851/0.9165/0.8525 |
| Temporary mark consumption | PASS | marks count is zero after intra-image prediction |
| Cross-image VPE generation | PASS | real VPE `[1,1,512]`, float32, cuda:0 |
| Different target detection | PASS | 2 detections on `zidane.jpg` |
| Rendered result | PASS | `tests/output/cross_image_result.jpg`, 67,947 bytes |
| Phase 2 backend unit tests | PASS | 10 tests plus 10 parameterized subtests |
| Multiple same-class reference boxes | PASS | 2 boxes produced VPE `[1,1,512]` |
| Cached VPE on Target B/C | PASS | same tensor and model instance across both targets |
| Target predictor lifecycle | PASS | ordinary `SegmentationPredictor` after encoding |
| Text → Cross-Image → Text | PASS | text model reused; results unchanged |
| Prompt-Free → Cross-Image → Prompt-Free | PASS | prompt-free model reused; results unchanged |
| Phase 2 Target B | PASS | 2 editable Shapes, confidence 0.9025/0.7868 |
| Phase 2 Target C | PASS | 4 editable Shapes, confidence up to 0.9885 |
| Phase 3 ModelManager/UI tests | PASS | 20 tests plus 10 parameterized subtests in focused suite |
| Generate VPE QThread path | PASS | production panel button used existing model execution worker |
| Phase 3 Target B/C via Send | PASS | 2 then 4 editable Shapes using one cached VPE |
| Phase 3 panel state | PASS | marks-ready, VPE-ready, and cleared states displayed |
| Full application startup after UI change | PASS | main `pythonw` GUI process remained active after 8 seconds |
| Current Visual Prompt batch selection | PASS | explicit YOLOE mode, defaults to cached VPE when ready |
| Batch VPE lifecycle | PASS | one build and one set total; same tensor/model for every target |
| Three-image CUDA Batch | PASS | bus 4, zidane 2, composite 4 editable annotations |
| Batch progress | PASS | 1/4 through 4/4 including one damaged image |
| Batch cancel | PASS | worker stops before the next image and reports cancellation |
| Per-image error continuation | PASS | damaged image logged/skipped; later results retained |
| Existing annotation policy | PASS | manual `manual_keep` shape preserved and auto results appended |
| Profile unit suite | PASS | JSON/NPZ round-trip, validation, corruption and compatibility |
| Profile format safety | PASS | `allow_pickle=False`, SHA-256, fixed filename and 64 MiB limit |
| Real CUDA profile restore | PASS | exact VPE restored from CPU NPZ to cuda:0 |
| Loaded profile Target B/C | PASS | 2 then 4 editable Shapes; ordinary predictor |
| Phase 2–4 regression after persistence | PASS | mode isolation, Qt panel and 3-image Batch all rerun |
| Formatting check | PASS | Black and `git diff --check` |
| Repository Flake8 | BASELINE WARNINGS | 5 existing F841/C901 findings outside Phase 3 changes |

Commands used:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_auto_labeling/test_layout.py `
  tests/test_models/test_yoloe.py `
  tests/test_models/test_yoloe_model_manager.py -q

.\.venv\Scripts\python.exe tests/manual/yoloe_xanylabeling_baseline.py
.\.venv\Scripts\python.exe tests/manual/yoloe_cross_image_demo.py
.\.venv\Scripts\python.exe tests/manual/yoloe_phase2_integration.py
.\.venv\Scripts\python.exe tests/manual/yoloe_phase3_gui_integration.py
.\.venv\Scripts\python.exe tests/manual/yoloe_phase4_batch_integration.py
.\.venv\Scripts\python.exe tests/manual/yoloe_phase5_profile_integration.py
.\.venv\Scripts\python.exe -m pip check
```

Profile serialization and Batch integration are validated with the real CUDA
model. Generated evidence remains below ignored `tests/output/` directories.
