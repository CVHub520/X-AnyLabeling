# Phase 5 VisualPromptProfile validation

## Result

**Visual Prompt Persistence: PASS**

The production YOLOE backend generated a real VPE, saved it, cleared all
cross-image state, loaded it from disk, restored it to CUDA, and ran two
different target images.

| Field | Value |
| --- | --- |
| Profile | `product_A` |
| Files | `metadata.json` + `embedding.npz` |
| Schema | version 1 |
| Stored embedding | NumPy float32, `[1,1,512]` |
| Restored embedding | torch.float32, cuda:0, exact value match |
| Reference metadata | image path, size, two boxes, class retained |
| Target B | `zidane.jpg`, 2 Shapes |
| Target C | `bus_composite.jpg`, 4 Shapes |
| Target predictor | `SegmentationPredictor` |
| Profile load time | 0.0722 s |
| Result | PASS |

Unit coverage verifies valid round-trip, invalid names/metadata/reference
boxes, unsupported versions, checksum corruption, shape and dtype mismatch,
non-finite values, and incompatibility across model family/checkpoint size and
name/architecture/embedding dimension. NPZ loading explicitly disables
pickle. The live prompt is changed only after all format and compatibility
checks pass.

Generated evidence (ignored by Git):

- `tests/output/phase5_profile/phase5_profile_report.json`
- `tests/output/phase5_profile/visual_prompts/product_A/metadata.json`
- `tests/output/phase5_profile/visual_prompts/product_A/embedding.npz`
