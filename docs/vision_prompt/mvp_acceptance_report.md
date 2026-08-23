# VisionPrompt Studio core MVP acceptance

Validated on 2026-08-24 against X-AnyLabeling 4.0.3 at `8eb3fdcf`, YOLOE-11s,
Ultralytics 8.3.39, and an RTX 5060 Laptop GPU.

## Acceptance result

**Core MVP: PASS**

| Required case | Result | Evidence |
| --- | --- | --- |
| A — Reference A finds targets in different Target B | PASS | real `[1,1,512]` VPE; `zidane.jpg` produced 2 editable Shapes |
| B — same prompt remains usable on Target C | PASS | no rebuild/set; `bus_composite.jpg` produced 4 editable Shapes |
| C — same prompt processes a folder | PASS | existing Batch worker processed 3 valid images with one VPE/model instance |
| D — Batch results are normal editable/savable annotations | PASS | existing JSON save/merge pipeline produced Shapes; manual annotation preservation passed |

The user also manually accepted Generate VPE, Target B (2), Target C (4),
Shape editing, and Clear Visual Prompt in the production GUI.

## Persistence extension

Phase 5 additionally passes save, clear, and load using versioned JSON/NPZ.
The exact embedding is restored to CUDA, Target B/C results remain 2/4, and
the target predictor is the ordinary segmentation predictor. Profiles retain
reference metadata so an incompatible VPE can be regenerated later.

## Regression and safety

- Text → Cross-Image → Text: PASS; text model reused.
- Prompt-Free → Cross-Image → Prompt-Free: PASS; prompt-free model reused.
- Batch progress, cancel, individual failure continuation, and annotation
  preservation: PASS.
- Profile corruption/version/model compatibility checks: PASS.
- Focused automated suite: 35 tests and 21 subtests passed.
- Real Phase 2, Phase 3, Phase 4, and Phase 5 integrations: PASS.

## Deliberately outside this MVP

Multi-reference aggregation, multi-class VPEs, a rename/delete Prompt Manager,
video/camera prompting, model export, branding, and UI redesign remain future
work. No unverified VPE averaging has been introduced.
