# Headless inference CLI architecture

## Status

Implemented in this checkout. This document describes an extension to the
existing `xanylabeling` application CLI. It does not introduce a separate
server, package, or replacement application.

## Goal

Allow scripts and coding agents to run X-AnyLabeling's installed annotation
models without opening the Qt desktop window, then save reviewable candidate
annotations, per-feature masks, and machine-readable metadata. A user can open
the resulting native annotation in X-AnyLabeling, correct it, and pass approved
masks and metadata to downstream tools such as `i-mage-not-ai`.

The CLI creates **proposals**. Model predictions are not human-approved feature
ground truth and are not an aesthetic or AI-origin judgment.

## Current application boundary

The application entry point is `anylabeling/app.py:main`. It uses `argparse`,
dispatches `help`, `checks`, `version`, `config`, training workers, and `convert`
before importing `MainWindow` and creating the Qt GUI. The model registry is
`anylabeling/configs/models.yaml`; model implementations live in
`anylabeling/services/auto_labeling/`, and GUI inference currently goes through
`ModelManager` and each model's `predict_shapes()` method. Annotation shapes are
represented by `Shape` and persisted in the XLABEL JSON format by `LabelFile`.
Existing `convert` commands convert annotations that already exist; they do not
run inference.

Text-conditioned segmentation models already registered by the application
include `grounding_sam2` and `segment_anything_3`. Other registered SAM and
detection models expose the modes their model implementation supports. The CLI
must discover these capabilities from the application registry/configuration;
it must not promise text prompting for models that do not implement it.

## Design decisions

1. **Extend the existing `xanylabeling` command.** Add subcommands to the
   existing `argparse` tree and dispatch them before any `QApplication`, window,
   theme, or GUI-only imports. Preserve all current launch options and existing
   commands.
2. **Run the installed application model code in-process.** The CLI calls the
   same model implementation and `predict_shapes()` path as the desktop app. It
   does not shell out to a second segmentation framework or require
   X-AnyLabeling-Server. A small inference service should isolate CLI lifecycle
   and I/O from GUI widgets; model construction and image preprocessing should
   remain shared with the GUI.
3. **Use a headless Qt core only where model code requires Qt types.** Set a
   supported offscreen platform before importing Qt, create a `QCoreApplication`
   when needed, and do not construct `MainWindow` or `QApplication` for
   inference. Keep this optional so non-inference commands continue working in
   environments without a display.
4. **Save native XLABEL proposals and explicit feature masks.** Write a native
   annotation JSON that the GUI can reopen. Also write one same-size binary PNG
   per predicted shape and a sidecar JSON manifest linking the masks to source
   image, prompt, model/config, scores, shape geometry, and runtime. Use the
   application's existing shape rasterization and annotation conventions where
   possible; do not resize or alter the source image.
5. **Keep evidence stages explicit.** Output metadata marks every shape as a
   model candidate, records the exact model/config and input hash, and never
   labels a candidate as reviewed or approved. Human edits saved from the GUI
   are a later stage.
6. **Treat model downloads and execution providers as application behavior.**
   Reuse XAL model configuration, weight resolution/download, and provider
   selection. Do not silently install dependencies or switch to a remote
   service. Fail with a specific actionable error when required weights,
   libraries, or providers are unavailable.
7. **Support one image and a folder workflow.** A folder command processes
   supported images in deterministic path order, writes independent outputs,
   and reports per-image success/failure. A failed image must not erase earlier
   outputs or be reported as successful.

## Proposed modules

| Component | Responsibility |
| --- | --- |
| `anylabeling/app.py` | Register subcommands and dispatch them before GUI startup. |
| `anylabeling/services/auto_labeling/inference.py` | Headless model lifecycle, capability checks, prediction, cleanup, and normalized result. Share model factories and common model setup with `ModelManager`; keep Qt UI signals/threads out of this service. |
| `anylabeling/services/auto_labeling/model_manager.py` | Retain desktop behavior while delegating shared model construction/configuration to the inference service or a common factory. |
| `anylabeling/views/common/cli_inference.py` | Parse CLI inputs, call inference service, write XLABEL, feature masks, and metadata; handle batch progress and exit status. |
| `anylabeling/views/labeling/label_file.py` and shape utilities | Reuse native annotation serialization and existing geometry-to-mask rasterization. |
| `anylabeling/app_info.py` | Document new commands and usage examples. |

The module names are proposed; maintainers may choose a different placement if
the public behavior and separation remain the same.

## Data flow

```text
CLI arguments
    -> validate image, model/config, prompt or marks, and output directory
    -> load model with XAL's existing config, weights, and provider logic
    -> call the model's predict_shapes(image, filename, text_prompt/marks)
    -> normalize each XAL Shape without losing label, score, geometry, or type
    -> save native XLABEL JSON + per-shape binary masks + candidate metadata
    -> emit human-readable summary or JSON result
```

The CLI must not launch a local HTTP server as an implicit intermediate. A
future remote-model option may use a separately configured server, but that is
outside this proposal's first milestone.

## Candidate model capabilities

At minimum the first milestone should cover the application's existing
text-prompt segmentation paths (`grounding_sam2` and `segment_anything_3`) when
their required weights and runtime dependencies are available. Point/rectangle
prompting should be exposed only for models whose XAL implementation supports
marks. Automatic or class-based segmentation can be added for other models
through the same service, with their exact capability names surfaced by
`models list` / `models info`.

Model-list metadata should distinguish:

- model identifier and display name;
- model type and config source;
- supported task and prompt modes;
- whether the model needs local weights, network access, or an API credential;
- whether its configuration is built-in or custom.

## Operational properties

- **Determinism:** the CLI is a deterministic invocation and serialization
  surface, not a guarantee of bit-identical GPU inference. Record the model
  version/config, weight hashes when locally available, XAL version, provider,
  prompt/marks, thresholds, and source SHA-256 so runs can be audited.
- **Isolation:** no GUI window, update check, telemetry change, or unrelated
  persistent application preference mutation should be required for inference.
- **Batch behavior:** stable input ordering, one result record per image,
  continue-on-error by default, and a nonzero aggregate exit status if any
  image failed. An optional fail-fast switch may stop at the first error.
- **Safety:** write outputs only under the requested output directory, reject
  ambiguous input/output collisions, and never overwrite existing artifacts
  unless an explicit `--overwrite` is provided.
- **Compatibility:** preserve the XLABEL schema expected by the installed GUI;
  add sidecar metadata without making the GUI depend on it.

## Out of scope

- Replacing the Qt application or adding an independent model stack.
- Declaring predicted features correct, semantically complete, or approved.
- Measuring visual quality, uncanniness, or “AI slop”.
- Requiring X-AnyLabeling-Server for local inference.
- Editing source images or silently repairing model outputs.
- Training models or exporting new model formats.

## Current implementation boundary

This checkout implements the command surface, catalog/capability inspection,
no-download readiness checks, direct in-process model construction and
prediction, candidate XLABEL serialization, per-shape masks, and provenance
metadata. The serialization tests reopen XLABEL through XAL's `LabelFile` and
check mask dimensions and pixel values. A real-weight inference run remains an
environment-dependent validation: model weights must be present and compatible
runtime providers must be installed. No model is downloaded by `models check`.
