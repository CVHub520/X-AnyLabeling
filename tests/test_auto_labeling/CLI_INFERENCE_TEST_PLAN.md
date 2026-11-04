# Headless inference CLI test plan

## Inventory

- `test_cli_inference.py`: unit and output-contract tests, approximately 8 cases.
- `test_cli_inference.py::TestCLIProcess`: subprocess tests for the registered
  `xanylabeling models` and `xanylabeling segment` command surfaces.

## Unit test plan

### Model catalog and capability reporting

- Load built-in registry entries without initializing a GUI or downloading
  weights.
- Confirm GroundingSAM2 is reported as text-prompt segmentation and ordinary
  segmentation models do not falsely claim text-prompt support.
- Confirm `models check` reports missing local weights as not ready and does not
  download them.
- Cover an invalid model ID and invalid custom model configuration.

### Input and marks validation

- Accept a supported image and deterministically order files in a directory.
- Reject missing paths, unsupported image extensions, malformed marks JSON,
  invalid mark coordinates, and unsupported model prompt modes.

### Output contract

- Run the CLI handler with a fake XAL model that returns an actual XAL `Shape`.
- Verify the annotation can be reopened by `LabelFile`, the binary PNG mask has
  the source dimensions and 0/255 values, and candidate metadata records source
  hash, model, prompt, score, weight provenance, and unreviewed status.
- Verify output collisions fail without `--overwrite` and can be replaced with
  `--overwrite`.
- Verify batch results preserve order and report failures/skipped files.

## Subprocess and realistic workflow scenarios

### Model discovery for a generated-image review

- Run `xanylabeling models list --task segmentation --json` through
  `python -m anylabeling.app`.
- Parse stdout as exactly one JSON document and confirm the known built-in
  GroundingSAM2 entry is present.

### Candidate segmentation output for agent review

- Generate a temporary image and use a fake model implementing the same
  `predict_shapes(image, filename, text_prompt=...)` interface as XAL models.
- Invoke the segment handler, then reopen its native XLABEL output and inspect
  every exported candidate mask and metadata reference.
- Real model inference is an environment-dependent integration gate requiring
  a model weight set and compatible execution provider; it is not simulated as
  a successful live-model run by this test suite.
