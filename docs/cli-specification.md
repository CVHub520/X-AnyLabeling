# Headless inference CLI specification

## Status

Implemented in this checkout. This complements, rather than replaces, the existing
[CLI documentation](./en/cli.md). Current CLI commands cover GUI launch,
system/version/configuration information, and label-format conversion. In
particular, `convert --task xlabel2mask` exports masks from annotations that
already exist; it does not run a model. This implementation adds model inference to
the existing `xanylabeling` executable.

## User workflow

1. Find an installed or configured XAL model and its capabilities.
2. Run XAL inference over one image or a directory to create first-pass
   candidate features.
3. Open the native annotation JSON in the X-AnyLabeling GUI and review, split,
   merge, rename, or correct candidate shapes.
4. Use approved masks and metadata in a separate deterministic image-review
   tool. The XAL output remains candidate evidence until a person or agent
   reviews it.

## Command surface

The syntax and output contract below are implemented in this checkout.

### List configured models

```bash
xanylabeling models list [--task segmentation] [--json]
```

Lists built-in and valid custom model entries from XAL's existing model
registry/configuration. The optional task filter selects models capable of
producing segment shapes. Text, point/box, automatic, and class-based modes must
be reported individually; a model is listed only for modes its implementation
supports.

### Inspect or check a model

```bash
xanylabeling models info <model-id> [--json]
xanylabeling models check <model-id> [--device auto|cpu|gpu] [--json]
```

`info` returns registry/configuration metadata and supported prompt modes.
`check` verifies that the selected configuration can be resolved, required
runtime packages are importable, required local model assets are present (or
identifies which assets would need downloading), and the requested execution
provider is available. `check` must not download multi-gigabyte weights unless
the user explicitly requests download/prepare behavior.

### Generate candidate annotations and masks

```bash
xanylabeling segment \
  --input ./generated/fox-forest.png \
  --model groundingdino_swint_sam2_large-r20240806 \
  --prompt "fox" \
  --prompt "mushroom" \
  --output ./review-candidates \
  --json
```

`--input` accepts one supported image or a directory. For a directory, process
direct child images in stable, case-insensitive path order. Recursive traversal
is opt-in (`--recursive`). Each `--prompt` is a separate semantic query passed
through the selected XAL model's existing prompt API. A model that does not
support text prompting must reject `--prompt` with a capability error rather
than silently ignore it.

Additional optional inputs:

```bash
--marks-json ./marks.json
--model-config ./custom-model.yaml
--device auto|cpu|gpu
--confidence 0.35
--overwrite
--fail-fast
--json
```

`--marks-json` supplies point or rectangle marks only for models that support
XAL marks. `--model` selects a registered model ID; `--model-config` selects a
custom model configuration and is mutually exclusive with `--model`.
`--device` uses XAL's existing device selection (`auto` uses the existing
preference; `cpu` and `gpu` request CPU or GPU). A requested but unavailable
device fails clearly; it must not silently fall back to a different device.
`--confidence` is accepted only when the model exposes a compatible confidence
threshold; otherwise return a clear unsupported-option error. CLI values must
not silently override model-config semantics. `--overwrite` permits replacing
outputs; without it, an existing target is an error. `--fail-fast` stops a
directory run at the first inference error; default behavior continues and
reports every image result.

## Output contract

For each input image, write:

```text
<output>/
  annotations/<relative-image-path>.json
  candidates/<relative-image-path>.candidates.json
  masks/<relative-image-path>/candidate-0001.png
  masks/<relative-image-path>/candidate-0002.png
```

For example, an input `generated/scene/fox.png` produces
`annotations/scene/fox.png.json`, `candidates/scene/fox.png.candidates.json`,
and an image-specific mask directory. Preserve relative directories in batch
mode so equal basenames do not collide.

### Native annotation

`annotations/...json` uses the XLABEL JSON schema and fields expected by the
current GUI (`version`, `flags`, `shapes`, `imagePath`, `imageData`, dimensions,
and existing optional metadata conventions). It must be reopenable in the GUI.
Do not invent a second annotation schema. Store paths so the original image can
be resolved when the annotation is opened; the CLI must not rewrite the source
image. `imageData` should remain null by default; an explicit future option may
embed image bytes if a standalone annotation is needed.

### Binary feature masks

Each `candidate-NNNN.png` is a single-channel, same-width and same-height mask
for exactly one predicted shape: 0 outside, 255 inside. Masks retain their
candidate association and may overlap. Rasterize with the application's shape
utilities so polygons, rectangles, and other returned XAL geometry follow XAL
coordinate semantics. Do not combine multiple candidates into one indexed mask
as the only output; downstream feature review needs one independently editable
mask per candidate.

### Candidate metadata

`candidates/...candidates.json` has schema identifier
`xanylabeling.candidates.v1` and contains:

```json
{
  "schema": "xanylabeling.candidates.v1",
  "status": "unreviewed_candidates",
  "source": {
    "path": "../../generated/fox-forest.png",
    "sha256": "<sha256>",
    "width": 1024,
    "height": 1024
  },
  "application": {"name": "X-AnyLabeling", "version": "<version>"},
  "inference": {
    "model_id": "<id>",
    "model_type": "<type>",
    "config": "<registry id or custom config path>",
    "provider": "<actual provider or unknown>",
    "prompts": ["fox", "mushroom"],
    "parameters": {}
  },
  "candidates": [
    {
      "id": "candidate-0001",
      "label": "fox",
      "score": 0.91,
      "shape_type": "polygon",
      "points": [[12.0, 20.0], [30.0, 40.0]],
      "mask": "../../masks/scene/fox.png/candidate-0001.png",
      "status": "unreviewed"
    }
  ]
}
```

The actual point list contains the full geometry emitted by XAL. `score` and
`provider` are nullable when the model does not supply them. Record model/config
thresholds and the local weight checksum when it can be obtained without
unbounded work; use `null` with a reason if unavailable. The input SHA-256 is
required. Paths are relative to the metadata file and must resolve inside the
requested output directory except the source-image path, which points to the
input.

The schema deliberately has no `approved`, `ground_truth`, or quality-score
field. Downstream review creates its own approved feature manifest.

## Machine-readable output

When `--json` is supplied, stdout contains exactly one JSON document and no
progress bars, warnings, or log text. Diagnostics and progress go to stderr.

Single-image success:

```json
{
  "schema": "xanylabeling.cli-result.v1",
  "status": "success",
  "input": "<absolute path>",
  "outputs": {
    "annotation": "<absolute path>",
    "metadata": "<absolute path>",
    "masks": ["<absolute path>"]
  },
  "candidate_count": 1,
  "model_id": "<id>"
}
```

Directory runs return `status: success`, `partial`, or `failed`, with an ordered
`results` array containing an entry for every discovered input. Each item
includes `input`, `status`, output paths when successful, and a structured
`error` (`code`, `message`) when unsuccessful. Do not drop failed files from the
result.

Human-readable mode may display progress and a concise summary. It must convey
the same counts and failures as JSON mode.

## Errors and exit codes

| Exit code | Meaning |
| ---: | --- |
| `0` | Every requested image completed successfully. |
| `1` | One or more images failed or the batch was partial. |
| `2` | Invalid command arguments, unsupported model capability, invalid input, or output collision. |
| `3` | Model/runtime setup failure, such as missing dependency, unavailable provider, or failed weight resolution. |

Keep `argparse`'s standard usage-error behavior unless the existing application
has a stronger project-wide exit-code convention. In JSON mode, emit structured
errors to stdout only after argument parsing succeeds; argparse usage errors may
remain its standard stderr output.

## Model registry behavior

- Resolve built-in model IDs through `anylabeling/configs/models.yaml` and their
  bundled YAML resources.
- Resolve custom model entries through the same XAL user configuration and
  validation rules as the GUI.
- Do not accept arbitrary model code or import paths from CLI input.
- Present capabilities from the model implementation/registry, not by guessing
  from display names.
- Use the same local cache and download code as XAL when a model is actually
  loaded. Network access must be visible in diagnostics and must not be confused
  with local-only execution.

## Compatibility requirements

- Existing invocations (`xanylabeling`, GUI options, `help`, `checks`, `version`,
  `config`, `convert`, `train-worker`, and `training-worker`) retain their current
  behavior.
- New headless commands dispatch before GUI-only modules and do not initialize a
  visible window.
- `--filename`, `--output`, and other existing top-level GUI flags keep their
  current meanings. New command options are scoped to their subcommands to avoid
  argparse option collisions.
- `xanylabeling convert --task xlabel2mask` remains an annotation conversion
  operation; documentation must distinguish it from model inference.

## Acceptance criteria and validation status

1. **Implemented and unit-tested:** `models list/info/check` report registry-backed models and capabilities in
   text and JSON forms.
2. **Implemented; real-model run requires model assets:** A text-prompt segmentation model that is already supported by the installed
   XAL version runs through the same model class and prediction method used by
   the GUI, with no visible GUI window and no server requirement.
3. **Serialization-tested:** The produced XLABEL annotation reloads with XAL's `LabelFile`; opening it in the desktop GUI still needs a live GUI check.
4. **Implemented and unit-tested:** Each predicted shape has a separate dimension-matched binary PNG mask; its
   pixels correspond to that shape's XLABEL geometry.
5. **Implemented and unit-tested:** Metadata contains the source hash, XAL/model/config/runtime provenance, exact
   prompt or marks, scores when available, and an unreviewed status.
6. **Implemented:** Unsupported prompts/options fail explicitly. Missing weights/dependencies do
   not produce empty-success output.
7. **Implemented:** Directory processing preserves relative paths, is stable in ordering, and
   reports partial failures without losing results for completed images.
8. **Regression check pending:** Existing CLI commands remain in the parser; full GUI launch validation requires a display session.
