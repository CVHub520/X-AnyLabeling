# COCO RLE import

COCO polygon import accepts compressed and uncompressed RLE segmentations.
Install the optional decoder in the environment running X-AnyLabeling:

```bash
pip install pycocotools
```

Disconnected regions are imported as polygons with a shared group ID.
The conversion checks that filling the resulting polygons reproduces the
decoded mask exactly. Holes and thin regions that cannot be preserved by this
conversion cause the import to fail with the annotation and image IDs.
Missing decoder dependencies also cause an explicit failure.

All annotations are converted before any output label files are written, so
these failures leave existing annotations unchanged. Resolve the reported
annotation or dependency before retrying. This import does not silently
approximate unsupported regions.
