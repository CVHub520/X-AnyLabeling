import json
import os.path as osp

import imgviz

from . import utils


def assert_labelfile_sanity(filename):
    assert osp.exists(filename)

    with open(filename) as f:
        data = json.load(f)

    assert "image_path" in data
    image_data = data.get("image_data", None)
    if image_data is None:
        parent_dir = osp.dirname(filename)
        img_file = osp.join(parent_dir, data["image_path"])
        assert osp.exists(img_file)
        img = imgviz.io.imread(img_file)
    else:
        img = utils.img_b64_to_arr(image_data)

    H, W = img.shape[:2]
    assert H == data["image_height"]
    assert W == data["image_width"]

    assert "shapes" in data
    for shape in data["shapes"]:
        assert "label" in shape
        assert "points" in shape
        for x, y in shape["points"]:
            assert 0 <= x <= W
            assert 0 <= y <= H
