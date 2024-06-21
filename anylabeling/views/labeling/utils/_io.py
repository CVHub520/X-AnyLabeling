import os.path as osp
import re

import imgviz
import numpy as np
import PIL.Image


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return bool(re.search('[\u4e00-\u9fff]', str(s)))

def lblsave(filename, lbl):
    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            f"[{filename}] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format."
        )
