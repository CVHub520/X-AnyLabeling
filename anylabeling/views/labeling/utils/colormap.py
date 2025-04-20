import numpy as np


def label_colormap(n_label=256):
    """
    Generates a colormap with 'fresh' colors suitable for modern web UIs.

    Parameters
    ----------
    n_label: int
        Number of labels (default: 256).

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap. Index 0 is black (background).
    """
    # Base colors - "fresh" and suitable for modern UIs
    base_colors_rgb = [
        (170, 170, 255),
        (255, 170, 170),
        (170, 255, 170),
        (255, 255, 170),  # Pastels
        (170, 255, 255),
        (255, 170, 255),
        (85, 170, 255),
        (255, 170, 0),  # Bright/User examples
        (40, 255, 255),
        (0, 255, 127),
        (255, 105, 180),
        (127, 255, 212),  # Bright/Greens/Pinks
        (255, 215, 0),
        (100, 149, 237),
        (255, 182, 193),
        (64, 224, 208),  # Gold/Blues/Pinks
        (255, 223, 186),
        (147, 112, 219),
        (0, 191, 255),
        (240, 128, 128),  # Oranges/Purples/Blues
        (152, 251, 152),
        (173, 216, 230),
        (255, 192, 203),
        (221, 160, 221),  # Greens/Blues/Pinks
        (135, 206, 250),
        (255, 250, 205),
        (175, 238, 238),
        (250, 128, 114),  # Blues/Yellows/Corals
        (154, 205, 50),
        (32, 178, 170),
        (255, 160, 122),
        (176, 224, 230),  # Greens/Teals/Oranges
    ]

    # Initialize colormap with black for label 0 (background)
    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    num_base_colors = len(base_colors_rgb)

    # Assign colors starting from label 1, cycling through the base colors
    for i in range(1, n_label):
        color_index = (i - 1) % num_base_colors
        cmap[i] = base_colors_rgb[color_index]

    return cmap
