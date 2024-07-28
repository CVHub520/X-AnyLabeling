# Tagging Annotation Example

Tagging annotation involves the task of recognizing and assigning labels to the elements within an image or shape.

## Tagging Model

X-AnyLabeling offers a range of tagging models for annotation, including [RAM](../../../anylabeling/configs/auto_labeling/ram_swin_large_14m.yaml) and [RAM++](../../../anylabeling/configs/auto_labeling/ram_plus_swin_large_14m.yaml). These models can be utilized for object tagging at the image-level or shape-level, and they can even be integrated with other models to create more intriguing pipelines like [GroundingSAM](../../../anylabeling/configs/auto_labeling/groundingdino_swinb_attn_fuse_sam_hq_vit_l_quant.yaml).

<img src=".data/ram_grounded_sam.jpg" width="100%" />

Here:

- **[RAM](https://arxiv.org/abs/2306.03514)** is a strong image tagging model, which can recognize any common category with high accuracy.
- **[RAM++](https://arxiv.org/abs/2310.15200)** is the next generation of RAM, which can recognize any category with high accuracy, including both predefined common categories and diverse open-set categories.

## Image-level Tagging

<img src=".data/annotated-image-level-tagging-data.png" width="100%" />

## Shape-level Tagging

<img src=".data/annotated-shape-level-tagging-data.png" width="100%" />

For detailed output examples, refer to [this folder](./sources/).
