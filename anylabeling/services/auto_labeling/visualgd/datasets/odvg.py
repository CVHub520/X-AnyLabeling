from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))

import datasets.transforms as T


class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, "r") as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with open(anno, "r") as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert("RGB")
        exemplars = torch.tensor(meta["exemplars"], dtype=torch.int64)
        w, h = image.size
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            neg_labels = self.label_index.difference(pos_labels)

            vg_labels = list(pos_labels)
            num_to_add = min(
                len(neg_labels), self.max_labels - len(pos_labels)
            )
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))

            # shuffle
            for i in range(len(vg_labels) - 1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {
                item: index for index, item in enumerate(caption_list)
            }

            caption = " . ".join(caption_list) + " ."
            classes = [
                caption_dict[self.label_map[str(obj["label"])]]
                for obj in instances
            ]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = " . ".join(uni_caption_list) + " ."
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            caption_list = uni_caption_list
        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        target["exemplars"] = exemplars
        target["labels_uncropped"] = torch.clone(classes)
        # size, cap_list, caption, bboxes, labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # Check that transforms does not change the identity of target['labels'].
        if len(target["labels"]) > 0:
            assert target["labels"][0] == target["labels_uncropped"][0]
            print(
                "Asserted that transforms does not change the identity of target['labels']."
            )

        return image, target

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(
    image_set, fix_size=False, strong_aug=False, args=None
):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [
            int(i * data_aug_scale_overlap) for i in scales2_resize
        ]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == "train":
        if fix_size:
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ]
            )

        if strong_aug:
            import datasets.sltransform as SLT

            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales2_resize),
                                T.RandomSizeCrop(*scales2_crop),
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    SLT.RandomSelectMulti(
                        [
                            SLT.RandomCrop(),
                            SLT.LightingNoise(),
                            SLT.AdjustBrightness(2),
                            SLT.AdjustContrast(2),
                        ]
                    ),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(scales2_resize),
                            T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set in ["val", "eval_debug", "train_reg", "test"]:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == "INFO":
            print(
                "Under debug mode for flops calculation only!!!!!!!!!!!!!!!!"
            )
            return T.Compose(
                [
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = (
        datasetinfo["label_map"] if "label_map" in datasetinfo else None
    )
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = ODVGDataset(
        img_folder,
        ann_file,
        label_map,
        max_labels=args.max_labels,
        transforms=make_coco_transforms(
            image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args
        ),
    )
    return dataset


if __name__ == "__main__":
    dataset_vg = ODVGDataset(
        "path/GRIT-20M/data/",
        "path/GRIT-20M/anno/grit_odvg_10k.jsonl",
    )
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)]
    print(data)
    dataset_od = ODVGDataset(
        "pathl/V3Det/",
        "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
        "path/V3Det/annotations/v3det_label_map.json",
    )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)]
    print(data)
