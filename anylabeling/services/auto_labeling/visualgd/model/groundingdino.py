# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List
import torchvision.transforms.functional as vis_F
from torchvision.transforms import InterpolationMode
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torchvision.ops import roi_align
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizerFast,
)
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

from ..util import box_ops, get_tokenlizer
from ..util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from ..util.utils import get_phrases_from_posmap
from ..util.visualizer import COCOVisualizer
from ..util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

from .matcher import build_matcher
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ..util.visualizer import renorm


def numpy_2_cv2(np_img):
    if np.min(np_img) < 0:
        raise Exception(
            "image min is less than 0. Img min: " + str(np.min(np_img))
        )
    if np.max(np_img) > 1:
        raise Exception(
            "image max is greater than 1. Img max: " + str(np.max(np_img))
        )
    np_img = (np_img * 255).astype(np.uint8)
    # Need to somehow ensure image is in RGB format. Note this line shows up in SAM demo: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_image = np.asarray(np_img)
    return cv2_image


def vis_exemps(image, exemp, f_name):
    plt.imshow(image)
    plt.gca().add_patch(
        Rectangle(
            (exemp[0], exemp[1]),
            exemp[2] - exemp[0],
            exemp[3] - exemp[1],
            edgecolor="red",
            facecolor="none",
            lw=1,
        )
    )
    plt.savefig(f_name)
    plt.close()


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # visual exemplar cropping
        self.feature_map_proj = nn.Conv2d(
            (256 + 512 + 1024), hidden_dim, kernel_size=1
        )

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(
            text_encoder_type
        )
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(
            self.bert.config.hidden_size, self.hidden_dim, bias=True
        )
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            backbone.num_channels[-1],
                            hidden_dim,
                            kernel_size=1,
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [
            _class_embed for i in range(transformer.num_decoder_layers)
        ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed
                )

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed
                )

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def add_exemplar_tokens(
        self, tokenized, text_dict, exemplar_tokens, labels
    ):
        input_ids = tokenized["input_ids"]

        device = input_ids.device
        new_input_ids = []
        encoded_text = text_dict["encoded_text"]
        new_encoded_text = []
        text_token_mask = text_dict["text_token_mask"]
        new_text_token_mask = []
        position_ids = text_dict["position_ids"]
        text_self_attention_masks = text_dict["text_self_attention_masks"]

        for sample_ind in range(len(labels)):
            label = labels[sample_ind][0]
            exemplars = exemplar_tokens[sample_ind]
            label_count = -1
            assert len(input_ids[sample_ind]) == len(position_ids[sample_ind])
            for token_ind in range(len(input_ids[sample_ind])):
                input_id = input_ids[sample_ind][token_ind]
                if (input_id not in self.specical_tokens) and (
                    token_ind == 0
                    or (
                        input_ids[sample_ind][token_ind - 1]
                        in self.specical_tokens
                    )
                ):
                    label_count += 1
                if label_count == label:
                    # Get the index where to insert the exemplar tokens.
                    ind_to_insert_exemplar = token_ind
                    while (
                        input_ids[sample_ind][ind_to_insert_exemplar]
                        not in self.specical_tokens
                    ):
                        ind_to_insert_exemplar += 1
                    break

            # Handle no text case.
            if label_count == -1:
                ind_to_insert_exemplar = 1
            # * token indicates exemplar.
            new_input_ids.append(
                torch.cat(
                    [
                        input_ids[sample_ind][:ind_to_insert_exemplar],
                        torch.tensor([1008] * exemplars.shape[0]).to(device),
                        input_ids[sample_ind][ind_to_insert_exemplar:],
                    ]
                )
            )
            new_encoded_text.append(
                torch.cat(
                    [
                        encoded_text[sample_ind][:ind_to_insert_exemplar, :],
                        exemplars,
                        encoded_text[sample_ind][ind_to_insert_exemplar:, :],
                    ]
                )
            )
            new_text_token_mask.append(
                torch.full((len(new_input_ids[sample_ind]),), True).to(device)
            )

        tokenized["input_ids"] = torch.stack(new_input_ids)

        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, None
        )

        return {
            "encoded_text": torch.stack(new_encoded_text),
            "text_token_mask": torch.stack(new_text_token_mask),
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

    def combine_features(self, features):
        (bs, c, h, w) = (
            features[0].decompose()[0].shape[-4],
            features[0].decompose()[0].shape[-3],
            features[0].decompose()[0].shape[-2],
            features[0].decompose()[0].shape[-1],
        )

        x = torch.cat(
            [
                F.interpolate(
                    feat.decompose()[0],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                for feat in features
            ],
            dim=1,
        )

        x = self.feature_map_proj(x)

        return x

    def forward(
        self,
        samples: NestedTensor,
        exemplar_images: NestedTensor,
        exemplars: List,
        labels,
        targets: List = None,
        cropped=False,
        orig_img=None,
        crop_width=0,
        crop_height=0,
        **kw,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # encoder texts

        tokenized = self.tokenizer(
            captions, padding="longest", return_tensors="pt"
        ).to(samples.device)

        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][
                :, : self.max_text_len
            ]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.max_text_len
            ]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(
            bert_output["last_hidden_state"]
        )  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if not cropped:
            features, poss = self.backbone(samples)
            features_exemp, _ = self.backbone(exemplar_images)
            combined_features = self.combine_features(features_exemp)
            # Get visual exemplar tokens.
            bs = len(exemplars)
            num_exemplars = exemplars[0].shape[0]
            if num_exemplars > 0:
                exemplar_tokens = (
                    roi_align(
                        combined_features,
                        boxes=exemplars,
                        output_size=(1, 1),
                        spatial_scale=(1 / 8),
                        aligned=True,
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                    .reshape(bs, num_exemplars, -1)
                )
            else:
                exemplar_tokens = None

        else:
            features, poss = self.backbone(samples)
            (h, w) = (
                samples.decompose()[0][0].shape[1],
                samples.decompose()[0][0].shape[2],
            )
            (orig_img_h, orig_img_w) = orig_img.shape[1], orig_img.shape[2]
            bs = len(samples.decompose()[0])

            exemp_imgs = []
            new_exemplars = []
            ind = 0
            for exemp in exemplars[0]:
                center_x = (exemp[0] + exemp[2]) / 2
                center_y = (exemp[1] + exemp[3]) / 2
                start_x = max(int(center_x - crop_width / 2), 0)
                end_x = min(int(center_x + crop_width / 2), orig_img_w)
                start_y = max(int(center_y - crop_height / 2), 0)
                end_y = min(int(center_y + crop_height / 2), orig_img_h)
                scale_x = w / (end_x - start_x)
                scale_y = h / (end_y - start_y)
                exemp_imgs.append(
                    vis_F.resize(
                        orig_img[:, start_y:end_y, start_x:end_x],
                        (h, w),
                        interpolation=InterpolationMode.BICUBIC,
                    )
                )
                new_exemplars.append(
                    [
                        (exemp[0] - start_x) * scale_x,
                        (exemp[1] - start_y) * scale_y,
                        (exemp[2] - start_x) * scale_x,
                        (exemp[3] - start_y) * scale_y,
                    ]
                )

                vis_exemps(
                    renorm(exemp_imgs[-1].cpu()).permute(1, 2, 0).numpy(),
                    [coord.item() for coord in new_exemplars[-1]],
                    str(ind) + ".jpg",
                )
                vis_exemps(
                    renorm(orig_img.cpu()).permute(1, 2, 0).numpy(),
                    [coord.item() for coord in exemplars[0][ind]],
                    "orig-" + str(ind) + ".jpg",
                )
                ind += 1

            exemp_imgs = nested_tensor_from_tensor_list(exemp_imgs)
            features_exemp, _ = self.backbone(exemp_imgs)
            combined_features = self.combine_features(features_exemp)
            new_exemplars = [
                torch.tensor(exemp).unsqueeze(0).to(samples.device)
                for exemp in new_exemplars
            ]

            # Get visual exemplar tokens.
            exemplar_tokens = (
                roi_align(
                    combined_features,
                    boxes=new_exemplars,
                    output_size=(1, 1),
                    spatial_scale=(1 / 8),
                    aligned=True,
                )
                .squeeze(-1)
                .squeeze(-1)
                .reshape(3, 256)
            )

            exemplar_tokens = torch.stack([exemplar_tokens] * bs)

        if exemplar_tokens is not None:
            text_dict = self.add_exemplar_tokens(
                tokenized, text_dict, exemplar_tokens, labels
            )

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs,
            masks,
            input_query_bbox,
            poss,
            input_query_label,
            attn_mask,
            text_dict,
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                layer_ref_sig
            )
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }

        # Used to calculate losses
        bs, len_td = text_dict["text_token_mask"].shape
        out["text_mask"] = torch.zeros(
            bs, self.max_text_len, dtype=torch.bool
        ).to(samples.device)
        for b in range(bs):
            for j in range(len_td):
                if text_dict["text_token_mask"][b][j] == True:
                    out["text_mask"][b][j] = True

        # for intermediate outputs
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord_list
            )
        out["token"] = one_hot_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(
                hs_enc[-1], text_dict
            )
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha, focal_gamma, losses):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(
            1
        )
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(
            src_boxes[:, :2], target_boxes[:, :2], reduction="none"
        )

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def token_sigmoid_binary_focal_loss(
        self, outputs, targets, indices, num_boxes
    ):
        pred_logits = outputs["pred_logits"]
        new_targets = outputs["one_hot"].to(pred_logits.device)
        text_mask = outputs["text_mask"]

        assert new_targets.dim() == 3
        assert pred_logits.dim() == 3  # batch x from x to

        bs, n, _ = pred_logits.shape
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(
                outputs["text_mask"].shape[0],
                -1,
                outputs["text_mask"].shape[1],
            )
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets = new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, new_targets, reduction="none"
        )
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos = 0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos, 1.0)
        loss = loss.sum() / num_pos_avg_per_gpu

        losses = {"loss_ce": loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.token_sigmoid_binary_focal_loss,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self, outputs, targets, cat_list, caption, return_indices=False
    ):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device = next(iter(outputs.values())).device
        one_hot = torch.zeros(
            outputs["pred_logits"].size(), dtype=torch.int64
        )  # torch.Size([bs, 900, 256])
        token = outputs["token"]

        label_map_list = []
        indices = []
        for j in range(len(cat_list)):  # bs
            label_map = []
            for i in range(len(cat_list[j])):
                label_id = torch.tensor([i])
                per_label = create_positive_map_exemplar(
                    token["input_ids"][j], label_id, [101, 102, 1012, 1029]
                )
                label_map.append(per_label)
            label_map = torch.stack(label_map, dim=0).squeeze(1)

            label_map_list.append(label_map)
        for j in range(len(cat_list)):  # bs
            for_match = {
                "pred_logits": outputs["pred_logits"][j].unsqueeze(0),
                "pred_boxes": outputs["pred_boxes"][j].unsqueeze(0),
            }

            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        # import pdb; pdb.set_trace()
        tgt_ids = [v["labels"].cpu() for v in targets]
        # len(tgt_ids) == bs
        for i in range(len(indices)):
            tgt_ids[i] = tgt_ids[i][indices[i][1]]
            one_hot[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(
                torch.long
            )
        outputs["one_hot"] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = []
                for j in range(len(cat_list)):  # bs
                    aux_output_single = {
                        "pred_logits": aux_outputs["pred_logits"][j].unsqueeze(
                            0
                        ),
                        "pred_boxes": aux_outputs["pred_boxes"][j].unsqueeze(
                            0
                        ),
                    }
                    inds = self.matcher(
                        aux_output_single, [targets[j]], label_map_list[j]
                    )
                    indices.extend(inds)
                one_hot_aux = torch.zeros(
                    outputs["pred_logits"].size(), dtype=torch.int64
                )
                tgt_ids = [v["labels"].cpu() for v in targets]
                for i in range(len(indices)):
                    tgt_ids[i] = tgt_ids[i][indices[i][1]]
                    one_hot_aux[i, indices[i][0]] = label_map_list[i][
                        tgt_ids[i]
                    ].to(torch.long)
                aux_outputs["one_hot"] = one_hot_aux
                aux_outputs["text_mask"] = outputs["text_mask"]
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                        **kwargs,
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = []
            for j in range(len(cat_list)):  # bs
                interm_output_single = {
                    "pred_logits": interm_outputs["pred_logits"][j].unsqueeze(
                        0
                    ),
                    "pred_boxes": interm_outputs["pred_boxes"][j].unsqueeze(0),
                }
                inds = self.matcher(
                    interm_output_single, [targets[j]], label_map_list[j]
                )
                indices.extend(inds)
            one_hot_aux = torch.zeros(
                outputs["pred_logits"].size(), dtype=torch.int64
            )
            tgt_ids = [v["labels"].cpu() for v in targets]
            for i in range(len(indices)):
                tgt_ids[i] = tgt_ids[i][indices[i][1]]
                one_hot_aux[i, indices[i][0]] = label_map_list[i][
                    tgt_ids[i]
                ].to(torch.long)
            interm_outputs["one_hot"] = one_hot_aux
            interm_outputs["text_mask"] = outputs["text_mask"]
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(
        self,
        num_select=100,
        text_encoder_type="text_encoder_type",
        nms_iou_threshold=-1,
        use_coco_eval=False,
        args=None,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO

            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item["name"] for item in category_dict]
        else:
            cat_list = args.label_list
        caption = " . ".join(cat_list) + " ."
        tokenized = self.tokenizer(
            caption, padding="longest", return_tensors="pt"
        )
        label_list = torch.arange(len(cat_list))
        pos_map = create_positive_map(tokenized, label_list, cat_list, caption)
        # build a mapping from label_id to pos_map
        if args.use_coco_eval:
            id_map = {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 7,
                7: 8,
                8: 9,
                9: 10,
                10: 11,
                11: 13,
                12: 14,
                13: 15,
                14: 16,
                15: 17,
                16: 18,
                17: 19,
                18: 20,
                19: 21,
                20: 22,
                21: 23,
                22: 24,
                23: 25,
                24: 27,
                25: 28,
                26: 31,
                27: 32,
                28: 33,
                29: 34,
                30: 35,
                31: 36,
                32: 37,
                33: 38,
                34: 39,
                35: 40,
                36: 41,
                37: 42,
                38: 43,
                39: 44,
                40: 46,
                41: 47,
                42: 48,
                43: 49,
                44: 50,
                45: 51,
                46: 52,
                47: 53,
                48: 54,
                49: 55,
                50: 56,
                51: 57,
                52: 58,
                53: 59,
                54: 60,
                55: 61,
                56: 62,
                57: 63,
                58: 64,
                59: 65,
                60: 67,
                61: 70,
                62: 72,
                63: 73,
                64: 74,
                65: 75,
                66: 76,
                67: 77,
                68: 78,
                69: 79,
                70: 80,
                71: 81,
                72: 82,
                73: 84,
                74: 85,
                75: 86,
                76: 87,
                77: 88,
                78: 89,
                79: 90,
            }
            new_pos_map = torch.zeros((91, 256))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map = new_pos_map

        self.nms_iou_threshold = nms_iou_threshold
        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind] = (
                    pos_maps[label_ind] / pos_maps[label_ind].sum()
                )

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(prob.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(
            topk_indexes, prob.shape[2], rounding_mode="trunc"
        )
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
        )

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]
        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
    }
    weight_dict["loss_giou"] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != "no":
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
            "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {
                k + f"_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
            }
        )
        weight_dict.update(interm_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality']
    losses = ["labels", "boxes"]

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        losses=losses,
    )
    criterion.to(device)
    postprocessors = {
        "bbox": PostProcess(
            num_select=args.num_select,
            text_encoder_type=args.text_encoder_type,
            nms_iou_threshold=args.nms_iou_threshold,
            args=args,
        )
    }

    return model, criterion, postprocessors


def create_positive_map(tokenized, tokens_positive, cat_list, caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        # except Exception as e:
        #     print("beg:", beg, "end:", end)
        #     print("token_positive:", tokens_positive)
        #     # print("beg_pos:", beg_pos, "end_pos:", end_pos)
        #     raise e
        # if beg_pos is None:
        #     try:
        #         beg_pos = tokenized.char_to_token(beg + 1)
        #         if beg_pos is None:
        #             beg_pos = tokenized.char_to_token(beg + 2)
        #     except:
        #         beg_pos = None
        # if end_pos is None:
        #     try:
        #         end_pos = tokenized.char_to_token(end - 2)
        #         if end_pos is None:
        #             end_pos = tokenized.char_to_token(end - 3)
        #     except:
        #         end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map


def create_positive_map_exemplar(input_ids, label, special_tokens):
    tokens_positive = torch.zeros(256, dtype=torch.float)
    count = -1
    for token_ind in range(len(input_ids)):
        input_id = input_ids[token_ind]
        if (input_id not in special_tokens) and (
            token_ind == 0 or (input_ids[token_ind - 1] in special_tokens)
        ):
            count += 1
        if count == label:
            ind_to_insert_ones = token_ind

            while input_ids[ind_to_insert_ones] not in special_tokens:
                tokens_positive[ind_to_insert_ones] = 1
                ind_to_insert_ones += 1
            break
    return tokens_positive
