from typing import Dict, List, Tuple, Union

from torch import Tensor
from mmdet.structures import SampleList, OptSampleList
from mmdet.models.detectors.rtdetr import RTDETR
from mmdet.registry import MODELS
import torch
import torch.nn.functional as F


@MODELS.register_module()
class HRW_RTDETR(RTDETR):
    def __init__(self, *args, patch_bs: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_bs = patch_bs
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses
 
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs, batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    
    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        assert len(batch_data_samples) == 1, "Only batch_size==1 is supported"
        windows = batch_data_samples[0].get('windows', None)
        patch_h, patch_w = batch_data_samples[0].get('patch_shape', None)
        bs = self.patch_bs
        img = batch_inputs[0]
        start = 0
        img_feats_per_level = None
        while True:
            patch_datas = []
            if (start + bs) > len(windows):
                end = len(windows)
            else:
                end = start + bs
            for window in windows[start:end]:
                data = img[:, window[1]:window[3], window[0]:window[2]]
                h, w = data.shape[-2:]
                pad_h = patch_h - h
                pad_w = patch_w - w
                if pad_h != 0 or pad_w != 0:
                    data = F.pad(data, (0, pad_w, 0, pad_h))
                patch_datas.append(data)
            patch_datas = torch.stack(patch_datas, dim=0)

            
            x = self.backbone(patch_datas)
            if self.with_neck:
                x = self.neck(x)
            if not isinstance(x, (list, tuple)):
                x = (x,)
            if img_feats_per_level is None:
                img_feats_per_level = [[] for _ in range(len(x))]
            for lvl, feat in enumerate(x):
                img_feats_per_level[lvl].append(feat)


            if end >= len(windows):
                break
            start += bs

        assert img_feats_per_level is not None
        img_feats = tuple(torch.cat(level_feats, dim=0)
                          for level_feats in img_feats_per_level)
        return img_feats

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict


    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare encoder/decoder inputs for patch-based features."""

        row_col_indices = None
        grid_shape = None

        if batch_data_samples is not None and len(batch_data_samples) == 1:
            sample = batch_data_samples[0]
            windows = sample.get('windows', None)
            patch_shape = sample.get('patch_shape', None)
            if windows is not None and patch_shape is not None and len(windows) > 0:
                patch_h, patch_w = patch_shape
                row_col_indices = []
                for idx, (x1, y1, _, _) in enumerate(windows):
                    row = int(y1) // patch_h
                    col = int(x1) // patch_w
                    row_col_indices.append((row, col, idx))
                rows = max(r for r, _, _ in row_col_indices) + 1
                cols = max(c for _, c, _ in row_col_indices) + 1
                assert rows * cols >= len(windows), \
                    f'推断网格 {rows}x{cols} 与窗口数量 {len(windows)} 不匹配'
                row_col_indices.sort()
                grid_shape = (rows, cols)

        if row_col_indices is None or grid_shape is None:
            # fallback to default流程：直接调用父类实现
            return super().pre_transformer(mlvl_feats, batch_data_samples)

        # 预先计算空间尺寸，用于 decoder
        rows, cols = grid_shape
        device = mlvl_feats[0].device
        spatial_shapes_list = []
        for feat_lvl in mlvl_feats:
            _, _, h_l, w_l = feat_lvl.shape
            spatial_shapes_list.append((rows * h_l, cols * w_l))
        spatial_shapes = torch.tensor(
            spatial_shapes_list, device=device, dtype=torch.long)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1]))

        encoder_inputs_dict = dict(
            patch_feats=mlvl_feats,
            row_col_indices=row_col_indices,
            grid_shape=grid_shape,
            spatial_shapes=spatial_shapes,
            chunk_size=self.patch_bs)
        decoder_inputs_dict = dict(
            memory_mask=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=None)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(
            self,
            patch_feats: Tuple[Tensor],
            row_col_indices: List[Tuple[int, int, int]],
            grid_shape: Tuple[int, int],
            spatial_shapes: Tensor,
            chunk_size: int = 0) -> Dict:
        """Encode patch features in mini-batches then stitch back to full maps."""
        if chunk_size <= 0:
            chunk_size = len(row_col_indices)

        num_levels = len(patch_feats)
        encoded_chunks: List[List[Tensor]] = [[] for _ in range(num_levels)]
        total = patch_feats[0].shape[0]

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = tuple(feat[start:end] for feat in patch_feats)
            encoded_chunk = self.encoder(chunk)
            for lvl, feat in enumerate(encoded_chunk):
                encoded_chunks[lvl].append(feat)

        encoded_patch_feats = []
        for lvl_chunks in encoded_chunks:
            if len(lvl_chunks) == 1:
                encoded_patch_feats.append(lvl_chunks[0])
            else:
                encoded_patch_feats.append(torch.cat(lvl_chunks, dim=0))

        rows, cols = grid_shape
        reassembled_feats: List[Tensor] = []
        for lvl, feat_lvl in enumerate(encoded_patch_feats):
            n, c, h_l, w_l = feat_lvl.shape
            assert n == len(row_col_indices), \
                f'编码后的特征数量 {n} 与窗口数量 {len(row_col_indices)} 不一致'
            canvas = feat_lvl.new_zeros((1, c, rows * h_l, cols * w_l))
            for row, col, idx in row_col_indices:
                top = row * h_l
                left = col * w_l
                canvas[:, :, top:top + h_l, left:left + w_l] = \
                    feat_lvl[idx:idx + 1]
            reassembled_feats.append(canvas)

        mlvl_feats = tuple(reassembled_feats)

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        memory = torch.cat(feat_flatten, 1)
        encoder_outputs_dict = dict(
            memory=memory, memory_mask=None, spatial_shapes=spatial_shapes)
        return encoder_outputs_dict