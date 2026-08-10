import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple, Union, Dict
from mmengine.structures import InstanceData
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models.detectors import RTDETR
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from mmdet.structures import SampleList, DetDataSample
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.models.layers.transformer.rtdetr_layers import RTDETRHybridEncoder
from mmdet.models.detectors.deformable_detr import DeformableDETR
from .lpm import LocalRankLoss
from .uhrdetr_layers import UHRDETRDecoder



@MODELS.register_module()
class UHR_DETR(RTDETR):
    def __init__(self,
                 *args,
                 globalNet_cfg: ConfigType,
                 top_k_patch: int,
                 top_k_patch_infer: int,
                 min_num_query: int = 300,
                 max_num_query: int = 3000,
                 num_query_expansion_ratio: float = 1.5,
                 patch_size: Tuple[int, int] = (512, 512),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.global_backbone = MODELS.build(globalNet_cfg.backbone)
        self.global_neck = MODELS.build(globalNet_cfg.neck)
        self.global_pre_process = nn.MaxPool2d(kernel_size=4, stride=4)

        sal_fusion_channels = 256
        self.map_head = self._build_head(
            sal_fusion_channels, sal_fusion_channels, 7)
        self.map_head_loss = MODELS.build(globalNet_cfg.map_head_loss)
        self.map_rank_loss = LocalRankLoss(margin=0.01, loss_weight=10.0)
        self.patch_size = patch_size
        self.top_k_patch = top_k_patch
        self.top_k_patch_infer = top_k_patch_infer
        self.num_query_expansion_ratio = num_query_expansion_ratio
        self.max_num_query = max_num_query
        self.min_num_query = min_num_query

        # Global feature projection for global cross-attention
        self.global_feat_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.global_feat_norm = nn.LayerNorm(self.embed_dims)


    def _init_layers(self) -> None:
        """Initialize layers."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.embed_dims = self.encoder.in_channels[0]
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        num_levels = len(self.encoder.in_channels)
        self.decoder = UHRDETRDecoder(
            num_layers=self.decoder['num_layers'],
            embed_dims=self.embed_dims,
            num_heads=self.decoder.get('num_heads', 8),
            num_levels=num_levels,
            num_points=self.decoder.get('num_points', 4),
            ffn_channels=self.decoder.get('ffn_channels', 1024),
            dropout=self.decoder.get('dropout', 0.0),
            with_checkpoint=self.decoder.get('with_checkpoint', False),
        )

    def init_weights(self) -> None:
        """Initialize weights for encoder, decoder and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.global_feat_proj.weight)

    def _build_head(self, in_channels: int, feat_channels: int,
                    out_channels: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        return layer

    def pre_transformer(self,
                        patches_features: Tuple[Tensor],
                        patch_coords: Tensor,
                        global_feat: Tensor,
                        batch_data_samples: SampleList = None) -> Tuple[Dict, Dict]:
        """Prepare encoder and decoder inputs.

        Args:
            patches_features: tuple of [B*K, C, H_l, W_l] from neck
            patch_coords: [B, K, 4] patch coordinates
            global_feat: [B, C, H_g, W_g] global features
            batch_data_samples: data samples

        Returns:
            encoder_inputs_dict: inputs for forward_encoder
            decoder_inputs_dict: partial inputs for forward_decoder
        """
        B = patch_coords.shape[0]
        K = patch_coords.shape[1]

        encoder_inputs_dict = dict(
            feat=patches_features,
            batch_size=B,
            num_patches=K,
        )

        decoder_inputs_dict = dict(
            patch_coords=patch_coords,
            global_feat=global_feat,
            batch_data_samples=batch_data_samples,
        )

        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tuple[Tensor], batch_size: int,
                        num_patches: int, **kwargs) -> Dict:
        """Forward with RTDETRHybridEncoder and flatten features.

        Args:
            feat: tuple of [B*K, C, H_l, W_l] from neck
            batch_size: B
            num_patches: K

        Returns:
            dict with 'local_memory', 'local_spatial_shapes', etc.
        """
        # Encoder forward
        patches_features = self.encoder(feat)

        # Flatten multi-scale features
        feat_flatten = []
        spatial_shapes_list = []
        for feat_level in patches_features:
            _, c, h, w = feat_level.shape
            spatial_shapes_list.append(
                torch.tensor([h, w], device=feat_level.device))
            feat_flatten.append(
                feat_level.flatten(2).permute(0, 2, 1))  # [B*K, h*w, C]

        local_memory = torch.cat(feat_flatten, 1)  # [B*K, sum(h*w), C]
        local_spatial_shapes = torch.stack(
            spatial_shapes_list)  # [num_levels, 2]
        local_level_start_index = torch.cat([
            local_spatial_shapes.new_zeros(1),
            local_spatial_shapes.prod(1).cumsum(0)[:-1]
        ]).long()

        encoder_outputs_dict = dict(
            local_memory=local_memory,
            local_spatial_shapes=local_spatial_shapes,
            local_level_start_index=local_level_start_index,
            batch_size=batch_size,
            num_patches=num_patches,
        )

        return encoder_outputs_dict

    def pre_decoder(self,
                    local_memory: Tensor,
                    local_spatial_shapes: Tensor,
                    batch_size: int,
                    num_patches: int,
                    patch_coords: Tensor,
                    global_feat: Tensor,
                    batch_data_samples: SampleList = None,
                    **kwargs) -> Tuple[Dict, Dict]:
        """Prepare decoder inputs by selecting top-k queries.

        Args:
            local_memory: [B*K, num_feat_pts, C]
            local_spatial_shapes: [num_levels, 2]
            batch_size: B
            num_patches: K
            patch_coords: [B, K, 4]
            global_feat: [B, C, H_g, W_g]
            batch_data_samples: optional

        Returns:
            decoder_inputs_dict: inputs for forward_decoder
            head_inputs_dict: inputs for bbox_head (encoder outputs)
        """
        B = batch_size
        K = num_patches
        num_feat_per_patch = local_memory.shape[1]

        # Score encoder features and generate proposals
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            local_memory, None, local_spatial_shapes)
        # output_proposals: [B*K, num_feat, 4] LOCAL inverse_sigmoid

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        # enc_outputs_coord_unact: [B*K, num_feat, 4] LOCAL inverse_sigmoid

        # Combine across patches per image
        enc_cls_combined = enc_outputs_class.reshape(
            B, K * num_feat_per_patch, -1)
        output_mem_combined = output_memory.reshape(
            B, K * num_feat_per_patch, -1)
        local_proposals_combined = enc_outputs_coord_unact.reshape(
            B, K * num_feat_per_patch, 4)

        # Track which patch each proposal comes from
        patch_origin_idx = torch.arange(
            K, device=patch_coords.device)[None, :, None].expand(
            B, K, num_feat_per_patch).reshape(B, K * num_feat_per_patch)

        # Dynamic query count
        # if self.training:
        if batch_data_samples[0].metainfo.get('cnt_map_pred', None) is not None:
            # Get cnt_map_pred from batch_data_samples metainfo
            cnt_map_pred = torch.tensor(
                [sample.metainfo.get('cnt_map_pred', self.num_queries)
                 for sample in batch_data_samples],
                device=local_memory.device)
            num_queries = int(
                cnt_map_pred.max().item() * self.num_query_expansion_ratio)
            num_queries = max(
                self.min_num_query,
                min(self.max_num_query, num_queries))
        else:
            num_queries = self.num_queries

        num_queries = min(num_queries, K * num_feat_per_patch)

        # Select top-scoring queries
        topk_indices = torch.topk(
            enc_cls_combined.max(-1)[0], k=num_queries, dim=1)[1]

        c = output_mem_combined.shape[-1]
        query = torch.gather(
            output_mem_combined, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, c))
        ref_unact_local = torch.gather(
            local_proposals_combined, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4))

        # Track which patch each selected query came from
        query_patch_idx = torch.gather(
            patch_origin_idx, 1, topk_indices)

        img_shape = batch_data_samples[0].metainfo['batch_input_shape']
        if self.training:
            cls_out_features = enc_cls_combined.shape[-1]
            topk_score = torch.gather(
                enc_cls_combined, 1,
                topk_indices.unsqueeze(-1).expand(-1, -1, cls_out_features))
            topk_coords_local = ref_unact_local.sigmoid()
            ref_unact_local = ref_unact_local.detach()

            topk_coords = UHRDETRDecoder.local_to_global_coords(
                topk_coords_local, patch_coords, query_patch_idx,
                img_shape).detach()
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()

            num_dn = dn_meta['num_denoising_queries']
            dn_bbox_query = dn_bbox_query.type_as(ref_unact_local)

            if num_dn > 0:
                # --- 1. Assign DN queries to patches based on bbox center ---
                dn_bbox_global = dn_bbox_query.sigmoid()  # [B, num_dn, 4]
                img_H, img_W = img_shape
                dn_cx_pixel = dn_bbox_global[..., 0] * img_W  # [B, num_dn]
                dn_cy_pixel = dn_bbox_global[..., 1] * img_H

                # Check which patch contains each DN query center
                # patch_coords: [B, K, 4] (x1, y1, x2, y2)
                pc_x1 = patch_coords[:, :, 0].unsqueeze(1)  # [B, 1, K]
                pc_y1 = patch_coords[:, :, 1].unsqueeze(1)
                pc_x2 = patch_coords[:, :, 2].unsqueeze(1)
                pc_y2 = patch_coords[:, :, 3].unsqueeze(1)

                dn_cx = dn_cx_pixel.unsqueeze(2)  # [B, num_dn, 1]
                dn_cy = dn_cy_pixel.unsqueeze(2)

                in_patch = ((dn_cx >= pc_x1) & (dn_cx < pc_x2) &
                            (dn_cy >= pc_y1) & (dn_cy < pc_y2))  # [B, num_dn, K]

                # For queries in at least one patch, take the first match
                # For queries not in any patch, assign to nearest patch center
                has_patch = in_patch.any(dim=2)  # [B, num_dn]
                dn_patch_idx = torch.zeros(
                    B, num_dn, dtype=torch.long, device=query.device)

                if has_patch.any():
                    # argmax on float gives first True index
                    dn_patch_idx[has_patch] = \
                        in_patch[has_patch].float().argmax(dim=-1)

                if (~has_patch).any():
                    # Fallback: assign to nearest patch center
                    patch_cx = ((patch_coords[:, :, 0] + patch_coords[:, :, 2])
                                / 2).unsqueeze(1)  # [B, 1, K]
                    patch_cy = ((patch_coords[:, :, 1] + patch_coords[:, :, 3])
                                / 2).unsqueeze(1)
                    dist = ((dn_cx - patch_cx) ** 2 +
                            (dn_cy - patch_cy) ** 2)  # [B, num_dn, K]
                    dn_patch_idx[~has_patch] = dist[~has_patch].argmin(dim=-1)

                # --- 2. Convert dn_bbox from global to local coordinates ---
                dn_bbox_local = UHRDETRDecoder.global_to_local_coords(
                    dn_bbox_global, patch_coords, dn_patch_idx, img_shape)
                # Clamp to valid range for inverse_sigmoid
                dn_bbox_local = dn_bbox_local.clamp(min=1e-3, max=1.0 - 1e-3)
                dn_bbox_unact_local = inverse_sigmoid(dn_bbox_local)

                # --- 3. Update query_patch_idx: prepend DN patch indices ---
                query_patch_idx = torch.cat(
                    [dn_patch_idx, query_patch_idx], dim=1)

                # --- 4. Concatenate DN and matching queries ---
                query = torch.cat([dn_label_query, query], dim=1)
                ref_unact_local = torch.cat(
                    [dn_bbox_unact_local, ref_unact_local], dim=1)

                # --- 5. Rebuild dn_mask with actual num_matching_queries ---
                # CdnQueryGenerator uses fixed num_matching_queries, but ours is
                # dynamic. Rebuild the mask with the actual count.
                num_matching = num_queries
                dn_mask = self._rebuild_dn_mask(dn_meta, num_matching,
                                                device=query.device)
            else:
                # No DN queries (e.g., no GT in batch)
                query = torch.cat([dn_label_query, query], dim=1)
                ref_unact_local = torch.cat(
                    [dn_bbox_query, ref_unact_local], dim=1)
                dn_mask, dn_meta = None, None
        else:
            ref_unact_local = ref_unact_local
            dn_mask, dn_meta = None, None

        # Prepare global features
        global_memory, global_key_pos, global_spatial_shape = self.prepare_global_features(
            global_feat)

        decoder_inputs_dict = dict(
            query=query,
            reference_points=ref_unact_local,
            local_memory=local_memory,
            local_spatial_shapes=local_spatial_shapes,
            global_memory=global_memory,
            global_key_pos=global_key_pos,
            global_spatial_shape=global_spatial_shape,
            patch_coords=patch_coords,
            query_patch_idx=query_patch_idx,
            img_shape=img_shape,
            self_attn_mask=dn_mask,
        )

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta,
        ) if self.training else dict()

        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        reference_points: Tensor,
                        local_memory: Tensor,
                        local_spatial_shapes: Tensor,
                        global_memory: Tensor,
                        global_key_pos: Tensor,
                        patch_coords: Tensor,
                        query_patch_idx: Tensor,
                        img_shape: Tuple[int, int],
                        local_level_start_index: Tensor = None,
                        self_attn_mask: Tensor = None,
                        global_spatial_shape: Tuple[int, int] = None,
                        **kwargs) -> Dict:
        """
        Args:
            query: [B, N, C]
            reference_points: [B, N, 4] LOCAL inverse_sigmoid
            local_memory: [B*K, num_feat_pts, C]
            local_spatial_shapes: [num_levels, 2]
            global_memory: [B, H_g*W_g, C]
            global_key_pos: [1, H_g*W_g, C]
            patch_coords: [B, K, 4]
            query_patch_idx: [B, N]
            img_shape: (img_H, img_W)
            local_level_start_index: [num_levels]

        Returns:
            dict with 'hidden_states' and 'references'
        """
        if local_level_start_index is None:
            local_level_start_index = torch.cat([
                local_spatial_shapes.new_zeros(1),
                local_spatial_shapes.prod(1).cumsum(0)[:-1]
            ]).long()

        all_cls, all_coords = self.decoder(
            query=query,
            reference_points=reference_points,
            local_memory=local_memory,
            local_spatial_shapes=local_spatial_shapes,
            local_level_start_index=local_level_start_index,
            global_memory=global_memory,
            global_key_pos=global_key_pos,
            patch_coords=patch_coords,
            query_patch_idx=query_patch_idx,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=self.bbox_head.cls_branches,
            self_attn_mask=self_attn_mask,
            img_shape=img_shape,
            eval_idx=self.eval_idx,
            global_spatial_shape=global_spatial_shape,
        )

        decoder_outputs_dict = dict(
            hidden_states=all_cls,
            references=all_coords,
        )

        return decoder_outputs_dict

    def forward_transformer(self,
                            patches_features: Tuple[Tensor],
                            patch_coords: Tensor,
                            global_feat: Tensor,
                            batch_data_samples: SampleList = None) -> Dict:
        """Forward process of Transformer.

        The forward procedure: pre_transformer -> forward_encoder -> 
        pre_decoder -> forward_decoder

        Args:
            patches_features: tuple of [B*K, C, H_l, W_l] from neck
            patch_coords: [B, K, 4] patch coordinates
            global_feat: [B, C, H_g, W_g] global features
            batch_data_samples: data samples

        Returns:
            head_inputs_dict: inputs for bbox_head
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            patches_features, patch_coords, global_feat, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, **decoder_inputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs: [B, C, img_H, img_W]
            batch_data_samples: data samples with gt_instances

        Returns:
            dict of losses
        """
        global_outputs_dict = self.global_forward_and_get_patches(
            batch_inputs, batch_data_samples)
        patches = global_outputs_dict['patches']
        patch_coords = global_outputs_dict['patch_coords']
        global_feat = global_outputs_dict['global_feat']

        filter_data_samples = self.filter_gt_instances(
            patch_coords, batch_data_samples)

        # 3. Extract patch features (backbone + neck)
        patches_features = self.extract_feat(
            patches)  # tuple of [B*K, C, H_l, W_l]

        # 4-8. Transformer pipeline
        head_inputs_dict = self.forward_transformer(
            patches_features, patch_coords, global_feat, filter_data_samples)

        # 9. Compute detection losses via bbox_head
        det_losses = self.bbox_head.loss(
            **head_inputs_dict,
            batch_data_samples=filter_data_samples,
        )

        # 10. Combine all losses
        losses = dict()
        losses['loss_map'] = global_outputs_dict['loss_map']
        losses['loss_map_ranking'] = global_outputs_dict['loss_map_ranking']
        losses.update(det_losses)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples.

        Args:
            batch_inputs: [B, C, img_H, img_W]
            batch_data_samples: data samples
            rescale: whether to rescale results to original image size

        Returns:
            batch_data_samples with pred_instances
        """
        global_outputs_dict = self.global_forward_and_get_patches(
            batch_inputs, batch_data_samples)
        patches = global_outputs_dict['patches']
        patch_coords = global_outputs_dict['patch_coords']
        global_feat = global_outputs_dict['global_feat']

        # 3. Extract patch features (backbone + neck)
        patches_features = self.extract_feat(
            patches)  # tuple of [B*K, C, H_l, W_l]

        # 4-8. Transformer pipeline
        head_inputs_dict = self.forward_transformer(
            patches_features, patch_coords, global_feat, batch_data_samples)

        # 9. Predict via bbox_head
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        # 10. Add predictions to data samples
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        return batch_data_samples

    def filter_gt_instances(self, patch_coords: Tensor,
                            batch_data_samples: SampleList) -> SampleList:
        """Filter GT instances for each image based on all its patches.

        For each GT box in the image, if it has at least 50% area inside 
        ANY of the patches, keep it. Otherwise, discard it.

        Args:
            patch_coords: [B, K, 4] patch coordinates in original image space (x1, y1, x2, y2)
            batch_data_samples: List of DetDataSample, length B

        Returns:
            filtered_data_samples: List of DetDataSample with filtered gt_instances, length B
        """
        B, K = patch_coords.shape[:2]

        filtered_data_samples = []

        for b in range(B):
            data_sample = batch_data_samples[b]
            gt_instances = data_sample.gt_instances
            # [N, 4] (x1, y1, x2, y2) in original image coords
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels  # [N]

            if gt_bboxes.shape[0] == 0:
                # No GT boxes, keep as is
                filtered_data_samples.append(data_sample)
                continue

            # Calculate original bbox areas
            orig_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            orig_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            orig_area = orig_w * orig_h  # [N]

            # For each GT box, check if it has >= 50% overlap with ANY patch
            max_iof_per_box = torch.zeros(
                gt_bboxes.shape[0], device=gt_bboxes.device)

            for k in range(K):
                # Get patch coordinates
                p_x1, p_y1, p_x2, p_y2 = patch_coords[b, k]  # pixel coords

                # Calculate intersection with this patch
                inter_x1 = torch.maximum(gt_bboxes[:, 0], p_x1)
                inter_y1 = torch.maximum(gt_bboxes[:, 1], p_y1)
                inter_x2 = torch.minimum(gt_bboxes[:, 2], p_x2)
                inter_y2 = torch.minimum(gt_bboxes[:, 3], p_y2)

                inter_w = (inter_x2 - inter_x1).clamp(min=0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0)
                inter_area = inter_w * inter_h  # [N]

                # Calculate IoF (Intersection over Foreground)
                iof = inter_area / (orig_area + 1e-6)  # [N]

                # Update max IoF for each box
                max_iof_per_box = torch.maximum(max_iof_per_box, iof)

            # Filter: keep boxes with max IoF >= 0.5 (at least 50% in some patch)
            valid_mask = max_iof_per_box >= 0.5

            # Create new data sample with filtered GT instances
            filtered_sample = DetDataSample()
            filtered_sample.set_metainfo(data_sample.metainfo.copy())

            # Copy all other fields from original sample
            if hasattr(data_sample, 'pred_instances'):
                filtered_sample.pred_instances = data_sample.pred_instances

            # Create filtered gt_instances (keep in original image coordinates)
            filtered_gt_instances = InstanceData()
            filtered_gt_instances.bboxes = gt_bboxes[valid_mask]
            filtered_gt_instances.labels = gt_labels[valid_mask]
            filtered_sample.gt_instances = filtered_gt_instances

            filtered_data_samples.append(filtered_sample)

        return filtered_data_samples

    def _rebuild_dn_mask(self, dn_meta, num_matching, device):
        """Rebuild DN attention mask with the actual num_matching_queries.

        CdnQueryGenerator builds the mask with a fixed num_matching_queries,
        but UHRDETR has dynamic query counts. This method rebuilds the mask
        with the correct dimensions.

        Args:
            dn_meta: dict with 'num_denoising_queries' and 'num_denoising_groups'
            num_matching: actual number of matching queries
            device: torch device

        Returns:
            attn_mask: [num_total, num_total] bool tensor
        """
        num_dn = dn_meta['num_denoising_queries']
        num_groups = dn_meta['num_denoising_groups']
        max_num_target = num_dn // (2 * num_groups) if num_groups > 0 else 0
        num_total = num_dn + num_matching

        attn_mask = torch.zeros(
            num_total, num_total, device=device, dtype=torch.bool)
        # Matching part cannot see DN part
        attn_mask[num_dn:, :num_dn] = True
        # DN groups cannot see each other
        for i in range(num_groups):
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1), num_dn)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask

    def global_forward_and_get_patches(self,
                                       batch_inputs: Tensor,
                                       batch_data_samples: SampleList):
        # 1. Global network forward
        globalNet_outputs_dict = self.globalNet_forward(batch_inputs,
                                                        batch_data_samples)
        cnt_map_pred = globalNet_outputs_dict['cnt_map_pred']

        # Store cnt_map_pred in metainfo for dynamic query selection
        for i, sample in enumerate(batch_data_samples):
            sample.set_metainfo({'cnt_map_pred': cnt_map_pred[i].item()})

        # 2. Get patches
        # patches, patch_coords, topk_scores = self.get_patches(
        #     batch_inputs, globalNet_outputs_dict['map_pred'])
        patches, patch_coords, topk_scores = self.get_patches_fast(
            batch_inputs, globalNet_outputs_dict['map_pred'])

        # if True:
        #     visualize_map_pred(
        #         globalNet_outputs_dict['map_pred'], save_path='heatmap.jpg', batch_idx=0)
        # visualize_patches(batch_inputs, patch_coords,
        #                       save_path='patches_vis.jpg', batch_idx=0, draw_boxes=True, dark_ratio=0.5, batch_data_samples=batch_data_samples)
            # cal_coverage_rate(batch_data_samples, patch_coords, savepath='coverage_rate_soda_a_k100.pkl')
            # cal_coverage_rate_per_k(batch_data_samples, patch_coords, savepath='coverage_per_k_soda_a_k100.pkl')

        # patches: [B, K, C, pH, pW], patch_coords: [B, K, 4]
        # K = patches.shape[1]
        patches = patches.reshape(-1, *patches.shape[-3:])  # [B*K, C, pH, pW]

        global_outputs_dict = dict(
            global_feat=globalNet_outputs_dict['global_feat'],
            patches=patches,
            patch_coords=patch_coords,
            topk_scores=topk_scores,
            loss_map=globalNet_outputs_dict['loss_map'] if self.training else None,
            loss_map_ranking=globalNet_outputs_dict['loss_map_ranking'] if self.training else None,
        )
        return global_outputs_dict

    def prepare_global_features(self, global_feat):
        """Prepare global features for global cross-attention.

        Args:
            global_feat: [B, C, H_g, W_g] from global neck

        Returns:
            global_memory: [B, H_g*W_g, C] projected features
            global_key_pos: [1, H_g*W_g, C] positional encoding
            global_spatial_shape: (H_g, W_g)
        """
        B, C, H_g, W_g = global_feat.shape

        # Flatten and project
        global_memory = global_feat.flatten(
            2).permute(0, 2, 1)  # [B, H_g*W_g, C]
        global_memory = self.global_feat_proj(global_memory)
        global_memory = self.global_feat_norm(global_memory)

        # 2D sincos positional encoding (reuse RTDETRHybridEncoder's method)
        global_key_pos = RTDETRHybridEncoder.build_2d_sincos_position_embedding(
            W_g, H_g, C, device=global_feat.device)
        # [1, H_g*W_g, C]

        return global_memory, global_key_pos, (H_g, W_g)

    def extract_global_feat(self, batch_inputs: Tensor):
        x = self.global_pre_process(batch_inputs)
        x = self.global_backbone(x)
        x = self.global_neck(x)

        return x

    def globalNet_forward(self, batch_inputs: Tensor,
                          batch_data_samples: SampleList):
        global_features = self.extract_global_feat(batch_inputs)
        global_feat = global_features[2]
        # global_feat = global_features[-1]
        map_pred_logits = self.map_head(global_feat)
        map_pred_probs = torch.softmax(map_pred_logits, dim=1)
        bins = torch.arange(7, device=batch_inputs.device).view(
            1, 7, 1, 1).float()
        map_pred = (map_pred_probs * bins).sum(dim=1, keepdim=True)
        cnt_map_pred = torch.sum(map_pred, dim=[1, 2, 3]).detach()
        # cnt_map_pred = torch.sum(global_feat.sigmoid(), dim=[1, 2, 3]).detach()
        map_target = None
        if self.training:
            # if True:
            batch_gt_instances = []
            batch_img_metas = []
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
                batch_gt_instances.append(data_sample.gt_instances)
            loss_map, map_target = self.loss_map_head(
                map_pred_logits, batch_gt_instances, batch_img_metas
            )
            loss_map_ranking = self.map_rank_loss(map_pred, map_target)

        globalNet_outputs_dict = dict(
            global_feat=global_feat,
            map_pred=map_pred,
            cnt_map_pred=cnt_map_pred,
            map_target=map_target,
            loss_map=None if not self.training else loss_map,
            loss_map_ranking=None if not self.training else loss_map_ranking,
        )
        return globalNet_outputs_dict

    def loss_map_head(self, map_pred, batch_gt_instances, batch_img_metas):
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        map_target = self.generate_iof_gain_map(
            gt_bboxes, map_pred.shape[-2:], img_shape)

        map_target = torch.sqrt(map_target)
        map_target_for_loss = map_target.clamp(min=0.0, max=6.0-1e-6)

        loss_map = self.map_head_loss(map_pred, map_target_for_loss.squeeze(1))

        return loss_map, map_target

    def generate_iof_gain_map(self, batch_gt_bboxes, feat_shape, img_shape):
        """生成整个 batch 的收益图。"""
        batch_gain_maps = [
            self.generate_iof_gain_map_single(
                gt_bboxes, feat_shape, img_shape, self.patch_size)
            for gt_bboxes in batch_gt_bboxes
        ]
        return torch.stack(batch_gain_maps, dim=0)

    def generate_iof_gain_map_single(self, gt_bboxes, feat_shape, img_shape, patch_size):
        """
        生成基于 IoF 贡献的收益图 (Gain Map)

        Args:
            gt_bboxes: [N, 4] Tensor (x1, y1, x2, y2) 原图尺度
            feat_shape: (H, W) 特征图尺寸
            img_shape: (img_H, img_W) 原图尺寸
            patch_size: (pH, pW) Patch 的实际像素尺寸

        Returns:
            gain_map: [1, H, W] 每个点的值 = Σ(IoF of GTs covered by this patch)
        """
        device = gt_bboxes.device
        N = gt_bboxes.shape[0]
        H, W = feat_shape
        img_H, img_W = img_shape
        pH, pW = patch_size

        if N == 0:
            return torch.zeros((1, H, W), device=device)

        # 1. 生成所有潜在 Patch 中心的坐标 (原图尺度)
        stride_h = img_H / H
        stride_w = img_W / W

        # 构造网格中心
        # shift_x: [W], shift_y: [H]
        shift_x = (torch.arange(0, W, device=device) + 0.5) * stride_w
        shift_y = (torch.arange(0, H, device=device) + 0.5) * stride_h
        # grid_y, grid_x: [H, W]
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')

        # 展平以便广播计算: [M, 1] 其中 M = H*W
        centers_x = grid_x.reshape(-1)
        centers_y = grid_y.reshape(-1)

        # 2. 计算每个中心点对应的 Patch 框 [M, 4]
        patch_x1 = centers_x - pW / 2
        patch_y1 = centers_y - pH / 2
        patch_x2 = centers_x + pW / 2
        patch_y2 = centers_y + pH / 2

        # 3. 广播计算 Patch 与 GT 的 Intersection (交集面积)
        # patch: [M, 1, 4] vs gt: [1, N, 4] -> intersection: [M, N]

        # 左上角 max
        lt_x = torch.maximum(patch_x1[:, None], gt_bboxes[None, :, 0])
        lt_y = torch.maximum(patch_y1[:, None], gt_bboxes[None, :, 1])
        # 右下角 min
        rb_x = torch.minimum(patch_x2[:, None], gt_bboxes[None, :, 2])
        rb_y = torch.minimum(patch_y2[:, None], gt_bboxes[None, :, 3])

        # 宽高 clamp 0
        inter_w = (rb_x - lt_x).clamp(min=0)
        inter_h = (rb_y - lt_y).clamp(min=0)
        inter_area = inter_w * inter_h  # [M, N]

        # 4. 计算 GT 自身面积
        gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * \
            (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_area = gt_area[None, :].clamp(min=1e-6)  # [1, N]

        # 5. 计算 IoF (Intersection over Foreground)
        # 贡献值：如果 GT 完全在 Patch 内，iof=1；一半在内，iof=0.5
        iof = inter_area / gt_area  # [M, N]

        # 6. 聚合得到 Map (求和)
        gain_flat = iof.sum(dim=1)  # [M]

        gain_map = gain_flat.reshape(H, W)
        # gain_map = torch.sqrt(gain_map)
        # gain_map = gain_map.clamp(min=0.0, max=6.0-1e-6)

        return gain_map[None, :, :]

    def get_patches(self, batch_inputs, map_pred):
        """
        从batch图像中裁剪出topk个patch。

        Args:
            batch_inputs: [B, C, img_H, img_W] 原图
            map_pred: [B, 1, H, W] 预测的收益图

        Returns:
            patches: [B, K, C, pH, pW] 裁剪的patch
            patch_coords: [B, K, 4] 原图坐标 (x1, y1, x2, y2)
        """
        B, C, img_H, img_W = batch_inputs.shape
        if self.training:
            topk_coords, topk_scores = self.select_patches_with_subtraction_batch(
                map_pred, patch_size=self.patch_size, img_shape=(img_H, img_W), k=self.top_k_patch)
        else:
            topk_coords, topk_scores = self.select_patches_with_subtraction_batch(
                map_pred, patch_size=self.patch_size, img_shape=(img_H, img_W), k=self.top_k_patch_infer)

        pH, pW = self.patch_size
        device = batch_inputs.device
        K = topk_coords.shape[1]

        # 1. 特征图坐标 -> 原图坐标 (中心点)
        # topk_coords: [B, K, 2] (y, x) 特征图坐标
        # 需要计算原图尺度下的中心点
        feat_H, feat_W = map_pred.shape[2], map_pred.shape[3]
        stride_h = img_H / feat_H
        stride_w = img_W / feat_W

        # 中心点坐标 (原图尺度)
        center_y = (topk_coords[..., 0] + 0.5) * stride_h  # [B, K]
        center_x = (topk_coords[..., 1] + 0.5) * stride_w  # [B, K]

        # 2. 构建 grid_sample 所需的采样网格
        # grid_sample 要求坐标归一化到 [-1, 1]
        # 每个 patch 内部构建相对坐标网格

        # patch 内的相对坐标: [pH, pW]
        # y_offset: [-pH/2, pH/2], x_offset: [-pW/2, pW/2]
        y_offset = torch.arange(
            pH, device=device).float() - pH / 2 + 0.5  # [pH]
        x_offset = torch.arange(
            pW, device=device).float() - pW / 2 + 0.5  # [pW]

        # 构建网格: [pH, pW, 2]
        grid_y, grid_x = torch.meshgrid(y_offset, x_offset, indexing='ij')
        # grid_y: [pH, pW], grid_x: [pH, pW]

        # 3. 扩展维度进行广播
        # center_y/center_x: [B, K] -> [B, K, 1, 1]
        center_y = center_y.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        center_x = center_x.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]

        # grid_y/grid_x: [pH, pW] -> [1, 1, pH, pW]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, pH, pW]
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, pH, pW]

        # 计算采样点的绝对坐标 (原图尺度)
        sample_y = center_y + grid_y  # [B, K, pH, pW]
        sample_x = center_x + grid_x  # [B, K, pH, pW]

        # 4. 归一化到 [-1, 1]
        sample_y = 2.0 * sample_y / img_H - 1.0
        sample_x = 2.0 * sample_x / img_W - 1.0

        # 组合网格: [B, K, pH, pW, 2]
        sample_grid = torch.stack([sample_x, sample_y], dim=-1)

        # 5. 使用 grid_sample 进行批量裁剪
        # 需要将 batch_inputs 从 [B, C, H, W] 扩展到 [B*K, C, H, W]
        # sample_grid 从 [B, K, pH, pW, 2] 变为 [B*K, pH, pW, 2]

        # 扩展输入图像: [B, C, H, W] -> [B, K, C, H, W] -> [B*K, C, H, W]
        batch_inputs_expanded = batch_inputs.unsqueeze(
            1).expand(-1, K, -1, -1, -1)
        batch_inputs_flat = batch_inputs_expanded.reshape(
            B * K, C, img_H, img_W)

        # 展平网格: [B, K, pH, pW, 2] -> [B*K, pH, pW, 2]
        sample_grid_flat = sample_grid.reshape(B * K, pH, pW, 2)

        # 执行采样
        patches_flat = F.grid_sample(
            batch_inputs_flat,
            sample_grid_flat,
            mode='bilinear',
            padding_mode='zeros',  # 边界外填充0
            align_corners=False
        )  # [B*K, C, pH, pW]

        # 恢复形状: [B*K, C, pH, pW] -> [B, K, C, pH, pW]
        patches = patches_flat.reshape(B, K, C, pH, pW)

        # 6. 计算patch在原图上的坐标 [B, K, 4] (x1, y1, x2, y2)
        # 中心点坐标
        center_y_orig = (topk_coords[..., 0] + 0.5) * stride_h  # [B, K]
        center_x_orig = (topk_coords[..., 1] + 0.5) * stride_w  # [B, K]

        patch_x1 = (center_x_orig - pW / 2).clamp(min=0, max=img_W)
        patch_y1 = (center_y_orig - pH / 2).clamp(min=0, max=img_H)
        patch_x2 = (center_x_orig + pW / 2).clamp(min=0, max=img_W)
        patch_y2 = (center_y_orig + pH / 2).clamp(min=0, max=img_H)

        patch_coords = torch.stack(
            [patch_x1, patch_y1, patch_x2, patch_y2], dim=-1)  # [B, K, 4]

        return patches, patch_coords, topk_scores

    def get_patches_fast(self, batch_inputs, map_pred):
        """Fast patch extraction using integer indexing.

        与get_patches输入输出完全一致，但假设topk_coords是整数坐标，
        直接用索引切片提取patches，比grid_sample更快且无插值误差。

        Args:
            batch_inputs: [B, C, img_H, img_W] 原图
            map_pred: [B, 1, H, W] 预测的收益图（仅用于获取topk坐标）

        Returns:
            patches: [B, K, C, pH, pW] 裁剪的patch
            patch_coords: [B, K, 4] 原图坐标 (x1, y1, x2, y2)
            topk_scores: [B, K] topk分数
        """
        B, C, img_H, img_W = batch_inputs.shape
        # 获取topk坐标和分数（与get_patches相同）
        if self.training:
            topk_coords, topk_scores = self.select_patches_with_subtraction_batch(
                map_pred, patch_size=self.patch_size, img_shape=(img_H, img_W), k=self.top_k_patch)
        else:
            topk_coords, topk_scores = self.select_patches_with_subtraction_batch(
                map_pred, patch_size=self.patch_size, img_shape=(img_H, img_W), k=self.top_k_patch_infer)

        pH, pW = self.patch_size
        device = batch_inputs.device
        K = topk_coords.shape[1]

        # 特征图坐标 -> 原图像素坐标（整数）
        # topk_coords: [B, K, 2] (y, x) 特征图坐标
        feat_H, feat_W = map_pred.shape[2], map_pred.shape[3]
        stride_h = img_H / feat_H
        stride_w = img_W / feat_W

        # 计算patch左上角坐标（原图尺度，整数）
        # 中心点 -> 左上角
        center_y = (topk_coords[..., 0] + 0.5) * stride_h  # [B, K]
        center_x = (topk_coords[..., 1] + 0.5) * stride_w  # [B, K]
        y1_all = (center_y - pH / 2).long()  # [B, K]
        x1_all = (center_x - pW / 2).long()  # [B, K]

        # 计算右下角坐标
        y2_all = y1_all + pH
        x2_all = x1_all + pW

        # 计算padding量（处理越界）
        pad_top = int((-y1_all).clamp(min=0).max().item())
        pad_left = int((-x1_all).clamp(min=0).max().item())
        pad_bottom = int((y2_all - img_H).clamp(min=0).max().item())
        pad_right = int((x2_all - img_W).clamp(min=0).max().item())

        # Pad图像
        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            batch_inputs_padded = F.pad(
                batch_inputs, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            y1_padded = y1_all + pad_top
            x1_padded = x1_all + pad_left
        else:
            batch_inputs_padded = batch_inputs
            y1_padded = y1_all
            x1_padded = x1_all

        # 构造采样坐标网格
        y_offsets = torch.arange(pH, device=device)  # [pH]
        x_offsets = torch.arange(pW, device=device)  # [pW]

        # y_indices: [B, K, pH], x_indices: [B, K, pW]
        y_indices = y1_padded.unsqueeze(-1) + \
            y_offsets.unsqueeze(0).unsqueeze(0)
        x_indices = x1_padded.unsqueeze(-1) + \
            x_offsets.unsqueeze(0).unsqueeze(0)

        # 高级索引一次性提取所有patches
        # 构造采样坐标: [B, K, pH, pW]
        y_idx = y_indices.unsqueeze(-1).expand(-1, -1, -1, pW)
        x_idx = x_indices.unsqueeze(-2).expand(-1, -1, pH, -1)

        # 构造batch索引
        b_idx = torch.arange(B, device=device).view(
            B, 1, 1, 1).expand(-1, K, pH, pW)

        # 高级索引提取: [B*K*pH*pW, C] -> reshape -> [B, K, C, pH, pW]
        sampled = batch_inputs_padded[b_idx.flatten(
        ), :, y_idx.flatten(), x_idx.flatten()]
        patches = sampled.view(B, K, pH, pW, C).permute(
            0, 1, 4, 2, 3).contiguous()

        # 计算patch坐标 (x1, y1, x2, y2)
        patch_coords = torch.stack([
            x1_all.float().clamp(min=0, max=img_W),
            y1_all.float().clamp(min=0, max=img_H),
            (x1_all + pW).float().clamp(min=0, max=img_W),
            (y1_all + pH).float().clamp(min=0, max=img_H)
        ], dim=-1)  # [B, K, 4]

        return patches, patch_coords, topk_scores

    def select_patches_with_subtraction_batch(self, pred_gain_map, patch_size, img_shape, k):
        """
        改进的贪婪算法：使用"软减法"代替"掩码"，解决集合覆盖的残余重叠问题。

        Args:
            pred_gain_map: [B, C, H, W] GlobalNet 预测的收益图 (通常 C=1)
            patch_size: (pH, pW) Patch 的像素尺寸
            img_shape: (img_H, img_W)
            k: 每个样本选取的最大 Patch 数量 (对应之前的 max_patches)

        Returns:
            topk_coords: [B, K, 2]  (y, x) 顺序
            topk_scores: [B, K]
        """
        B, C, H, W = pred_gain_map.shape
        device = pred_gain_map.device
        img_H, img_W = img_shape
        pH, pW = patch_size

        # --- 1. 预计算"重叠率核" (Overlap Kernel) ---

        # 将 Patch 尺寸转为 Feature Map 上的尺寸 (单位: Grid)
        stride_h = img_H / H
        stride_w = img_W / W
        feat_pH = int(pH / stride_h)
        feat_pW = int(pW / stride_w)

        # 构建核的网格 [-feat_pH, feat_pH]
        k_h = 2 * feat_pH + 1
        k_w = 2 * feat_pW + 1

        # 构造 1D 线性衰减 (三角形)
        kernel_y = 1.0 - \
            torch.abs(torch.arange(k_h, device=device) -
                      feat_pH) / float(feat_pH)
        kernel_y = kernel_y.clamp(min=0)

        kernel_x = 1.0 - \
            torch.abs(torch.arange(k_w, device=device) -
                      feat_pW) / float(feat_pW)
        kernel_x = kernel_x.clamp(min=0)

        # 2D 核: [k_h, k_w]
        overlap_kernel = kernel_y[:, None] * kernel_x[None, :]

        # --- 2. Batch 迭代采样 ---
        batch_topk_coords = []
        batch_topk_scores = []

        process_map = pred_gain_map.clone().detach()

        for b in range(B):
            curr_map = process_map[b, 0]  # [H, W]
            coords = []
            scores = []

            for _ in range(k):
                # 2.1 找最大值
                # view(-1) 展平寻找最大值和索引
                max_val, max_idx = torch.max(curr_map.view(-1), 0)

                cy = max_idx.item() // W
                cx = max_idx.item() % W

                coords.append([cy, cx])  # [y, x] 格式
                scores.append(max_val)

                # 2.2 执行软减法更新 (Soft Subtraction)
                if max_val > 1e-6:
                    # 定义操作区域 (ROI)
                    y1 = max(0, cy - feat_pH)
                    y2 = min(H, cy + feat_pH + 1)
                    x1 = max(0, cx - feat_pW)
                    x2 = min(W, cx + feat_pW + 1)

                    # 定义 Kernel 的切片 (处理边界情况)
                    ky1 = feat_pH - (cy - y1)
                    ky2 = ky1 + (y2 - y1)
                    kx1 = feat_pW - (cx - x1)
                    kx2 = kx1 + (x2 - x1)

                    # 截取对应部分的核
                    sub_kernel = overlap_kernel[ky1:ky2, kx1:kx2]

                    # 执行减法：Neighbor_New = Neighbor_Old - (Max_Val * Overlap_Ratio)
                    curr_map[y1:y2, x1:x2] -= (sub_kernel * max_val)

                    # 修正：防止减成负数
                    curr_map[y1:y2, x1:x2] = curr_map[y1:y2,
                                                      x1:x2].clamp(min=0)

                    # 彻底将中心点置0，防止 float 误差导致死循环选中同一个点
                    curr_map[cy, cx] = 0
                else:
                    # 如果最大值已经是0了，为了防止后续死循环选同一个0点，
                    # 将该点设为 -1 (或极小值) 排除掉
                    curr_map[cy, cx] = -1.0

            batch_topk_coords.append(coords)
            batch_topk_scores.append(scores)

        # --- 3. 组合结果并返回 ---

        # topk_coords: [B, K, 2]
        topk_coords = torch.tensor(batch_topk_coords, device=device).float()

        # topk_scores: [B, K]
        # stack 列表中的 tensor (如果是 tensor) 或者直接转换 list
        # 为了保险起见，显式 stack
        topk_scores = torch.stack([torch.stack(s) if isinstance(
            s[0], torch.Tensor) else torch.tensor(s, device=device) for s in batch_topk_scores])

        return topk_coords, topk_scores







def visualize_map_pred(map_pred, save_path='gain_map.npy', batch_idx=0):
    """将 gain map 保存为 .npy 文件，供离线可视化脚本调色。

    Args:
        map_pred: [B, 1, H, W] 或 [B, H, W] 的 tensor
        save_path: 保存路径，默认 'gain_map.npy'
        batch_idx: 第几个 batch 样本，默认 0
    """
    import numpy as np

    if map_pred.dim() == 4:
        map_pred = map_pred.squeeze(1)

    if map_pred.dim() == 3:
        heatmap = map_pred[batch_idx]
    else:
        heatmap = map_pred

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    if not save_path.lower().endswith('.npy'):
        save_path = save_path.rsplit('.', 1)[0] + '.npy'
    np.save(save_path, heatmap)
    print(f"[gain_map] saved: {save_path}  "
          f"shape={heatmap.shape}  "
          f"range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")


def visualize_patches(batch_inputs, patch_coords, save_path='patches_vis.png',
                      batch_idx=0, draw_boxes=False, dark_ratio=0.5,
                      mean=(123.675, 116.28, 103.53),
                      std=(58.395, 57.12, 57.375),
                      batch_data_samples=None):
    """可视化原图和选中的 patch 区域，非 patch 区域变暗。

    Args:
        batch_inputs: [B, C, H, W] 的 tensor，原图
        patch_coords: [B, K, 4] 的 tensor，patch 坐标 (x1, y1, x2, y2)
        save_path: 保存路径，默认 'patches_vis.png'
        batch_idx: 可视化第几个 batch 样本，默认 0
        draw_boxes: 是否画出 patch 的边框，默认 True
        dark_ratio: 非 patch 区域的亮度比例 (0-1)，默认 0.3 (变暗到 30%)
        mean: 归一化时的均值，默认 (123.675, 116.28, 103.53)
        std: 归一化时的标准差，默认 (58.395, 57.12, 57.375)
        batch_data_samples: 可选，list[DetDataSample]，如果不为 None 则画出 GT bbox

    Returns:
        None (图片保存到 save_path)
    """
    import numpy as np
    import cv2

    # 取指定 batch 的原图
    if isinstance(batch_inputs, torch.Tensor):
        img = batch_inputs[batch_idx].detach().cpu().numpy()  # [C, H, W]
    else:
        img = batch_inputs[batch_idx]

    # 转换为 [H, W, C] 格式，并转为 uint8
    if img.shape[0] in [1, 3]:  # CHW 格式
        img = img.transpose(1, 2, 0)  # [H, W, C]

    # 反归一化: img = img * std + mean
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)

    # RGB -> BGR (因为 data_preprocessor 的 bgr_to_rgb=True)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 如果是灰度图，转为 BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    H, W = img.shape[:2]

    # 取指定 batch 的 patch 坐标
    if isinstance(patch_coords, torch.Tensor):
        patches = patch_coords[batch_idx].detach().cpu().numpy()  # [K, 4]
    else:
        patches = patch_coords[batch_idx]

    # 创建 mask，标记 patch 内的区域
    mask = np.zeros((H, W), dtype=np.uint8)

    for patch in patches:
        x1, y1, x2, y2 = patch
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    # 创建变暗版本的图像
    img_dark = (img * dark_ratio).astype(np.uint8)

    # 合成：patch 区域保持原亮度，其他区域变暗
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch == 255, img, img_dark)

    # 如果需要画边框
    if draw_boxes:
        for i, patch in enumerate(patches):
            x1, y1, x2, y2 = patch
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, min(x1, W))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H))
            y2 = max(0, min(y2, H))
            if x2 > x1 and y2 > y1:
                # 画绿色边框，线宽2
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 5)
                # 在左上角标注 patch 编号
                # cv2.putText(result, f'{i}', (x1 + 5, y1 + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 画 GT bbox (红色)
    if batch_data_samples is not None:
        gt_instances = batch_data_samples[batch_idx].gt_instances
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = gt_bboxes.detach().cpu().numpy()
        for gt_box in gt_bboxes:
            gx1, gy1, gx2, gy2 = gt_box
            gx1, gy1, gx2, gy2 = int(gx1), int(gy1), int(gx2), int(gy2)
            gx1 = max(0, min(gx1, W))
            gx2 = max(0, min(gx2, W))
            gy1 = max(0, min(gy1, H))
            gy2 = max(0, min(gy2, H))
            if gx2 > gx1 and gy2 > gy1:
                cv2.rectangle(result, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

    # 保存图片 (jpg 格式，压缩质量 70)
    if not save_path.lower().endswith(('.jpg', '.jpeg')):
        save_path = save_path.rsplit('.', 1)[0] + '.jpg'
    cv2.imwrite(save_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # cv2.imwrite(save_path, result)
    print(f"Patch可视化已保存到: {save_path}")
    print(f"图像尺寸: {W}x{H}, Patch数量: {len(patches)}")


def cal_coverage_rate(batch_data_samples, patch_coords, savepath='./coverage_rate.pkl'):
    """计算所有 patch 对 GT 的覆盖率 (基于 IoF, 无重复计算)。

    对每个 GT, 计算其被所有 patch 并集覆盖的面积占自身面积的比例 (IoF),
    将各 GT 的 IoF 求和即为"被覆盖的 GT 数"(可为小数)。
    结果按 batch 累积追加保存到 savepath。

    Args:
        batch_data_samples: list[DetDataSample], 包含 gt_instances.bboxes
        patch_coords: Tensor [B, K, 4] (x1, y1, x2, y2) 像素坐标
        savepath: 累积保存的 pkl 路径
    """
    import pickle
    import os

    B = patch_coords.shape[0]
    K = patch_coords.shape[1]
    results = []

    for b in range(B):
        data_sample = batch_data_samples[b]
        gt_bboxes = data_sample.gt_instances.bboxes  # [N, 4]
        N = gt_bboxes.shape[0]

        if N == 0:
            results.append(dict(
                img_path=data_sample.img_path,
                num_gt=0, covered_gt=0.0, coverage=1.0,
            ))
            continue

        gt_np = gt_bboxes.detach().cpu().numpy()       # [N, 4]
        patches_np = patch_coords[b].detach().cpu().numpy()  # [K, 4]

        covered_gt = 0.0

        for i in range(N):
            gx1, gy1, gx2, gy2 = gt_np[i]
            gt_area = (gx2 - gx1) * (gy2 - gy1)
            if gt_area <= 0:
                continue

            # 收集该 GT 与所有 patch 的交集矩形
            inter_rects = []
            for k in range(K):
                px1, py1, px2, py2 = patches_np[k]
                ix1 = max(gx1, px1)
                iy1 = max(gy1, py1)
                ix2 = min(gx2, px2)
                iy2 = min(gy2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    inter_rects.append((ix1, iy1, ix2, iy2))

            if not inter_rects:
                continue

            # --- 坐标压缩法: 精确计算交集矩形的并集面积 ---
            xs = sorted(set(r[0] for r in inter_rects) |
                        set(r[2] for r in inter_rects))
            ys = sorted(set(r[1] for r in inter_rects) |
                        set(r[3] for r in inter_rects))

            union_area = 0.0
            for xi in range(len(xs) - 1):
                for yi in range(len(ys) - 1):
                    # cell 中心点, 用于判定是否被某个交集矩形覆盖
                    cx = (xs[xi] + xs[xi + 1]) * 0.5
                    cy = (ys[yi] + ys[yi + 1]) * 0.5
                    for r in inter_rects:
                        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                            union_area += (xs[xi + 1] - xs[xi]) * \
                                          (ys[yi + 1] - ys[yi])
                            break  # 只计一次, 避免重复

            iof = min(union_area / gt_area, 1.0)
            covered_gt += iof

        coverage = covered_gt / N
        results.append(dict(
            img_path=data_sample.img_path,
            num_gt=N,
            covered_gt=round(covered_gt, 4),
            coverage=round(coverage, 4),
        ))

    # 累积追加到已有文件
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            existing = pickle.load(f)
        existing.extend(results)
    else:
        existing = results

    with open(savepath, 'wb') as f:
        pickle.dump(existing, f)

    # 打印当前累计统计
    total_gt = sum(r['num_gt'] for r in existing)
    total_covered = sum(r['covered_gt'] for r in existing)
    avg_cov = total_covered / total_gt if total_gt > 0 else 0.0
    print(f'[CoverageRate] images={len(existing)}, '
          f'total_gt={total_gt}, covered={total_covered:.1f}, '
          f'avg_coverage={avg_cov:.4f}')


def _union_area_of_rects(rects):
    """坐标压缩法: 精确计算一组矩形的并集面积。"""
    if not rects:
        return 0.0
    xs = sorted(set(r[0] for r in rects) | set(r[2] for r in rects))
    ys = sorted(set(r[1] for r in rects) | set(r[3] for r in rects))
    area = 0.0
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            cx = (xs[xi] + xs[xi + 1]) * 0.5
            cy = (ys[yi] + ys[yi + 1]) * 0.5
            for r in rects:
                if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                    area += (xs[xi + 1] - xs[xi]) * (ys[yi + 1] - ys[yi])
                    break
    return area


def cal_coverage_rate_per_k(batch_data_samples, patch_coords,
                            savepath='./coverage_per_k.pkl'):
    """计算前 1, 2, ..., K 个 patch 各自的累积覆盖率。

    对每张图, 返回长度为 K 的覆盖率数组 cov_curve[k] = 使用前 k+1 个
    patch 时的覆盖率。结果按 batch 累积追加保存到 savepath。

    Args:
        batch_data_samples: list[DetDataSample]
        patch_coords: Tensor [B, K, 4] (x1, y1, x2, y2)
        savepath: pkl 保存路径
    """
    import pickle
    import os

    B = patch_coords.shape[0]
    K = patch_coords.shape[1]
    results = []

    for b in range(B):
        data_sample = batch_data_samples[b]
        gt_bboxes = data_sample.gt_instances.bboxes
        N = gt_bboxes.shape[0]

        if N == 0:
            results.append(dict(
                img_path=data_sample.img_path,
                num_gt=0,
                cov_curve=[1.0] * K,
            ))
            continue

        gt_np = gt_bboxes.detach().cpu().numpy()       # [N, 4]
        patches_np = patch_coords[b].detach().cpu().numpy()  # [K, 4]

        # 为每个 GT 维护其与前 k 个 patch 的交集矩形列表
        per_gt_rects = [[] for _ in range(N)]
        gt_areas = []
        for i in range(N):
            gx1, gy1, gx2, gy2 = gt_np[i]
            gt_areas.append((gx2 - gx1) * (gy2 - gy1))

        cov_curve = []

        for k in range(K):
            px1, py1, px2, py2 = patches_np[k]

            # 增量: 为每个 GT 添加与第 k 个 patch 的交集
            for i in range(N):
                if gt_areas[i] <= 0:
                    continue
                gx1, gy1, gx2, gy2 = gt_np[i]
                ix1 = max(gx1, px1)
                iy1 = max(gy1, py1)
                ix2 = min(gx2, px2)
                iy2 = min(gy2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    per_gt_rects[i].append((ix1, iy1, ix2, iy2))

            # 计算当前前 k+1 个 patch 的总覆盖
            covered_gt = 0.0
            for i in range(N):
                if gt_areas[i] <= 0:
                    continue
                union = _union_area_of_rects(per_gt_rects[i])
                iof = min(union / gt_areas[i], 1.0)
                covered_gt += iof

            cov_curve.append(round(covered_gt / N, 6))

        results.append(dict(
            img_path=data_sample.img_path,
            num_gt=N,
            cov_curve=cov_curve,
        ))

    # 累积追加
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            existing = pickle.load(f)
        existing.extend(results)
    else:
        existing = results

    with open(savepath, 'wb') as f:
        pickle.dump(existing, f)

    # 打印摘要
    all_curves = [r['cov_curve'] for r in existing if r['num_gt'] > 0]
    if all_curves:
        import numpy as _np
        mean_curve = _np.mean(all_curves, axis=0)
        print(f'[CovPerK] images={len(existing)}, K={len(mean_curve)}, '
              f'cov@1={mean_curve[0]:.4f}, '
              f'cov@K={mean_curve[-1]:.4f}')
