import torch
from torch import nn
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models.layers.transformer.utils import MLP
from torch.utils.checkpoint import checkpoint

class UHRDETRDecoderLayer(nn.Module):
    """Decoder layer: self-attn + global cross-attn + local deformable cross-attn + FFN.

    Each decoder layer processes queries through (coarse-to-fine):
    1. Self-attention: queries attend to each other
    2. Global cross-attention: all queries attend to the global feature map (coarse)
    3. Local cross-attention: each query attends to its patch features via DeformableAttention (fine)
    4. FFN
    """

    def __init__(self, embed_dims=256, num_heads=8, num_levels=3,
                 num_points=4, ffn_channels=1024, dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims

        self.self_attn = MultiheadAttention(
            embed_dims=embed_dims, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)

        self.local_cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points, im2col_step=100,
            batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.global_cross_attn = MultiheadAttention(
            embed_dims=embed_dims, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=ffn_channels,
            num_fcs=2, ffn_drop=dropout,
            act_cfg=dict(type='ReLU', inplace=True))
        self.norm4 = nn.LayerNorm(embed_dims)

    def forward_self_attn(self, query, query_pos, self_attn_mask=None):
        query = self.self_attn(
            query=query, key=query, value=query,
            query_pos=query_pos, key_pos=query_pos,
            attn_mask=self_attn_mask)
        return self.norm1(query)

    def forward_local_cross_attn(self, query, query_pos, value,
                                 reference_points, spatial_shapes,
                                 level_start_index):
        query = self.local_cross_attn(
            query=query, value=value, query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        return self.norm2(query)

    def forward_global_cross_attn(self, query, query_pos,
                                  global_value, global_key_pos):
        query = self.global_cross_attn(
            query=query, key=global_value, value=global_value,
            query_pos=query_pos, key_pos=global_key_pos)
        return self.norm3(query)

    def forward_ffn(self, query):
        query = self.ffn(query)
        return self.norm4(query)


class UHRDETRDecoder(nn.Module):
    """Decoder with local deformable + global cross-attention.

    For each layer (coarse-to-fine):
    1. Self-attention among all queries (batch B)
    2. Global cross-attention with global feature map (batch B, coarse)
    3. Local DeformableAttention: queries grouped by patch (batch B*K, fine)
       - reference_points transformed from global to patch-local coords
    4. FFN
    5. Iterative box refinement in global coords
    """

    def __init__(self, num_layers=6, embed_dims=256, num_heads=8,
                 num_levels=3, num_points=4, ffn_channels=1024, dropout=0.0,
                 with_checkpoint=False):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.with_checkpoint = with_checkpoint

        self.layers = nn.ModuleList([
            UHRDETRDecoderLayer(
                embed_dims, num_heads, num_levels,
                num_points, ffn_channels, dropout)
            for _ in range(num_layers)
        ])
        self.ref_point_head = MLP(4, embed_dims * 2, embed_dims, 2)

    def forward(self, query, reference_points,
                local_memory, local_spatial_shapes, local_level_start_index,
                global_memory, global_key_pos,
                patch_coords, query_patch_idx,
                reg_branches, cls_branches,
                self_attn_mask=None, img_shape=None, eval_idx=-1,
                global_spatial_shape=None):
        """
        Args:
            query: [B, N, C] selected queries
            reference_points: [B, N, 4] inverse_sigmoid, LOCAL coords (cx,cy,w,h)
            local_memory: [B*K, num_feat_pts, C] per-patch encoder features
            local_spatial_shapes: [num_levels, 2] (H, W) per level
            local_level_start_index: [num_levels]
            global_memory: [B, H_g*W_g, C] global feature map flattened
            global_key_pos: [1, H_g*W_g, C] positional encoding
            patch_coords: [B, K, 4] (x1, y1, x2, y2) pixel coords
            query_patch_idx: [B, N] patch index (0..K-1) for each query
            reg_branches, cls_branches: nn.ModuleList
            self_attn_mask: optional
            img_shape: (img_H, img_W)
            eval_idx: inference layer index

        Returns:
            all_cls: list of [B, N, num_classes]
            all_coords: list of [B, N, 4] sigmoid coords in global system
        """

        B, N, C = query.shape
        K = local_memory.shape[0] // B

        unact_ref = reference_points  # LOCAL inverse_sigmoid
        ref = unact_ref.sigmoid()     # [B, N, 4] LOCAL normalized [0,1]

        if eval_idx < 0:
            eval_idx = eval_idx + self.num_layers

        all_cls = []
        all_coords = []

        for lid, layer in enumerate(self.layers):
            # Global ref for positional encoding (self-attn & global cross-attn)
            ref_global = self.local_to_global_coords(
                ref, patch_coords, query_patch_idx, img_shape)

            query_pos = self.ref_point_head(ref_global)

            # === 1. Self-attention (batch B, global positional encoding) ===
            if self.with_checkpoint and self.training:
                query = checkpoint(
                    layer.forward_self_attn, query, query_pos, self_attn_mask,
                    use_reentrant=False)
            else:
                query = layer.forward_self_attn(
                    query, query_pos, self_attn_mask)

            # === 2. Global cross-attention (batch B, coarse) ===
            query_pos = self.build_sincos_query_pos(
                ref_global, self.embed_dims,
                global_spatial_shape=global_spatial_shape)
            if self.with_checkpoint and self.training:
                query = checkpoint(
                    layer.forward_global_cross_attn,
                    query, query_pos, global_memory, global_key_pos,
                    use_reentrant=False)
            else:
                query = layer.forward_global_cross_attn(
                    query, query_pos, global_memory, global_key_pos)

            # === 3. Local cross-attention (batch B*K, fine) ===
            # Local positional encoding from local reference_points
            query_pos_local = self.ref_point_head(ref)

            # a. Gather queries per patch
            (gathered_q, gathered_pos, gathered_ref,
             q_counts, q_indices) = self.gather_per_patch(
                query, query_pos_local, ref,
                query_patch_idx, B, K)

            # b. Expand ref_points for multi-level DeformableAttention
            num_levels = local_spatial_shapes.shape[0]
            deform_ref = gathered_ref[:, :, None, :].expand(
                -1, -1, num_levels, -1)

            # c. Run local deformable cross-attention
            if self.with_checkpoint and self.training:
                gathered_q = checkpoint(
                    layer.forward_local_cross_attn,
                    gathered_q, gathered_pos, local_memory,
                    deform_ref, local_spatial_shapes, local_level_start_index,
                    use_reentrant=False)
            else:
                gathered_q = layer.forward_local_cross_attn(
                    gathered_q, gathered_pos, local_memory,
                    deform_ref, local_spatial_shapes, local_level_start_index)

            # d. Scatter back to [B, N, C]
            query = self.scatter_per_patch(
                gathered_q, q_indices, q_counts, B, N, C, K)

            # === 4. FFN ===
            if self.with_checkpoint and self.training:
                query = checkpoint(layer.forward_ffn, query,
                                   use_reentrant=False)
            else:
                query = layer.forward_ffn(query)

            # === 5. Iterative refinement in LOCAL coords ===
            tmp = reg_branches[lid](query)

            if self.training or lid == eval_idx:
                all_cls.append(cls_branches[lid](query))
                # Convert refined local coords to global for loss
                coords_local = (tmp + unact_ref).sigmoid()
                coords_global = self.local_to_global_coords(
                    coords_local, patch_coords, query_patch_idx, img_shape)
                all_coords.append(coords_global)
                if not self.training or lid == self.num_layers - 1:
                    break

            unact_ref = (tmp + unact_ref).detach()
            ref = unact_ref.sigmoid().detach()

        return all_cls, all_coords

    @staticmethod
    def build_sincos_query_pos(ref_global, embed_dim=256, temperature=10000.,
                               global_spatial_shape=None):
        """Generate sinusoidal position encoding from continuous (cx, cy) coords.

        Aligned with global_key_pos: both use 2D sincos encoding with the same
        format [sin(x), cos(x), sin(y), cos(y)].  When global_spatial_shape is
        given, coordinates are scaled from [0,1] to the feature-map grid range
        so that the query-pos and key-pos live in the same numerical space.

        Args:
            ref_global: [B, N, 4] (cx, cy, w, h) in [0,1] global normalised
            embed_dim:  output dimension (must be divisible by 4)
            temperature: base temperature (same default as global_key_pos)
            global_spatial_shape: (H_g, W_g) of global feature map.
                If provided, cx is scaled by W_g and cy by H_g so that
                the encoding matches the integer-grid key_pos exactly.
                If None, raw [0,1] coordinates are used.

        Returns:
            pos_enc: [B, N, embed_dim]
        """
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4

        cx = ref_global[..., 0:1]  # [B, N, 1]
        cy = ref_global[..., 1:2]  # [B, N, 1]

        # Scale to feature-map grid range for alignment with global_key_pos
        if global_spatial_shape is not None:
            H_g, W_g = global_spatial_shape
            cx = cx * W_g
            cy = cy * H_g

        omega = torch.arange(
            pos_dim, dtype=torch.float32, device=ref_global.device)
        omega = temperature ** (omega / -pos_dim)   # [pos_dim]

        out_x = cx * omega   # [B, N, pos_dim]
        out_y = cy * omega   # [B, N, pos_dim]

        return torch.cat([
            torch.sin(out_x), torch.cos(out_x),
            torch.sin(out_y), torch.cos(out_y)
        ], dim=-1)   # [B, N, embed_dim]

    @staticmethod
    def local_to_global_coords(ref_local, patch_coords, query_patch_idx,
                               img_shape):
        """Convert reference_points from patch-local to global normalized coords.

        Local coords:  [0,1] normalized to the patch
        Global coords: [0,1] normalized to full image (img_H, img_W)

        No clamping on global coords — small objects need fine-grained coords.
        In local [0,1] at patch 640px, 0.01 = 6.4px (acceptable).
        In global [0,1] at 8192px, 0.01 = 81.9px (too coarse for small objects).

        Args:
            ref_local: [B, N, 4] (cx, cy, w, h) in [0,1] patch-local
            patch_coords: [B, K, 4] (x1, y1, x2, y2) in pixel coords
            query_patch_idx: [B, N] which patch each query belongs to
            img_shape: (img_H, img_W)

        Returns:
            ref_global: [B, N, 4] (cx, cy, w, h) in [0,1] global
        """
        img_H, img_W = img_shape

        idx = query_patch_idx.unsqueeze(-1).expand(-1, -1, 4)
        qpc = torch.gather(patch_coords, 1, idx)  # [B, N, 4]

        p_x1 = qpc[..., 0]
        p_y1 = qpc[..., 1]
        p_w = (qpc[..., 2] - qpc[..., 0]).clamp(min=1.0)
        p_h = (qpc[..., 3] - qpc[..., 1]).clamp(min=1.0)

        cx_global = (ref_local[..., 0] * p_w + p_x1) / img_W
        cy_global = (ref_local[..., 1] * p_h + p_y1) / img_H
        w_global = ref_local[..., 2] * p_w / img_W
        h_global = ref_local[..., 3] * p_h / img_H

        ref_global = torch.stack(
            [cx_global, cy_global, w_global, h_global], dim=-1)
        return ref_global

    @staticmethod
    def global_to_local_coords(ref_global, patch_coords, query_patch_idx,
                               img_shape):
        """Convert reference_points from global normalized to patch-local coords.

        Inverse of local_to_global_coords.

        Args:
            ref_global: [B, N, 4] (cx, cy, w, h) in [0,1] global normalized
            patch_coords: [B, K, 4] (x1, y1, x2, y2) in pixel coords
            query_patch_idx: [B, N] which patch each query belongs to
            img_shape: (img_H, img_W)

        Returns:
            ref_local: [B, N, 4] (cx, cy, w, h) in [0,1] patch-local
        """
        img_H, img_W = img_shape

        idx = query_patch_idx.unsqueeze(-1).expand(-1, -1, 4)
        qpc = torch.gather(patch_coords, 1, idx)  # [B, N, 4]

        p_x1 = qpc[..., 0]
        p_y1 = qpc[..., 1]
        p_w = (qpc[..., 2] - qpc[..., 0]).clamp(min=1.0)
        p_h = (qpc[..., 3] - qpc[..., 1]).clamp(min=1.0)

        cx_local = (ref_global[..., 0] * img_W - p_x1) / p_w
        cy_local = (ref_global[..., 1] * img_H - p_y1) / p_h
        w_local = ref_global[..., 2] * img_W / p_w
        h_local = ref_global[..., 3] * img_H / p_h

        ref_local = torch.stack(
            [cx_local, cy_local, w_local, h_local], dim=-1)
        return ref_local

    def gather_per_patch(self, query, query_pos, local_ref,
                         query_patch_idx, B, K):
        """Gather queries per patch for batched local DeformableAttention.

        Returns:
            gathered_q:   [B*K, max_q, C]
            gathered_pos: [B*K, max_q, C]
            gathered_ref: [B*K, max_q, 4]
            q_counts:     [B, K]
            q_indices:    list[list[Tensor]]
        """
        N, C = query.shape[1], query.shape[2]
        device = query.device

        q_counts = torch.zeros(B, K, dtype=torch.long, device=device)
        q_indices = [[None] * K for _ in range(B)]

        for b in range(B):
            for k in range(K):
                mask = query_patch_idx[b] == k
                q_indices[b][k] = mask.nonzero(as_tuple=True)[0]
                q_counts[b, k] = q_indices[b][k].shape[0]

        max_q = max(q_counts.max().item(), 1)

        gathered_q = query.new_zeros(B * K, max_q, C)
        gathered_pos = query.new_zeros(B * K, max_q, C)
        gathered_ref = local_ref.new_full((B * K, max_q, 4), 0.5)
        gathered_ref[..., 2:] = 0.1  # default w, h for empty slots

        for b in range(B):
            for k in range(K):
                idx = q_indices[b][k]
                n_k = idx.shape[0]
                if n_k == 0:
                    continue
                bk = b * K + k
                gathered_q[bk, :n_k] = query[b, idx]
                gathered_pos[bk, :n_k] = query_pos[b, idx]
                gathered_ref[bk, :n_k] = local_ref[b, idx]

        return gathered_q, gathered_pos, gathered_ref, q_counts, q_indices

    def scatter_per_patch(self, gathered_q, q_indices, q_counts, B, N, C, K):
        """Scatter gathered queries back to [B, N, C]."""
        query = gathered_q.new_zeros(B, N, C)

        for b in range(B):
            for k in range(K):
                idx = q_indices[b][k]
                n_k = idx.shape[0]
                if n_k == 0:
                    continue
                bk = b * K + k
                query[b, idx] = gathered_q[bk, :n_k]

        return query