import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """Attention-based fusion of spatial and frequency features"""
    
    def __init__(
            self, 
            spatial_dim: int, freq_dim: int, hidden_dim: int = 256, 
            drop_out: float = 0.3):
        """
        Args:
            spatial_dim: Dimension of spatial features
            freq_dim: Dimension of frequency features
            hidden_dim: Hidden dimension for attention
            drop_out: Drop_out layer prob
        """
        super(AttentionFusion, self).__init__()

        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim
        self.out_dim = spatial_dim + freq_dim

        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.freq_norm = nn.LayerNorm(self.freq_dim)


        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, self.out_dim),
            nn.Sigmoid()
        )

        self.output_dropout = nn.Dropout(drop_out)
    
    def forward(
        self, 
        spatial_features: torch.Tensor, 
        freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, spatial_dim)
            freq_features: (B, freq_dim)
        Returns:
            Fused features (B, spatial_dim + freq_dim)
        """
        # Normalize feature
        spatial_norm_feat = self.spatial_norm(spatial_features)
        freq_norm_feat = self.freq_norm(freq_features)

        # Joint context -> observe 2 branch and decide
        joint = torch.cat([spatial_norm_feat, freq_norm_feat], dim=1)

        # Calculate attention mask (Element-wise Attention Weights)
        gate_attn = self.gate(joint)

        # Apply Residual Attention
        gate_spatial, gate_freq = torch.split(gate_attn, [self.spatial_dim, self.freq_dim], dim=1)


        # Apply attention
        weighted_spatial = spatial_norm_feat * (1 + gate_spatial)
        weighted_freq = freq_norm_feat * (1 + gate_freq)
        
        # Concatenate
        fused = torch.cat([
            weighted_spatial, weighted_freq
        ], dim=1)

        return self.output_dropout(fused)
    
    @torch.no_grad()
    def get_attention_weights(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor
    ):
        """
        Return attention weights used in forward pass.
        Values are in [0, 1] independently (sigmoid gate).
        """

        spatial_norm = self.spatial_norm(spatial_feat)
        freq_norm = self.freq_norm(freq_feat)
        joint = torch.cat([spatial_norm, freq_norm], dim=1)

        # attention_context 
        attn_gate = self.gate(joint)

        spatial_weight, freq_weight = torch.split(
            attn_gate, [self.spatial_dim, self.freq_dim], dim=1
        )

        return spatial_weight, freq_weight