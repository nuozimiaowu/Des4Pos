import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, residual_in_fp32=False,
                 lstm_hidden_size=128):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.relu = nn.ReLU()

        self.conv_layer_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=2, batch_first=True)

        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=256, batch_first=True, bidirectional=True)

        self.norm = nn.LayerNorm(out_channels)

        self.fc = nn.Linear(512, 256)

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        r"""Pass the input through the convolution block.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = ConvBlock(LN(residual))
        """
        hidden_states_residual = hidden_states

        if residual is not None:
            residual = hidden_states + residual
        else:
            residual = hidden_states

        hidden_states = hidden_states.permute(0, 2, 1).contiguous()

        hidden_states1 = self.conv_layer_1(hidden_states) + hidden_states
        hidden_states2 = self.conv_layer_3(hidden_states) + hidden_states
        hidden_states3 = self.conv_layer_5(hidden_states) + hidden_states

        hidden_states1 = hidden_states1.permute(0, 2, 1).contiguous()
        hidden_states2 = hidden_states2.permute(0, 2, 1).contiguous()
        hidden_states3 = hidden_states3.permute(0, 2, 1).contiguous()

        atten_states1, _ = self.attention(hidden_states1, hidden_states1, hidden_states1)
        atten_states1 = atten_states1 + hidden_states1
        atten_states2, _ = self.attention(hidden_states2, hidden_states2, hidden_states2)
        atten_states3, _ = self.attention(hidden_states3, hidden_states3, hidden_states3)
        atten_states3 = atten_states3 + hidden_states3

        cross_states1, _ = self.attention(atten_states2, atten_states1, atten_states1)
        cross_states2, _ = self.attention(atten_states2, atten_states3, atten_states3)
        fused_feature_map = cross_states1 * cross_states2
        fused_feature = fused_feature_map * atten_states2
        encodings_point_cloud = hidden_states_residual + fused_feature

        lstm_output, _ = self.lstm(encodings_point_cloud)

        indices = list(range(0, 512, 2))

        narrowed_tensor = lstm_output[:, :, indices]

        return narrowed_tensor, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # 如果需要，可能用于推理阶段的缓存分配
        return None