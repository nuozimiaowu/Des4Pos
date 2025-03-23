from torch.nn import TransformerEncoderLayer
import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer
from typing import List

class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, hugging_model=None, t5_attention_mlp_path=None,
                 inter_module_num_heads=4):

        super(LanguageEncoder, self).__init__()

        self.t5_attention_mlp = T5WithAttentionMLP(
            t5_model_name=hugging_model,
            hidden_size=embedding_dim,
            num_layers=3,  # 需与训练时配置一致
            output_size=embedding_dim
        )
        self.t5_attention_mlp.load_state_dict(torch.load(t5_attention_mlp_path, map_location="cpu"))
        for param in self.t5_attention_mlp.parameters():
            param.requires_grad = False
        self.new_attention = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=inter_module_num_heads,
            dim_feedforward=embedding_dim * 4
        )
        self.new_maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, descriptions: List[str]) -> torch.Tensor:

        with torch.no_grad():

            split_sentences = []
            for desc in descriptions:
                split_sentences.extend(text_tokenize.sent_tokenize(desc))

            tokenized = self.t5_attention_mlp.t5_tokenizer(
                split_sentences,
                return_tensors="pt",
                padding="longest",
                truncation=True
            ).to(self.device)

            base_embeddings = self.t5_attention_mlp(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"]
            )  # (batch_size, embedding_dim)

        sequence = base_embeddings.unsqueeze(1).permute(1, 0, 2)

        attn_output = self.new_attention(sequence)  # (seq_len, batch_size, embedding_dim)

        pooled = self.new_maxpool(attn_output.permute(1, 2, 0))  # (batch_size, embedding_dim, 1)

        return pooled.squeeze(-1)  # (batch_size, embedding_dim)

    @property
    def device(self):
        return next(self.new_attention.parameters()).device