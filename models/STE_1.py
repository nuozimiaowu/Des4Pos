import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer
from typing import List

class TextDataset(Dataset):
    def __init__(self, texts: List[str], clip_model, clip_tokenizer, t5_tokenizer, max_length=77):
        self.texts = texts
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_length = max_length

        with torch.no_grad():
            clip_inputs = clip.tokenize(texts, truncate=True).to(clip_model.device)
            self.clip_embeddings = clip_model.encode_text(clip_inputs).float().cpu()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        t5_encoding = self.t5_tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": t5_encoding["input_ids"].squeeze(0),
            "attention_mask": t5_encoding["attention_mask"].squeeze(0),
            "clip_embedding": self.clip_embeddings[idx]  # CLIP 的文本编码作为监督信号
        }


class T5WithAttentionMLP(nn.Module):
    def __init__(self, t5_model_name: str, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(t5_model_name)
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_size, num_heads=8) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        t5_outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = t5_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        for attention_layer in self.attention_layers:
            attn_output, _ = attention_layer(hidden_states, hidden_states, hidden_states)
            hidden_states = hidden_states + attn_output

        hidden_states = hidden_states.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
        pooled_output = self.max_pool(hidden_states).squeeze(2)  # (batch_size, hidden_size)
        output = self.fc(pooled_output)
        return output



def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - nn.functional.cosine_similarity(pred, target, dim=-1).mean()



def train_model(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device,
                num_epochs: int):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clip_embeddings = batch["clip_embedding"].to(device)

            # 前向传播
            pred_embeddings = model(input_ids, attention_mask)

            # 计算余弦相似度损失
            loss = cosine_loss(pred_embeddings, clip_embeddings)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


def main():

    t5_model_name = "t5-small"
    clip_model_name = "ViT-B/32"
    hidden_size = 512
    num_layers = 3
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval()  # 固定 CLIP 模型参数

    # 加载 T5 tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    # 创建数据集和数据加载器
    dataset = TextDataset(texts, clip_model, clip.tokenize, t5_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = T5WithAttentionMLP(t5_model_name, hidden_size, num_layers, output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, optimizer, device, num_epochs)

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "t5_aligned_with_clip.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()