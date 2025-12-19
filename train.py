import torch
from torch.optim import AdamW
import sentencepiece as spm
from model.gpt import GPT
from torch.nn.utils import clip_grad_norm_

VOCAB_SIZE = 240
BLOCK_SIZE = 32
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/teacher.model")

PAD_ID = sp.pad_id()

with open("rag/data/teacher.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small ({len(data)} tokens) for block_size={block_size}"
        )

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = GPT(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_len=512
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

print("🚀 Training started")

for epoch in range(EPOCHS):
    model.train()

    x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)

    logits = model(x)
    loss = loss_fn(
        logits.view(-1, VOCAB_SIZE),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()

    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "teacher_llm.pth")

print("✅ Training complete")
