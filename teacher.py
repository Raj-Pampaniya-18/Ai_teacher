import torch
import torch.nn.functional as F
import sentencepiece as spm
from model.gpt import GPT
from rag.retriever import retrieve

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/teacher.model")

PAD_ID = sp.pad_id()
EOS_ID = sp.eos_id()

model = GPT(
    vocab_size=240,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_len=512
)

model.load_state_dict(torch.load("teacher_llm.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded successfully")

def generate(
    prompt,
    max_tokens=80,
    temperature=0.6,
    repetition_penalty=1.3
):
    input_ids = sp.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated = []

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(x)[:, -1, :]

        for token in set(generated):
            logits[0, token] /= repetition_penalty

        logits /= temperature
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == EOS_ID:
            break

        generated.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

    return sp.decode(generated).replace("�", "").strip()

with open("rag/data/teacher.txt", "r", encoding="utf-8") as f:
    docs = f.readlines()

while True:
    question = input("\nStudent: ")
    if question.lower() in ["exit", "quit"]:
        break

    context = retrieve(question, docs)

    final_prompt = f"""
    You are a teacher.
    Use the context to answer.
    Context:
    {context}
    Student Question:
    {question}
    Answer simply:
    """

    answer = generate(final_prompt)
    print("\nTeacher:", answer)
