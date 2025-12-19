import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
import json

from model.gpt import GPT
from rag.retriever import retrieve

st.set_page_config(page_title="AI Teacher", page_icon="📘")

st.title("📘 AI Teacher (From Scratch)")
st.write("Local LLM • RAG Enabled • JSON Output • No API")

@st.cache_resource
def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/teacher.model")
    return sp

@st.cache_resource
def load_model():
    model = GPT(
        vocab_size=240,      
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_len=512
    )
    model.load_state_dict(
        torch.load("teacher_llm.pth", map_location="cpu")
    )
    model.eval()
    return model

sp = load_tokenizer()
model = load_model()

with open("rag/data/teacher.txt", "r", encoding="utf-8") as f:
    docs = f.readlines()

def generate(prompt, max_tokens=80, temperature=0.8, repetition_penalty=1.2):
    input_ids = sp.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    eos_id = sp.eos_id()
    generated = []

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(x)[:, -1, :]

        for token in set(generated):
            logits[0, token] /= repetition_penalty

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == eos_id:
            break

        generated.append(next_token)
        x = torch.cat(
            [x, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )

    return sp.decode(generated).replace("�", "").strip()

question = st.text_input("Ask your question:")

if st.button("Ask Teacher"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Teacher is thinking..."):

            context, score = retrieve(question, docs)

            final_prompt = f"""
You are a teacher.
Answer ONLY using the given context.
If the answer is not in the context, say:
"I don't know based on the provided material."

Context:
{context}

Question:
{question}

Answer:
"""

            answer = generate(final_prompt)

            result = {
                "question": question,
                "answer": answer,
                "score": score
            }

        st.subheader("📖 Teacher Output (JSON)")
        st.json(result)
