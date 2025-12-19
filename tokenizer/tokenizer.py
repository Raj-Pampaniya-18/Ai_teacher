import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="rag/data/teacher.txt",
    model_prefix="tokenizer/teacher",
    vocab_size=240,
    model_type="bpe",

    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,

    pad_piece="<pad>",
    bos_piece="<bos>",
    eos_piece="<eos>",
    unk_piece="<unk>"
)
