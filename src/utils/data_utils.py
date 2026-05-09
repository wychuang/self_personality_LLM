"""Data loading, tokenization, batching utilities."""
import torch
from typing import Iterator
from transformers import PreTrainedTokenizer


def batch_generator(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = False,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield tokenized batches from a list of texts."""
    if shuffle:
        import random
        random.shuffle(texts)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield encoded


def chunk_dataset(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 512,
    overlap: int = 0,
) -> list[str]:
    """Split long texts into overlapping chunks of chunk_size tokens."""
    chunks = []
    for text in texts:
        tokens = tokenizer.encode(text)
        step = chunk_size - overlap
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + chunk_size]
            if len(chunk_tokens) < chunk_size // 2:
                continue
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks
