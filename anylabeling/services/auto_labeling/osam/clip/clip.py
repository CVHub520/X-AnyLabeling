# https://github.com/openai/CLIP/blob/main/clip/clip.py

import numpy as np

from .simple_tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()


def tokenize(
    texts: list[str], context_length: int = 77, truncate: bool = False
) -> np.ndarray:
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = tokens

    return result
