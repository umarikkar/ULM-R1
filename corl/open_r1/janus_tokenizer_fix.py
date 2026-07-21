"""Correctly-encoding tokenizer for Janus-Pro.

Janus ships `tokenizer_class: LlamaTokenizer` (slow SentencePiece) in its
tokenizer_config, but the actual vocab is byte-level BPE (space == "Ġ"). The
slow tokenizer *decodes* fine yet **mangles `encode`**: it silently drops
spaces, e.g. "This is a CT scan" -> ['This','isa','CT','sc','ano','ft','he'].
Loading via AutoTokenizer / LlamaTokenizer(Fast) all resolve to this broken
slow behaviour; only building a PreTrainedTokenizerFast straight from the raw
`tokenizer.json` (ByteLevel BPE backend) encodes correctly.

Any code that *tokenizes text as a training target or prompt* (not just decode)
must use `load_fast_tokenizer` instead of `processor.tokenizer`, or spaces are
lost and the model learns to emit space-less captions.
"""
import json
import os

from transformers import PreTrainedTokenizerFast


def _resolve(model, fname):
    if os.path.isdir(model):
        return os.path.join(model, fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, fname)


def load_fast_tokenizer(model="deepseek-ai/Janus-Pro-1B"):
    """Return a PreTrainedTokenizerFast whose `encode` preserves spaces.

    Same vocab / special-token ids as the Janus processor tokenizer
    (image_id=100581, bos=100000, eos=100001), so it is a drop-in replacement
    for `processor.tokenizer`.
    """
    cfg = json.load(open(_resolve(model, "tokenizer_config.json")))
    keep = ("bos_token", "eos_token", "pad_token", "unk_token",
            "additional_special_tokens", "add_bos_token", "add_eos_token",
            "model_max_length")
    kw = {k: cfg[k] for k in keep if k in cfg}
    return PreTrainedTokenizerFast(
        tokenizer_file=_resolve(model, "tokenizer.json"), **kw)
