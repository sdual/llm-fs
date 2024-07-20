import re
import os

from llmfs.tokenizer import SimpleTokenizerV2
from llmfs import ROOT_DIR


def test_encode():
    with open(
        os.path.join(ROOT_DIR, "resources/texts", "the-verdict.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV2(vocab)
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
    pride."""
    ids = tokenizer.encode(text)

    assert ids == [
        1,
        56,
        2,
        850,
        988,
        602,
        533,
        746,
        5,
        1126,
        596,
        5,
        1,
        67,
        7,
        38,
        851,
        1108,
        754,
        793,
        7,
    ]
