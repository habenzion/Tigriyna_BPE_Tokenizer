import os
from pathlib import Path
from tokenizers import Tokenizer

class TigrinyaTokenizer:
    """
    Tigrinya Byte-Pair Encoding (BPE) Tokenizer.

    Usage:
        from tigrinya_tokenizer import TigrinyaTokenizer
        tokenizer = TigrinyaTokenizer()
        tokens = tokenizer.encode("ሰላም ኩን ኣደርካ?")
        text = tokenizer.decode(tokens)
    """
    def __init__(self):
        # Load pre-trained tokenizer shipped in package
        pkg_dir = Path(__file__).parent
        tokenizer_path = pkg_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Pre-trained tokenizer not found at {tokenizer_path}. "
                "Please ensure tokenizer.json is included in the package."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str):
        """Return list of tokens from input text."""
        return self.tokenizer.encode(text).tokens

    def decode(self, tokens):
        """Return text string from list of token IDs or token strings."""
        if all(isinstance(t, int) for t in tokens):
            return self.tokenizer.decode(tokens)
        else:
            # convert token strings to ids first
            ids = [self.tokenizer.token_to_id(t) for t in tokens]
            return self.tokenizer.decode(ids)