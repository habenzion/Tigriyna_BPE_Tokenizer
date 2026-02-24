from pathlib import Path
from tokenizers import Tokenizer


class TigrinyaTokenizer:
    """
    Tigrinya Tokenizer Library

    Provides:
        - word_tokenize(text)
        - char_tokenize(text)
        - subword_tokenize(text)  (BPE-based)

    Users do NOT call encode/decode directly.
    """

    def __init__(self):
        # Load trained BPE tokenizer for subword tokenization
        pkg_dir = Path(__file__).parent
        tokenizer_path = pkg_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Pre-trained tokenizer not found at {tokenizer_path}. "
                "Ensure tokenizer.json is included in the package."
            )

        self._bpe_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # --------------------------------------------------
    # 1️⃣ Word Tokenization (Whitespace-Based)
    # --------------------------------------------------
    def word_tokenize(self, text: str):
        """
        Splits text by whitespace.

        Returns:
            List[str]
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        return text.split()

    # --------------------------------------------------
    # 2️⃣ Character Tokenization
    # --------------------------------------------------
    def char_tokenize(self, text: str):
        """
        Splits text into individual characters.

        Returns:
            List[str]
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        return list(text)

    # --------------------------------------------------
    # 3️⃣ Subword Tokenization (BPE-Based)
    # --------------------------------------------------
    def subword_tokenize(self, text: str):
        """
        Tokenizes text using the trained BPE tokenizer.

        Returns:
            List[str] — subword tokens
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        encoding = self._bpe_tokenizer.encode(text)
        return encoding.tokens