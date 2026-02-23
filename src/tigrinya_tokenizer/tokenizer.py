# src/tigrinya_tokenizer/tokenizer.py

import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFC
from tokenizers.decoders import BPEDecoder
import yaml


class TigrinyaTokenizer:
    """
    Simple Tigrinya BPE Tokenizer.
    Users can just call .tokenize(text) and get tokenized output.
    Automatically trains on default corpus or loads existing tokenizer if available.
    """
    
    def __init__(self, tokenizer_path="outputs/tokenizer/tokenizer.json",
                 corpus_file="data/processed/normalized.txt",
                 config_path="configs/bpe_50k.yaml"):
        self.tokenizer_path = tokenizer_path
        self.corpus_file = corpus_file
        self.config_path = config_path

        # Try loading existing tokenizer first
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            print(f"[INFO] Loaded existing tokenizer from {self.tokenizer_path}")
        else:
            print("[INFO] No existing tokenizer found, will train new one.")
            self.tokenizer = None

        # Load config
        self.cfg = self.load_config(self.config_path)

        # Train automatically if tokenizer not loaded
        if self.tokenizer is None:
            self.train()
            #Load the YAMLL CONFIG  tokenizer
def load_config(self, path="configs/bpe_50k.yaml"):
    """
    Load YAML config from package data
    """
    try:
        # Access file inside the installed package
        stream = pkg_resources.resource_stream(__name__, path)
        return yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found inside package: {path}")
    
    def load_config(self, path):
        """Load tokenizer config YAML"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def train(self):
        """Train BPE tokenizer from corpus"""
        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_file}")
        if os.path.getsize(self.corpus_file) == 0:
            raise ValueError(f"Corpus file is empty: {self.corpus_file}")

        print(f"[INFO] Training BPE tokenizer on: {self.corpus_file}")
        print(f"[INFO] Target vocab size: {self.cfg['tokenizer']['vocab_size']}")

        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.normalizer = Sequence([NFC()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder()

        trainer = BpeTrainer(
            vocab_size=self.cfg["tokenizer"]["vocab_size"],
            min_frequency=self.cfg["tokenizer"]["min_frequency"],
            special_tokens=self.cfg["special_tokens"],
        )

        self.tokenizer.train(files=[self.corpus_file], trainer=trainer)

        # Save tokenizer
        Path(self.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(self.tokenizer_path)
        print(f"[INFO] Tokenizer trained and saved to {self.tokenizer_path}")

    def tokenize(self, text):
        """Tokenize text and return list of tokens"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.encode(text).tokens

    def encode(self, text):
        """Return token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Convert token IDs back to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.decode(ids)