import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.normalizers import Sequence, NFC
from tqdm import tqdm
import yaml

def load_config(path="configs/bpe_50k.yaml"):
    """Load tokenizer config from YAML"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train():
    """Train readable Ge’ez BPE tokenizer on TLMD dataset"""

    cfg = load_config()
    corpus_file = "data/processed/normalized.txt"

    # Check corpus
    if not os.path.exists(corpus_file):
        print(f"[ERROR] Corpus file not found: {corpus_file}")
        return
    if os.path.getsize(corpus_file) == 0:
        print(f"[ERROR] Corpus file is empty: {corpus_file}")
        return

    print(f"[INFO] Training BPE tokenizer on: {corpus_file}")
    print(f"[INFO] Target vocab size: {cfg['tokenizer']['vocab_size']}")

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFC()])

    # Character-level pre-tokenizer (splits every character)
    tokenizer.pre_tokenizer = Split(pattern=r"", behavior="isolated")

    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        min_frequency=cfg["tokenizer"]["min_frequency"],
        special_tokens=cfg["special_tokens"]
    )

    # Add tqdm for visible progress
    print("[INFO] Starting training...")
    tokenizer.train(files=[corpus_file], trainer=trainer)
    print("[INFO] Training complete!")

    # Save tokenizer (Hugging Face compatible)
    out_dir = Path("outputs/tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tokenizer.json"
    tokenizer.save(str(out_file))
    print(f"[INFO] Tokenizer saved to: {out_file}")

    # Quick test
    test_text = "ኣብዚ ቦታ ምንባር ሰናይ ኢዩ።"
    tokens = tokenizer.encode(test_text).tokens
    print(f"[INFO] Sample encoding for '{test_text}': {tokens}")
    decoded = tokenizer.decode(tokenizer.encode(test_text).ids)
    print(f"[INFO] Decoded back: {decoded}")

if __name__ == "__main__":
    train()
