import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFC
from tokenizers.decoders import BPEDecoder
import yaml


def load_config(path="configs/bpe_50k.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    corpus_file = "data/processed/normalized.txt"

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

    # Unicode normalization (critical for Ge’ez script consistency)
    tokenizer.normalizer = Sequence([NFC()])

    # ✅ IMPORTANT FIX:
    # Use whitespace splitting so BPE can learn merges inside words
    tokenizer.pre_tokenizer = Whitespace()

    # Proper BPE decoder
    tokenizer.decoder = BPEDecoder()

    # Trainer configuration
    trainer = BpeTrainer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        min_frequency=cfg["tokenizer"]["min_frequency"],
        special_tokens=cfg["special_tokens"]
    )

    print("[INFO] Starting training...")
    tokenizer.train(files=[corpus_file], trainer=trainer)
    print("[INFO] Training complete!")

    # Save tokenizer
    out_dir = Path("outputs/tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "tokenizer.json"
    tokenizer.save(str(output_path))

    print(f"[INFO] Tokenizer saved to: {output_path}")

    # Quick sanity check
    test_text = "ሰላም ኩን ኣደርካ?"
    encoding = tokenizer.encode(test_text)

    print(f"\n[INFO] Sample text: {test_text}")
    print(f"[INFO] Tokens: {encoding.tokens}")
    print(f"[INFO] Decoded: {tokenizer.decode(encoding.ids)}")

    if tokenizer.decode(encoding.ids) == test_text:
        print("[INFO] Round-trip PASSED ✅")
    else:
        print("[WARNING] Round-trip FAILED ❌")

    # Check if merges were learned
    model = tokenizer.model
    if hasattr(model, "get_merges"):
        merges = model.get_merges()
        print(f"[INFO] Number of learned merges: {len(merges)}")
        if len(merges) == 0:
            print("[WARNING] No merges learned! Increase vocab_size or check corpus.")
    else:
        print("[INFO] Cannot directly inspect merges (check tokenizer.json).")


if __name__ == "__main__":
    train()