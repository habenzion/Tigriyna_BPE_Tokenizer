import os
from pathlib import Path
from tokenizers import Tokenizer
from collections import Counter
import matplotlib.pyplot as plt

def load_tokenizer(path="outputs/tokenizer/tokenizer.json"):
    """Load trained tokenizer"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at {path}")
    return Tokenizer.from_file(path)

def calculate_unk_rate(tokenizer, corpus_file):
    """Compute <unk> token rate over corpus"""
    total_tokens = 0
    unk_tokens = 0
    token_lengths = []

    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer.encode(line)
            total_tokens += len(enc.tokens)
            unk_tokens += enc.tokens.count("<unk>")
            token_lengths.extend([len(t) for t in enc.tokens])

    unk_rate = (unk_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    avg_token_length = sum(token_lengths)/len(token_lengths) if token_lengths else 0
    return unk_rate, avg_token_length, token_lengths

def plot_token_length_distribution(token_lengths, out_file="outputs/tokenizer/token_length_hist.png"):
    """Plot histogram of token lengths"""
    plt.figure(figsize=(8,5))
    plt.hist(token_lengths, bins=range(1, max(token_lengths)+2), color="skyblue", edgecolor="black")
    plt.title("Token Length Distribution")
    plt.xlabel("Token Length (characters)")
    plt.ylabel("Frequency")
    plt.savefig(out_file)
    plt.close()
    print(f"[INFO] Token length histogram saved to {out_file}")

def top_n_tokens(tokenizer, corpus_file, n=20):
    """Compute top N most frequent tokens"""
    counter = Counter()
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer.encode(line)
            counter.update(enc.tokens)
    return counter.most_common(n)

def sample_encoding(tokenizer, sentences):
    """Print encoded tokens and decoded text"""
    for s in sentences:
        enc = tokenizer.encode(s)
        print(f"Sentence: {s}")
        print(f"Tokens: {enc.tokens}")
        print(f"Decoded: {tokenizer.decode(enc.ids)}")
        print("-"*50)

def main():
    tokenizer_path = "outputs/tokenizer/tokenizer.json"
    corpus_file = "data/processed/normalized.txt"

    if not os.path.exists(corpus_file):
        print(f"[ERROR] Corpus file not found: {corpus_file}")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"[INFO] Loaded tokenizer from {tokenizer_path}")

    unk_rate, avg_len, token_lengths = calculate_unk_rate(tokenizer, corpus_file)
    print(f"[INFO] <unk> token rate: {unk_rate:.2f}%")
    print(f"[INFO] Average token length (chars): {avg_len:.2f}")

    plot_token_length_distribution(token_lengths)

    print("[INFO] Top 20 most frequent tokens:")
    for tok, freq in top_n_tokens(tokenizer, corpus_file):
        print(f"{tok} : {freq}")

    print("[INFO] Sample tokenization check:")
    sample_sentences = [
        "ሰላም ከመይ ሓዲርካ?",
        "ኣብ ትግርኛ መምህራን እዩ ምስ ትምህርቲ ኣተሓሳስባ"
    ]
    sample_encoding(tokenizer, sample_sentences)

if __name__ == "__main__":
    main()
