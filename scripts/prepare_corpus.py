from pathlib import Path
from tokenizer.corpus import iter_lines
from tokenizer.normalization import normalize_text
from tqdm import tqdm

RAW = Path("data/raw/tlmd/train.txt")
OUT = Path("data/processed/normalized.txt")

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as out:
    for line in tqdm(iter_lines(RAW)):
        clean = normalize_text(line)
        if clean:
            out.write(clean + "\n")