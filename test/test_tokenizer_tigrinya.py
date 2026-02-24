from tokenizers import Tokenizer

TOKENIZER_PATH = "outputs/tokenizer/tokenizer.json"

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)



# 1️⃣ Random Tigrinya Test Words


TEST_WORDS = [
    "ሰላም",
    "ትግርኛ",
    "ኣደርካ",
    "ሕብረት",
    "መንግስቲ",
    "ምምሕዳር",
    "ትምህርቲ",
    "ኤርትራ",
    "ሃገር",
    "ፍቕሪ",
    "ጸሓፊ",
    "ቤት",
    "ስራሕ",
    "ኣቦ",
    "ኣይተ"
]


def test_word(word):
    print("=" * 60)
    print(f"Original: {word}")

    encoding = tokenizer.encode(word)
    tokens = encoding.tokens
    decoded = tokenizer.decode(encoding.ids)

    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

    # Check unknown tokens
    if "<unk>" in tokens:
        print("⚠ WARNING: <unk> token detected")

    # Check round-trip correctness
    if decoded == word:
        print("✅ Round-trip OK")
    else:
        print("❌ Round-trip FAILED")


def run_tests():
    print("\nRunning Tigrinya Tokenizer Tests\n")
    for word in TEST_WORDS:
        test_word(word)



# 2️⃣ Sentence-Level Tests


SENTENCES = [
    "ሰላም ኩን ኣደርካ ዘይተፈጥሮውነት ?",
    "ኣብ ትግርኛ መምህራን ኣሎዉ።",
    "ትምህርቲ ኣገዳሲ ኢዩ፣ ወላ'ውን ጠቃሚ ኢዩ",
]


def run_sentence_tests():
    print("\nRunning Sentence Tests\n")
    for sentence in SENTENCES:
        test_word(sentence)


if __name__ == "__main__":
    run_tests()
    run_sentence_tests()
