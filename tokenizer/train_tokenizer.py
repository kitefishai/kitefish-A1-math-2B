from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
import json

# CONFIGURATION
DATA_FILE = "tokenizer_data.jsonl"
VOCAB_SIZE = 32000  # Standard size, multiple of 64 for efficiency

def train_tokenizer():
    # 1. Initialize BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence([
        Whitespace(),
        Digits(individual_digits=True)  # This is the "Magic" for Math
    ])
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # tokenizer.decoder = decoders.ByteLevel()

    # 2. Define Special Tokens
    # We include Llama-3 style special tokens for structure
    control_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[THOUGHT]", "[STEP]", "[RESULT]"]
    digits = [str(i) for i in range(10)]
    operators = ["+", "-", "=", "*", "/", "^", "_", "(", ")", "{", "}", "[", "]", "<", ">"]
    greek_vars = ["\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi", "\\phi", "\\psi"]
    latex_cmds = ["\\frac", "\\sqrt", "\\int", "\\sum", "\\infty", "\\text", "\\begin", "\\end"]

    initial_vocab = control_tokens + digits + operators + greek_vars + latex_cmds

    # 3. Setup Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=initial_vocab,
        # Ensure we don't split common LaTeX keywords excessively
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 4. Data Iterator (Generator to save RAM)
    def data_iterator():
        with open(DATA_FILE, "r") as f:
            for i, line in enumerate(f):
                if i > 500_000: break  # Train on first 500k docs only (enough for vocab)
                doc = json.loads(line)
                yield doc["text"]

    print("Training tokenizer... this may take a while.")
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)

    # 5. Post-Processing: Add BOS/EOS markers automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
    )

    # 6. Save
    tokenizer.save("math_tokenizer.json")
    print("Tokenizer saved to 'math_tokenizer.json'")

if __name__ == "__main__":
    train_tokenizer()