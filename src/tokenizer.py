from miditok import REMI, TokenizerConfig
from pathlib import Path
# shall implement this one day from scratch hehe

config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_rests=True)
tokenizer = REMI(config)
midi_folder = Path("../data/midi/")
tokenized_folder = Path("../data/tokenized/")
tokenized_folder.mkdir(parents=True, exist_ok=True)

midi_paths = list(midi_folder.glob("*.mid"))
print(midi_paths)

for midi_path in midi_paths[:3]:
    print(midi_path)
    try:
        tokens = tokenizer(midi_path)
        tokenizer.save_tokens(tokens, tokenized_folder / (midi_path.stem + ".json"))
    except Exception as e:
        print(f"Error tokenizing {midi_path}: {e}")