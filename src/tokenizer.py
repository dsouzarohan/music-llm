from miditok import REMI, TokenizerConfig
from pathlib import Path

# https://miditok.readthedocs.io/en/latest/train.html#tokenizer-models

# TODO: Our TokenizerConfig also contain configurations that can be seen as hyperparameters, for now keeping it to the bare miniimum
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_rests=True)
tokenizer = REMI(config)
midi_folder = Path("../data/midi/")
tokenized_folder = Path("../data/tokenized/")
tokenized_folder.mkdir(parents=True, exist_ok=True)

midi_paths = list(midi_folder.glob("*.mid"))
print(midi_paths)

# Trains the tokenizer using BPE to create unified representations of MIDI note sequences
# This will help us reduce the overall sequence length of examples
# That would help us with context window size
tokenizer.train(
    vocab_size=8000,  # TODO: Hyperparameter
    model="BPE",
    files_paths=midi_paths
)

tokenizer.tokenize_dataset(
    midi_paths,
    tokenized_folder,
    overwrite_mode=True
)