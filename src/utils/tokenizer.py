import random
import shutil

from miditok import REMI, TokenizerConfig
from miditok.data_augmentation import augment_dataset
from pathlib import Path
import argparse

from src.utils.hyperparameters import VOCAB_SIZE

# https://miditok.readthedocs.io/en/latest/train.html#tokenizer-models

# TODO: refactor this to function format, adding as a script flow for now
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="../../data/")
parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Vocabulary size")
args = parser.parse_args()

# TODO: Our TokenizerConfig also contain configurations that can be seen as hyperparameters, for now keeping it to the bare minimum
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_rests=True)
tokenizer = REMI(config)

# Define paths
data_root = Path(args.data_root)
midi_folder = data_root / "midi/"
augmented_folder = data_root / "augmented/"
tokenized_folder = data_root / "tokenized/"
splits_folder = data_root / "splits/"
train_tok_folder = tokenized_folder / "train-aug/"
val_tok_folder = tokenized_folder / "val/"
train_midi_folder = data_root / "train-midi/"
val_midi_folder = data_root / "val-midi/"

# persist the folders
tokenized_folder.mkdir(parents=True, exist_ok=True)
train_tok_folder.mkdir(parents=True, exist_ok=True)
val_tok_folder.mkdir(parents=True, exist_ok=True)
train_midi_folder.mkdir(parents=True, exist_ok=True)
val_midi_folder.mkdir(parents=True, exist_ok=True)

# clear and recreate train/val MIDI folders to ensure clean state
for folder in [splits_folder, train_midi_folder, val_midi_folder, augmented_folder, train_tok_folder, val_tok_folder]:
    if folder.exists():
        shutil.rmtree(folder)  # rm the folder directory
    folder.mkdir(parents=True, exist_ok=True)  # recreate the folder

# vocab size
vocab_size = args.vocab_size

# all MIDIs
all_orig_midis = list(midi_folder.glob("*.mid"))
random.shuffle(all_orig_midis)

TRAINING_SPLIT = 0.6
split = int(len(all_orig_midis) * TRAINING_SPLIT)

train_midis = all_orig_midis[:split]
val_midis = all_orig_midis[split:]  # to be untouched by tokenizer training and model learning

# Write the file splits by file name to text files for record-keeping
train_files_txt = splits_folder / "train_files.txt"
val_files_txt = splits_folder / "val_files.txt"

# writes all training file names
with open(train_files_txt, "w") as f:
    for midi in train_midis:
        f.write(f"{midi.name}\n")

# writes all testing file names
with open(val_files_txt, "w") as f:
    for midi in val_midis:
        f.write(f"{midi.name}\n")

# copy the files to their directories
for midi_path in train_midis:
    shutil.copy(midi_path, train_midi_folder / midi_path.name)

for midi_path in val_midis:
    shutil.copy(midi_path, val_midi_folder / midi_path.name)

# Data augmentation to increase dataset size and variability
# This will help us with generalization and overfitting

augment_dataset(
    data_path=train_midi_folder,
    pitch_offsets=[-12, 12],
    velocity_offsets=[-4, 5],
    duration_offsets=[-0.5, 1],
    all_offset_combinations=False,  # flip to True if you want Cartesian combos
    out_path=augmented_folder,  # write augmented JSONs here
    # duration_in_ticks=False,          # default: offsets are in beats
    # restrict_on_program_tessitura=True  # default: keeps pitches in instrument range
)

train_midi_paths = list(augmented_folder.glob("*.mid"))
val_midi_paths = list(val_midi_folder.glob("*.mid"))

# Trains the tokenizer using BPE to create unified representations of MIDI note sequences
# This will help us reduce the overall sequence length of examples
# That would help us with context window size
tokenizer.train(
    vocab_size=vocab_size,
    model="BPE",
    files_paths=train_midi_paths
)

# save tokenizer after training, creates tokenizer.json in tokenized_folder
tokenizer.save(tokenized_folder / "config/tokenizer.json")
print(f"Saved tokenizer to {tokenized_folder} with vocab={len(tokenizer)}")

tokenizer.tokenize_dataset(
    train_midi_paths,
    train_tok_folder,
    overwrite_mode=True
)
print(f"Tokenized training {len(train_midi_paths)} MIDIs to {train_tok_folder}")

tokenizer.tokenize_dataset(
    val_midi_paths,
    val_tok_folder,
    overwrite_mode=True
)
print(f"Tokenized val {len(val_midi_paths)} MIDIs to {val_tok_folder}")
