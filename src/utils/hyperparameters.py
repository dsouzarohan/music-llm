VOCAB_SIZE = 3000
BLOCK_SIZE = 512 # context window
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
EMBEDDING_DIM = 384
N_LAYER = 8
N_HEAD = 4
DROPOUT = 0.2
LEARNING_RATE = 3e-4
# block switches (see MusicTransformer)
POS = "huang"
NORM = "layernorm"
MLP = "gelu"
MLP_RATIO = 3
QK_NORM = False
TIE_EMBEDDINGS = False
WEIGHT_DECAY = 0.01
