# References

Papers, datasets, repos and docs behind this project. **Used** means the code or a past
decision depends on it. **Reading** means it's on the list for a planned phase (see
`docs/ROADMAP.md`). Entries carried over from the journal keep their original links.

## Core: what the model is built on

| Status | Reference | Why it's here |
|---|---|---|
| Used | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | The transformer, written from scratch (journal 2025-09-10) |
| Used | Huang et al., [Music Transformer](https://arxiv.org/abs/1809.04281) (2018) | Relative attention with the "skewing" trick, as in `RelativeHead` |
| Used | Shaw et al., [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) (2018) | The relative-position idea Music Transformer builds on |
| Used | Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) | Structure of the model and training loop |
| Used | Radford et al., [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) | Pre-LN block, init N(0, 0.02), residual scaling 1/√(2L) |
| Used | Hendrycks & Gimpel, [GELU](https://arxiv.org/abs/1606.08415) (2016) | MLP activation |
| Used | Smith & Topin, [Super-Convergence / one-cycle](https://arxiv.org/abs/1708.07120) (2017); [PyTorch OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) | LR schedule (journal 2025-12-04) |

## Tokenization and MIDI

| Status | Reference | Why it's here |
|---|---|---|
| Used | Fradet et al., [MidiTok](https://arxiv.org/abs/2310.17202) and its [docs](https://miditok.readthedocs.io/en/latest/) | Tokenizer library: REMI, BPE training, augmentation |
| Used | Huang & Yang, [Pop Music Transformer (REMI)](https://arxiv.org/abs/2002.00212) (2020) | Origin of the REMI encoding |
| Used | Fradet et al., [Byte Pair Encoding for Symbolic Music](https://aclanthology.org/2023.emnlp-main.123/) (EMNLP 2023) | BPE on music tokens |
| Used | Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) (2016) | Original BPE |
| Used | [symusic](https://github.com/Yikai-Liao/symusic) | Fast C++ MIDI parsing, used by miditok and for the bend scan |
| Used | [music21](https://web.mit.edu/music21/) | Current guitar-part extraction |
| Reading | [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) | Reading Guitar Pro files for a tab tokenizer (phase 4) |
| Used | [MIDI.org](https://www.midi.org/) and [MidiTok: the MIDI protocol](https://miditok.readthedocs.io/en/latest/midi.html) | MIDI basics (`docs/journal/learnings/midi-tokenization.md`) |

## Datasets

| Status | Reference | Notes |
|---|---|---|
| Used | Raffel, [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) | The "Clean" subset (~17k files) is the current source; LMD-full has ~177k |
| Reading | Lee et al., [GigaMIDI](https://arxiv.org/abs/2502.17726) (TISMIR 2025), [HF dataset](https://huggingface.co/datasets/Metacreation/GigaMIDI) | 1.4M files, 5.3M tracks, expressive-performance flags |
| Reading | Sarmento et al., [DadaGP](https://archives.ismir.net/ismir2021/paper/000076.pdf) (ISMIR 2021), [repo](https://github.com/dada-bots/dadaGP) | 26k Guitar Pro songs, ~116M tokens. Dataset available on request from the authors (research use) |
| Reading | Bradshaw & Colton, [Aria-MIDI](https://arxiv.org/abs/2504.15071) (ICLR 2025) | 1M+ transcribed piano MIDIs. Piano only, but shows transcription as a data source |

## Symbolic music models

| Status | Reference | Relevance |
|---|---|---|
| Reading | Thickstun et al., [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620) (TMLR 2024) | Infilling and accompaniment; 360M/780M on Lakh-full |
| Reading | Sarmento et al., [GTR-CTRL](https://arxiv.org/abs/2302.05393), [ShredGP](https://arxiv.org/abs/2307.05324), [ProgGP](https://arxiv.org/abs/2307.05328) (2023) | Guitar tab generation with control tokens |
| Reading | [MuPT](https://arxiv.org/abs/2404.06393) (ICLR 2025) | Scaling laws for symbolic music with repeated data |
| Reading | [NotaGen](https://arxiv.org/abs/2502.18008) (IJCAI 2025) | Pretrain → SFT → CLaMP-DPO; the reference recipe |
| Reading | Guo & Dixon, [Moonbeam](https://arxiv.org/abs/2505.15559) (2025) | RoPE per musical attribute (multidimensional relative attention) |
| Reading | [Fretting-Transformer](https://arxiv.org/abs/2506.14223) (2025) | MIDI-to-tab; playability metrics |
| Reading | [MIDI-RWKV](https://arxiv.org/abs/2506.13001) (2025) | Linear-attention symbolic model |
| Background | Ren et al., [PopMAG](https://arxiv.org/abs/2008.07703) (2020); OpenAI, [MuseNet](https://openai.com/index/musenet/) (2019) | Random segment sampling (journal 2025-09-15) |

## Scaling and data

| Status | Reference | Relevance |
|---|---|---|
| Reading | Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020) | Loss vs N, D, C |
| Reading | Hoffmann et al., [Chinchilla](https://arxiv.org/abs/2203.15556) (2022) | ~20 tokens per parameter |
| Reading | Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (2023) | Repeating data is fine for ~4 epochs |
| Reading | Hernandez et al., [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) (2021) | "Effective data transferred": the case for all-instrument pretraining |
| Background | [GPT-3](https://arxiv.org/abs/2005.14165), [The Pile](https://arxiv.org/abs/2101.00027), [Megatron-LM](https://arxiv.org/abs/1909.08053), [PaLM](https://arxiv.org/abs/2204.02311) | Random span sampling and corpus curation (journal 2025-09-15) |

## Modern architecture and training (phase 2)

| Status | Reference | Relevance |
|---|---|---|
| Reading | Su et al., [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) (2021) | Rotary position embeddings |
| Reading | Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) (2022) | Fused, memory-efficient attention |
| Reading | Zhang & Sennrich, [RMSNorm](https://arxiv.org/abs/1910.07467) (2019) | Normalisation |
| Reading | Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020) | SwiGLU |
| Reading | Jordan, [Muon](https://kellerjordan.github.io/posts/muon/) | Orthogonalised-update optimizer |
| Reading | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Modern training tricks in one small script |
| Reading | Karpathy, [nanochat](https://github.com/karpathy/nanochat) | Tokenizer → pretrain → SFT → RL end to end; the template for phases 5–6 |

## Post-training (phases 5–6)

| Status | Reference | Relevance |
|---|---|---|
| Reading | Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (2023) | DPO |
| Reading | Shao et al., [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) (2024) | GRPO |
| Reading | DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) | RL with verifiable rewards |
| Reading | Cideron et al., [MusicRL](https://arxiv.org/abs/2402.04229) (ICML 2024) | RLHF for music; limits of automatic rewards |
| Reading | [SMART](https://arxiv.org/abs/2504.16839) (2025); [GRPO extension](https://zenodo.org/records/22254725) (AIMC 2026) | RL on symbolic music with audio-domain rewards |

## Evaluation

| Status | Reference | Relevance |
|---|---|---|
| Reading | Retkowski et al., [Fréchet Music Distance](https://arxiv.org/abs/2412.07948) (2024), [library](https://github.com/jryban/frechet-music-distance) | FID-style metric on CLaMP embeddings |

## Tools and infrastructure

| Status | Reference | Relevance |
|---|---|---|
| Used | [uv](https://docs.astral.sh/uv/) | Environment and dependencies |
| Used | [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) | Cloud GPU runs (`docs/colab.md`) |
| Reading | [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/), [Triton](https://triton-lang.org/) | Attention variants and custom kernels (phase 7) |
| Reading | NVIDIA [CUDA Rust](https://forums.developer.nvidia.com/t/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/382704) (cuda-oxide, cutile-rs) | Rust GPU kernels (phase 7) |
