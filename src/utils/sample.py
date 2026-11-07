"""Samples continuations of val-song prompts from a checkpoint and reports REMI grammar error rates.

uv run python -m src.utils.sample --checkpoint data/runs/<run>/best.pt --data_root data --out_dir data/output/<run>
"""

import argparse
import json
import time
from pathlib import Path

import torch
from miditok import REMI, TokSequence

from src.model.model import MusicTransformer
from src.utils.checkpoint import load_checkpoint


def grammar_error_rate(tokenizer, ids):
    """Share of tokens that break REMI's Program, Pitch, Velocity, Duration note structure."""
    seq = TokSequence(ids=ids, are_ids_encoded=True)
    tokenizer.decode_token_ids(seq)
    tokenizer.complete_sequence(seq)
    return tokenizer.tokens_errors(seq)


def note_count(score):
    return sum(len(t.notes) for t in score.tracks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=4, help="Number of val songs, evenly spaced")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data = Path(args.data_root) / "tokenized"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    print(f"{args.checkpoint}: epoch {ckpt['epoch']}, val {ckpt.get('val_loss')}")
    model = MusicTransformer(vocab_size=ckpt["vocab_size"], **ckpt["hyperparams"]).to(device)
    model.load_state_dict(ckpt["model"])
    del ckpt
    tokenizer = REMI(params=data / "config" / "tokenizer.json")

    songs = sorted((data / "val").glob("*.json"))
    picks = [songs[i * len(songs) // args.n] for i in range(args.n)]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    rows = []
    for i, song in enumerate(picks):
        ids = json.loads(song.read_text())["ids"]
        P = args.prompt_len
        torch.manual_seed(i)
        t0 = time.perf_counter()
        with autocast:
            out = model.generate(torch.tensor([ids[:P]], device=device), max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)[0].tolist()
        secs = time.perf_counter() - t0
        full, prompt = tokenizer.decode(out), tokenizer.decode(out[:P])
        stem = f"{i}_{song.stem[:40]}"
        full.dump_midi(out_dir / f"{stem}_generated.mid")
        prompt.dump_midi(out_dir / f"{stem}_prompt.mid")
        rows.append({
            "song": song.stem,
            "generated_error_rate": grammar_error_rate(tokenizer, out[P:]),
            "real_error_rate": grammar_error_rate(tokenizer, ids[P:P + args.max_new_tokens]),
            "new_notes": note_count(full) - note_count(prompt),
            "seconds": secs,
        })
        print(json.dumps(rows[-1]))
    (out_dir / "samples.json").write_text(json.dumps({"args": vars(args), "samples": rows}, indent=1) + "\n")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
