"""CLI drop-in for `python3 -m mlx_lm.generate` backed by m4lm."""
import argparse
from . import generate, load


def main():
    ap = argparse.ArgumentParser(description="m4lm.generate — direct-Metal LLM CLI")
    ap.add_argument("--model", required=True, help="Local model directory or cached HF repo ID")
    ap.add_argument("--draft", default=None, help="Optional DFlash2 drafter directory or HF repo ID")
    ap.add_argument("--cross-draft", default=None, help="Optional smaller Qwen3.5 draft model (e.g. 0.8B)")
    ap.add_argument("--prompt", "-p", required=True)
    ap.add_argument("--max-tokens", "-m", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    model, tok = load(args.model, draft_model=args.draft, cross_draft_model=args.cross_draft)
    prompt = (
        args.prompt
        if args.no_chat
        else tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, enable_thinking=args.think)
    )
    generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
