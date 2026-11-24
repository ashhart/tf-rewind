"""CLI drop-in for `python3 -m mlx_lm.server` backed by m4lm native HTTP/SSE + /v1/stream-parallel."""
import argparse
from . import load


def main():
    ap = argparse.ArgumentParser(description="m4lm.server — direct-Metal OpenAI + /v1/stream-parallel server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--cross-draft", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--max-streams", type=int, default=16)
    args = ap.parse_args()

    model, _ = load(
        args.model,
        draft_model=args.draft,
        cross_draft_model=args.cross_draft,
        max_context=args.max_context,
        max_streams=args.max_streams,
    )
    model.serve(port=args.port, model_name=args.name)


if __name__ == "__main__":
    main()
