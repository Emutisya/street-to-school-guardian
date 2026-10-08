import argparse
import json
from pathlib import Path
import sys

from .model import Retriever, evaluate, load_catalog
from .server import create_server
from .service import RequestError, match


def main():
    parser = argparse.ArgumentParser(description="Street to School Guardian: fictional, consent-first support retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit catalog TF-IDF and save non-personal weights")
    train.add_argument("--output", default=str(Path("models") / "tfidf.json"))
    commands.add_parser("evaluate", help="Evaluate held-out synthetic queries, never used for fitting")
    serve = commands.add_parser("serve", help="Serve the dashboard on loopback only")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model", help="Validate a saved artifact; otherwise fit in memory")
    recommend = commands.add_parser("recommend", help="Read an anonymous example from stdin, not shell history")
    recommend.add_argument("--consent", action="store_true")
    recommend.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    args = parser.parse_args()
    try:
        catalog = load_catalog()
        model = Retriever.load(args.model, catalog) if getattr(args, "model", None) else Retriever(catalog).fit()
        if args.command == "train":
            model.save(args.output)
            print(json.dumps({"output": args.output, "documents": len(catalog), "vocabulary": len(model.idf)}, indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate(model), indent=2))
        elif args.command == "recommend":
            if not args.consent:
                raise RequestError("Use --consent only after agreeing to process an anonymous description.", 403)
            text = sys.stdin.read(801)
            print(json.dumps(match(model, {"text": text, "mode": args.mode, "consent": True}), indent=2))
        else:
            if not 0 <= args.port <= 65535:
                raise ValueError("Port must be between 0 and 65535.")
            server = create_server(model, args.port)
            print(f"Open http://127.0.0.1:{server.server_port} — fictional services, no payload logging.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (RequestError, ValueError, OSError) as exc:
        # Never echo user input or an exception representation containing a payload.
        print(str(exc) if isinstance(exc, RequestError) else "Could not load model, data, or start server. Check paths and port.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
