import argparse
import json
from pathlib import Path
import sys

from .model import Retriever, evaluate, evaluate_formats, load_catalog
from .server import create_server
from .service import RequestError, browse, compare, match, plan


def main():
    parser = argparse.ArgumentParser(description="Street to School Guardian: fictional, consent-first support retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit catalog TF-IDF and save non-personal weights")
    train.add_argument("--output", default=str(Path("models") / "tfidf.json"))
    commands.add_parser("evaluate", help="Evaluate held-out synthetic queries, never used for fitting")
    commands.add_parser("evaluate-formats", help="Audit fixed benchmark labels under any, offline and online filters")
    inspect = commands.add_parser("inspect", help="Describe the fitted catalog or validate a saved artifact without queries")
    inspect.add_argument("--model", help="Validate this artifact by exact deterministic refit")
    serve = commands.add_parser("serve", help="Serve the dashboard on loopback only")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model", help="Validate a saved artifact; otherwise fit in memory")
    recommend = commands.add_parser("recommend", help="Read an anonymous example from stdin, not shell history")
    recommend.add_argument("--consent", action="store_true")
    recommend.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    catalog_command = commands.add_parser("browse", help="Browse fictional metadata without a barrier description")
    catalog_command.add_argument("--consent", action="store_true")
    catalog_command.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    catalog_command.add_argument("--category")
    catalog_command.add_argument("--offset", type=int, default=0)
    catalog_command.add_argument("--limit", type=int, default=6)
    for name in ("compare", "plan"):
        selected = commands.add_parser(name, help=f"Create a fictional {name} from catalog IDs only")
        selected.add_argument("--consent", action="store_true")
        selected.add_argument("resource_ids", nargs="+")
    args = parser.parse_args()
    if args.command == "serve" and not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535; use 0 to choose an available port.")
    try:
        catalog = load_catalog()
        model = Retriever.load(args.model, catalog) if getattr(args, "model", None) else Retriever(catalog).fit()
        if args.command == "train":
            model.save(args.output)
            print(json.dumps({"output": args.output, "documents": len(catalog), "vocabulary": len(model.idf)}, indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate(model), indent=2))
        elif args.command == "evaluate-formats":
            print(json.dumps(evaluate_formats(model), indent=2))
        elif args.command == "inspect":
            print(json.dumps({**model.summary(), "source": "validated_artifact" if args.model else "in_memory_fit"}, indent=2))
        elif args.command == "browse":
            print(json.dumps(browse(model, {
                "consent": args.consent, "mode": args.mode, "category": args.category,
                "offset": args.offset, "limit": args.limit
            }), indent=2))
        elif args.command in ("compare", "plan"):
            operation = compare if args.command == "compare" else plan
            print(json.dumps(operation(model, {
                "consent": args.consent, "resource_ids": args.resource_ids
            }), indent=2))
        elif args.command == "recommend":
            if not args.consent:
                raise RequestError("Use --consent only after agreeing to process an anonymous description.", 403)
            text = sys.stdin.read(801)
            print(json.dumps(match(model, {"text": text, "mode": args.mode, "consent": True}), indent=2))
        else:
            try:
                server = create_server(model, args.port)
            except OSError:
                print(f"Could not start the local dashboard on port {args.port}. Choose another --port or use --port 0 for an available port.", file=sys.stderr)
                return 2
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
