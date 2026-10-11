import argparse
import json
from pathlib import Path
import sys

from .model import Retriever, audit_catalog, evaluate, evaluate_baselines, evaluate_formats, load_catalog
from .reporting import compare_evaluation, save_evaluation
from .artifacts import check_artifact
from .diagnostics import cardinality_report, cutoff_report, explain_case, pair_report, resource_balance_report, vocabulary_report
from .server import create_server
from .service import RequestError, browse, compare, explain, match, plan, plan_text, related, resource_detail


def main():
    parser = argparse.ArgumentParser(description="Street to School Guardian: fictional, consent-first support retrieval")
    parser.add_argument("--json-errors", action="store_true", help="Emit machine-readable runtime validation errors to stderr")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit catalog TF-IDF and save non-personal weights")
    train.add_argument("--output", default=str(Path("models") / "tfidf.json"))
    evaluation = commands.add_parser("evaluate", help="Evaluate held-out synthetic queries, never used for fitting")
    evaluation.add_argument("--output", help="Atomically write a synthetic benchmark report inside this repository")
    evaluation.add_argument("--model", help="Validate a saved artifact before evaluating it")
    comparison = commands.add_parser("compare-evaluation", help="Compare validated synthetic case ranks to a repository-local snapshot")
    comparison.add_argument("snapshot")
    commands.add_parser("evaluate-formats", help="Audit fixed benchmark labels under any, offline and online filters")
    commands.add_parser("audit-catalog", help="Find indistinguishable descriptions and synthetic vocabulary coverage gaps")
    commands.add_parser("evaluate-baselines", help="Compare fixed TF-IDF, lexical-overlap and always-abstain baselines")
    commands.add_parser("evaluate-vocabulary", help="Audit tokenizer coverage with explicit zero-usable denominators")
    commands.add_parser("evaluate-cardinality", help="Compare single and compound synthetic label slices")
    commands.add_parser("audit-pairs", help="Audit compound synthetic labels without family co-occurrence claims")
    commands.add_parser("evaluate-cutoffs", help="Compare bounded review-list sizes at the unchanged threshold")
    commands.add_parser("evaluate-resource-balance", help="Compare annotation-weighted and resource-weighted synthetic recall")
    inspect = commands.add_parser("inspect", help="Describe the fitted catalog or validate a saved artifact without queries")
    inspect.add_argument("--model", help="Validate this artifact by exact deterministic refit")
    check = commands.add_parser("check-artifact", help="Classify compatibility failures without echoing artifact contents")
    check.add_argument("artifact")
    serve = commands.add_parser("serve", help="Serve the dashboard on loopback only")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model", help="Validate a saved artifact; otherwise fit in memory")
    recommend = commands.add_parser("recommend", help="Read an anonymous example from stdin, not shell history")
    recommend.add_argument("--consent", action="store_true")
    recommend.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    recommend.add_argument("--limit", type=int, default=3, help="Return one through six fictional ideas")
    trace = commands.add_parser("explain", help="Explain bounded catalog exclusions using an anonymous example from stdin")
    trace.add_argument("--consent", action="store_true")
    trace.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    trace.add_argument("--limit", type=int, default=3)
    catalog_command = commands.add_parser("browse", help="Browse fictional metadata without a barrier description")
    catalog_command.add_argument("--consent", action="store_true")
    catalog_command.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    catalog_command.add_argument("--category")
    catalog_command.add_argument("--offset", type=int, default=0)
    catalog_command.add_argument("--limit", type=int, default=6)
    catalog_command.add_argument("--sort", choices=["id", "title", "category"], default="id")
    for name in ("compare", "plan"):
        selected = commands.add_parser(name, help=f"Create a fictional {name} from catalog IDs only")
        selected.add_argument("--consent", action="store_true")
        selected.add_argument("resource_ids", nargs="+")
        if name == "plan":
            selected.add_argument("--format", choices=["json", "text"], default="json",
                                  help="Readable synthetic checklist or default JSON; no file is written")
    neighbors = commands.add_parser("related", help="Explore bounded catalog-description similarities without a barrier")
    neighbors.add_argument("--consent", action="store_true")
    neighbors.add_argument("resource_id")
    neighbors.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    neighbors.add_argument("--limit", type=int, default=3)
    detail = commands.add_parser("resource", help="Inspect one fictional catalog ID without processing a description")
    detail.add_argument("--consent", action="store_true")
    detail.add_argument("resource_id")
    replay = commands.add_parser("explain-case", help="Replay one bundled synthetic benchmark ID")
    replay.add_argument("case_id")
    replay.add_argument("--consent", action="store_true")
    replay.add_argument("--mode", choices=["any", "offline", "online"], default="any")
    replay.add_argument("--limit", type=int, default=3)
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
            report = save_evaluation(model, args.output) if args.output else evaluate(model)
            print(json.dumps(report, indent=2))
        elif args.command == "evaluate-formats":
            print(json.dumps(evaluate_formats(model), indent=2))
        elif args.command == "audit-catalog":
            print(json.dumps(audit_catalog(model), indent=2))
        elif args.command == "evaluate-baselines":
            print(json.dumps(evaluate_baselines(model), indent=2))
        elif args.command == "evaluate-vocabulary":
            print(json.dumps(vocabulary_report(model), indent=2))
        elif args.command == "evaluate-cardinality":
            print(json.dumps(cardinality_report(model), indent=2))
        elif args.command == "audit-pairs":
            print(json.dumps(pair_report(model), indent=2))
        elif args.command == "evaluate-cutoffs":
            print(json.dumps(cutoff_report(model), indent=2))
        elif args.command == "evaluate-resource-balance":
            print(json.dumps(resource_balance_report(model), indent=2))
        elif args.command == "compare-evaluation":
            print(json.dumps(compare_evaluation(model, args.snapshot), indent=2))
        elif args.command == "inspect":
            print(json.dumps({**model.summary(), "source": "validated_artifact" if args.model else "in_memory_fit"}, indent=2))
        elif args.command == "check-artifact":
            report = check_artifact(args.artifact, catalog)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "valid" else 2
        elif args.command == "browse":
            print(json.dumps(browse(model, {
                "consent": args.consent, "mode": args.mode, "category": args.category,
                "offset": args.offset, "limit": args.limit, "sort": args.sort
            }), indent=2))
        elif args.command in ("compare", "plan"):
            operation = compare if args.command == "compare" else plan
            payload = {"consent": args.consent, "resource_ids": args.resource_ids}
            if args.command == "plan" and args.format == "text":
                print(plan_text(model, payload), end="")
            else:
                print(json.dumps(operation(model, payload), indent=2))
        elif args.command == "related":
            print(json.dumps(related(model, {"consent": args.consent, "resource_id": args.resource_id,
                                            "mode": args.mode, "limit": args.limit}), indent=2))
        elif args.command == "resource":
            print(json.dumps(resource_detail(model, {"consent": args.consent, "resource_id": args.resource_id}), indent=2))
        elif args.command == "explain-case":
            print(json.dumps(explain_case(model, args.case_id, args.consent, args.mode, args.limit), indent=2))
        elif args.command in ("recommend", "explain"):
            if not args.consent:
                raise RequestError("Use --consent only after agreeing to process an anonymous description.", 403, "consent_required")
            text = sys.stdin.read(801)
            operation = match if args.command == "recommend" else explain
            print(json.dumps(operation(model, {"text": text, "mode": args.mode, "limit": args.limit, "consent": True}), indent=2))
        else:
            try:
                server = create_server(model, args.port)
            except OSError:
                message = f"Could not start the local dashboard on port {args.port}. Choose another --port or use --port 0 for an available port."
                print(json.dumps({"error": message, "code": "port_unavailable"}) if args.json_errors else message, file=sys.stderr)
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
        message = str(exc) if isinstance(exc, RequestError) else "Could not load model, data, or start server. Check paths and port."
        if args.json_errors:
            print(json.dumps({"error": message, "code": exc.code if isinstance(exc, RequestError) else "local_data_failure"}), file=sys.stderr)
        else:
            print(message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
