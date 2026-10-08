# Street to School Guardian
### AI for Education · A bridge to support, not a label

An offline-capable, consent-first ML prototype that matches **anonymous, self-reported barriers to learning** with **fictional education support ideas**. A family or trusted adult controls the description, can decline suggestions, and can withdraw at any time.

The ambition is a community-owned bridge between unmet learning needs and verified human support, not a system for monitoring children. The current deliverable is a working local demonstration, not a deployed service or a claim of educational impact.

**No surveillance. No dropout-risk labels. No eligibility determinations. No automated punitive decisions.**

## What works today

- A polished, responsive light/dark dashboard with sample presets, anonymous barrier input, support-format preferences and a consent gate.
- Actual Python inference behind a loopback HTTP server; the frontend does not simulate the model.
- A trainable, dependency-free TF-IDF cosine retriever fitted on 12 synthetic support descriptions.
- Ranked ideas with shared-term explanations, honest no-match states, and an explicitly simulated human handoff.
- Reproducible held-out relevance evaluation, including difficult paraphrases and multilingual failure cases.
- Python unit, CLI and real HTTP integration tests; optional real-browser smoke testing without npm dependencies.
- No external APIs, weights, fonts, scripts, CDNs, analytics, account systems or keys.

**All supports are fictional.** “Free,” format, and language fields describe the demo catalog only. No organization, institution, contact, current availability or real referral is claimed.

## Quickstart

Requires **Python 3.11 or newer**. No packages need installing. Run commands from this repository's root.

```powershell
Set-Location C:\Users\emutisya\global-change-ml\street-to-school-guardian
python -m guardian train
python -m guardian evaluate
python -m unittest discover -s tests -v
python -m guardian serve --model models\tfidf.json
```

Open **http://127.0.0.1:8765**. Tick consent, choose a sample, explore ideas, then preview the human handoff. Stop the server with `Ctrl+C`.

On other operating systems, change into your cloned repository first and use `models/tfidf.json` for the model path. The Python implementation itself is portable.

You can also run `python -m guardian serve` without an artifact; it fits the synthetic catalog in memory at startup. `--port 0` chooses an available port and prints its URL. The saved artifact is checked against the current catalog and a deterministic refit before its weights are used. This is a correctness check, not a startup optimization.

### CLI inference

Use synthetic text on standard input rather than putting private details in shell arguments:

```powershell
'Bus fares and buying notebooks make learning difficult.' | python -m guardian recommend --consent --mode offline
```

`--consent` is explicit agreement to process an anonymous description. Without it, inference is refused and stdin is not read. This is not consent on behalf of another person or permission to enter personal information.

### Optional browser validation

Requires an already installed Chromium-based Edge/Chrome and Node 22+. No downloads or dependencies are needed.

```powershell
$env:GUARDIAN_BROWSER = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
node tests\browser_smoke.mjs
```

The test starts its own loopback server and isolated browser, exercises real dashboard inference, handoff, consent withdrawal, invalid/no-match states, late-response protection, mobile layout and themes, and closes them afterward. Screenshots go into ignored `browser-validation`; its browser profile is removed. It also checks that the page makes no external requests and uses no browser storage. This is not a full accessibility or security audit.

Observed local validation: **18 Python tests passed** on Windows / Python 3.14.3. The real Edge browser smoke test passed twice consecutively, including request cancellation and responsive light/dark screenshots. Saved-artifact CLI server startup, health and HTTP inference also passed. The GitHub Actions matrix is configured but has not yet run on GitHub; no repository has been created or published by this implementation.

## Architecture

```text
Synthetic catalog descriptions
       │ fit vocabulary + inverse document frequencies
       ▼
Normalized TF-IDF resource vectors ───── optional ignored JSON artifact
       │
Anonymous text + explicit consent + preferred format
       │ validation (no payload logs, no database)
       ▼
Local Python HTTP handler → cosine similarity → threshold + format filter
       │
       ▼
Dashboard: ranked fictional ideas + shared terms
       │ explicit human choice, consent checked again
       ▼
Handoff simulation: a verification checklist only; nothing sent or booked
```

| File / directory | Responsibility |
| --- | --- |
| `guardian/model.py` | Tokenization, fitting, persistence validation, retrieval and evaluation |
| `guardian/service.py` | Strict consent gate, anonymous-input backstop, matching and simulated handoff |
| `guardian/server.py` | Loopback-only HTTP routes and self-contained dashboard delivery |
| `guardian/__main__.py` | Train, evaluate, recommend and serve commands |
| `guardian/data/resources.json` | 12 entirely fictional catalog entries |
| `guardian/data/evaluation.json` | Held-out synthetic relevance and challenge judgments |
| `guardian/web/index.html` | Inline HTML, Clawpilot theme, CSS and JavaScript; no remote assets |
| `tests/` | Behavioral Python tests and optional browser integration |
| `.github/workflows/ci.yml` | Training, evaluation and tests on Windows/Linux, Python 3.11/3.14 |

The HTML is self-contained as a frontend. Opening it directly is a read-only preview; **working inference requires the Python server**. The UI gives setup instructions rather than substituting fake client-side results.

### HTTP contract

All routes are local only. There is no CORS or public API.

- `GET /` serves the dashboard.
- `GET /health` reports server health and catalog size, not user information.
- `POST /api/match` accepts exactly `{"consent": true, "text": "Bus fares are expensive.", "mode": "offline"}`. Mode is `any` (default), `offline` or `online`.
- `POST /api/handoff` accepts exactly `{"consent": true, "resource_id": "transport"}`. It returns a static human verification checklist, not a referral.

Consent must be the JSON boolean `true`, not `"true"` or `1`. Descriptions must contain 1–800 nonblank characters; request bodies are limited to 4,096 bytes. Unexpected fields, obvious identifiers, malformed JSON, invalid preferences and unknown resources are rejected. Empty/OOV or low-similarity queries produce **no match**, never a denial of assistance.

Responses do not echo the submitted description. Invalid input returns 400, missing consent 403, unknown routes/resources 404, oversized bodies 413, and unsupported content types 415. Host/Origin checks and query-string rejection (except the dashboard's explicit light/dark theme parameter) reduce cross-site use of the local server; they are not a substitute for production security.

## ML and data card

**Task:** resource-description retrieval, not a classifier of people. The model never predicts attendance, dropout, vulnerability, urgency, deservingness or future behavior.

**Training:** catalog descriptions only. Fitting learns a vocabulary, document frequencies, inverse document frequency weights and resource vectors. It is a real classical unsupervised ML / information-retrieval baseline, not a hard-coded keyword router or LLM.

**Algorithm:**

1. Lowercase and extract ASCII letter tokens; omit short tokens and a fixed documented-in-code stopword set. Generic words such as “learning,” “school,” and “support” are excluded to avoid arbitrary generic matches.
2. For term `t`, use `idf(t) = log((1 + N) / (1 + df(t))) + 1`.
3. Use sublinear term frequency `tf(t) = 1 + log(count(t))`, multiply by IDF and L2-normalize each document and query vector.
4. Rank by cosine similarity, filter by the chosen fictional format, and retain up to three results scoring at least **0.12**. Ties use the stable resource ID.
5. Explain the largest shared-term products. Unknown vocabulary contributes nothing; an all-unknown query abstains.

The fixed threshold is a demo heuristic, **not a calibrated safety boundary or probability cutoff**, and was not selected by optimizing this evaluation set. A rare-word overlap can still be misleading. Scores are not probabilities, need scores, eligibility scores, urgency or confidence in a service.

**Data provenance:** all catalog text, queries and relevance labels were manually authored for this project as synthetic examples. No scraped records, real children, names, birth dates, GPS, institutions or service contact claims. Data is distributed with the repository under MIT. Mode/language/cost are fictional catalog metadata, not eligibility filters.

**Split:** 12 catalog descriptions are used for fitting. Separately written query strings and relevance labels are held out from fitting: 26 catalog-like queries (including two multi-barrier examples), six unmatched controls, and eight harder paraphrase/multilingual probes. Evaluation checks for exact description/query overlap and duplicate queries. This prevents exact-text duplicates, **not lexical or conceptual overlap**: primary queries deliberately share much of the catalog vocabulary. The challenge set exposes the resulting optimism. None of the relevance labels are used in training.

### Observed evaluation

Reproduced locally with `python -m guardian evaluate` on Python 3.14.3:

| Metric | Observed result | What it means |
| --- | ---: | --- |
| Top-1 hit rate, 26 catalog-like queries | 1.000 | First result is one of the editorially relevant supports |
| Mean recall@3, same 26 queries | 1.000 | Mean fraction of labeled relevant resources appearing in the first three results |
| MRR@3, same 26 queries | 1.000 | Mean reciprocal position of the first relevant result; zero if absent |
| Unmatched rejection, six controls | 1.000 | No result returned for unrelated or generic text |
| Challenge hit@3, eight probes | **0.125** | At least one relevant result appears for just one harder query |
| Challenge abstention, eight probes | **0.875** | No result returned for seven harder queries |

Training learns **145 vocabulary terms** from 12 descriptions. Catalog SHA-256:

```text
dd2bb36d10f3ae03de56d33ba2d36bb401c39d3b122c88b6f079eb989ab9eb5b
```

Perfect performance on 26 vocabulary-aligned synthetic examples is **not** evidence of field accuracy, fairness, generalization or social impact. These small samples have no population representativeness or statistical assurance. The challenge results show severe synonym and multilingual limitations. English-only, bag-of-words retrieval cannot reliably interpret negation, context, intent or complex needs. Preference-specific relevance, ranking fairness, field usability and actual service fit have not been evaluated.

## Privacy and safeguarding

- **Consent before text processing:** both CLI/HTTP and the dashboard enforce it. Catalog fitting does not use submitted text. Every handoff preview requires consent again.
- **Withdrawal:** unticking consent or resetting clears page input, results and checklist, aborts pending browser fetches, and ignores late responses. Editing a barrier or preference invalidates old recommendations.
- **No personal data retention by the application:** no database, payload logs, accounts, cookies, localStorage, sessionStorage, telemetry or interaction history. Only synthetic model weights may be saved. Input exists transiently in browser/server memory to process a request and is not written to disk by application code.
- **Not secure erasure:** cancellation cannot undo processing already received by the server. Python/JavaScript garbage collection, browser internals, OS swap/crash dumps and terminal capture are outside the application's guarantees. Do not enter actual personal information.
- **Input backstop, not anonymization:** numbers (including dates/coordinates), email markers, links and control characters are rejected. Names, indirect identification and sensitive facts **cannot be reliably detected**. The user must supply only general anonymous or synthetic descriptions.
- **Minimal handoff:** the checklist receives a catalog ID, not the barrier text. Nothing is transmitted to a service, saved, booked or automatically copied. A future real referral must obtain fresh, specific consent.
- **No child surveillance or labeling:** there are no identity fields, attendance imports, location data, person profiles, risk scores or disciplinary/eligibility endpoints. Catalog matching must never be repurposed into those uses.
- **Human authority:** suggestions are optional conversation starters. A trained human must verify real availability, suitability, accessibility, safety, costs and actual service requirements. Lack of a match does not imply lack of need.
- **Not crisis/clinical support:** this system does not detect emergencies, assess abuse, diagnose conditions or alert authorities. Seek appropriate trusted local help outside the demo in an emergency.

The built-in HTTP server binds only to `127.0.0.1`. It has no TLS, authentication, rate limiting or production threat model. **Public-ready means the code and documentation can be published; do not expose this server to the internet, reverse proxy it, or use it with real families.** Real deployments require community governance, independent safeguarding/privacy/security review and jurisdiction-specific legal review.

## Limitations and responsible use

Resource matching is only one small part of improving education access. It cannot create transport, meals, safe services, funding, trust or institutional accountability. Fictional formats and “free” labels cannot substitute for verified real-world availability.

Use this repository for learning, synthetic demonstrations and community design discussions. Do not use it for selection, denial of services, school discipline, child monitoring, crisis response or real referral decisions. The model is intentionally transparent and lightweight, but that does not make it automatically safe or equitable.

CI verifies program behavior, not social outcomes or production readiness. The MIT license grants software rights; it does not certify fitness for a safeguarding use case.

## Ambitious, credible global roadmap

These are proposed stages, **not delivered capabilities, partners or measured impact**:

1. **Community ownership and a service-data contract.** Co-design with families, adult facilitators and safeguarding specialists. Define prohibited uses, complaint paths, withdrawal, service-owner verification, expiry/revalidation, clear cost/availability fields and minimum-data referral rules. No real listing goes live without permission and validation.
2. **Offline-first community networks.** Package verified regional catalogs as signed, expiring offline bundles for locally operated kiosks or facilitator devices. Keep retrieval local and make stale availability explicit. Begin with adult-mediated, small consented pilots, not school monitoring.
3. **Local-language access.** Build consented, anonymized, community-reviewed relevance benchmarks covering dialects, synonyms, mixed languages, accessibility and negation. Compare character n-grams or local multilingual models against this baseline; distribute any future weights with transparent licensing and offline execution. Do not claim support for a language before testing it.
4. **Human-reviewed service navigation.** Introduce an opt-in handoff only after humans verify availability and safeguards. No automatic enrollment, allocation or eligibility decision. Let families review and choose the minimum information shared directly with a verified provider.
5. **Evaluate benefit before scaling.** Assess retrieval errors, no-match rates by language/barrier, accessibility, usability, correction paths and service-data freshness. Use separately consented, minimized aggregate evaluation to study whether families actually reach useful support, not to label individuals. Publish uncertainty, adverse effects and opt-out experiences.
6. **Federated community stewardship.** Pursue interoperable local catalogs maintained by communities rather than a global database of children. Cross-region learning should share openly licensed descriptions and evaluation methods, not personal histories. Expansion is contingent on independent review and demonstrated benefit.

## Contributing

Keep all examples synthetic and anonymous. Add held-out query judgments when extending the catalog, do not copy training descriptions into tests, report challenge failures, and run the smallest relevant behavioral tests followed by evaluation. Preserve consent gating, no-retention behavior and fictional-service disclosure. Do not add external calls, tracking or identifying input fields.

Retrain when descriptions or tokenization change; ignored model outputs should not be committed. Secrets, virtual environments, caches, screenshots and model outputs are excluded by `.gitignore`.

## License

[MIT](LICENSE). Built by Elizabeth Mutisya as an open, local-first AI for Education prototype.
