# Coursework requirement matrix

Authoritative source: the supplied **AI Coursework 2025 Batch(2).pdf**, all ten pages. Lecturer approval is confirmed by the student's message. Status applies to this revision, not historical unverified artifacts.

| Criterion | Marks | Implementation / evidence | Current status |
| --- | ---: | --- | --- |
| Q1: source, licence, target, size, limitations | 10 total | `data/README.md`, acquisition/import audit, manifest | Publisher facts verified; approved bytes unavailable |
| Q1: preprocessing and leakage | — | shared `data/images.py`, grouped `data/manifest.py`, import/integrity tests | Implemented; development verified; real-release audit pending |
| Q1: comprehensive EDA and three constraints | — | `scripts/eda.py`, generated plots/findings | Development only; real EDA pending |
| Q1: costs and primary metric | — | report Q1, validation F2 policy and assumed recommendation costs | Defined; operational assumptions not company-validated |
| Q2.a: layers, dimensions, activations, parameters, diagram | 10 | `models/neural.py`, actual hook table, report Q2 | Custom/transfer architectures executable; final architecture awaits selection |
| Q2.a: separate design/loss justifications | — | report Q2, architecture guide | Draft provided; final evidence pending |
| Q2.b: alternatives and ANN suitability | 10 | report Q2, pre-evaluation hypothesis | Written before final test; student review required |
| Q3.a: initial/final ANN and learning curves | 10 | `training/runner.py`, selection, generated curves | Authentic development run only; full study NOT RUN |
| Q3.a: fairly tuned classical baseline | — | 24-configuration LBP/HOG SVM search, train-only scaler | One development smoke run verified; real search NOT RUN |
| Q3.a: reference comparison, timing and error analysis | — | frozen `final_evaluate.py`, group bootstrap, per-example records | Source implemented; approved final evaluation NOT RUN |
| Q3.b: three hyperparameters | 10 | 20 ANN configurations: learning rate, batch size, actual block depth | Search defined; real empirical conclusions PENDING |
| Q3.b: two optimisers | — | AdamW/SGD momentum, separate LR ranges | Implemented; real empirical comparison NOT RUN |
| Q3.b / section 2: four unsuccessful attempts | — | Actual run logs plus `docs/failed-approaches.json` after diagnosis | PENDING; no fabricated failures |
| Q3.b: two substantive failed approaches | — | Diagnoses citing real curves, controlled changes and run IDs | PENDING |
| Q4: trained ANN, frontend/backend, request-time prediction | 10 | FastAPI + local UI + trained development checkpoint, integration tests | Working on synthetic development data; real deployment model PENDING |
| Q4: score meaning, limitations, invalid/OOD inputs | — | Score labels, input validation, blank/noise checks, explicit OOD limitation | Implemented/tested; no reliable OOD-detection claim |
| Q5.a: three XAI methods | 5 | Grad-CAM, Integrated Gradients, LIME comparison | Draft provided |
| Q5.b: explanation UI and validation | 15 | Actual Grad-CAM, opacity, spatial metrics and top/low/random occlusion | Development execution verified; real explanations PENDING |
| Q5.b: at least two sanity-check cases | — | 28 development validation cases with measured occlusion | Development only; assessed real cases PENDING |
| Final checklist: three genuine incorrect predictions with XAI | — | Three inspected development errors in `docs/development-xai-cases.md`; real error-analysis workflow | Development examples only; assessed real case studies PENDING |
| Q6.a: real generative/foundation model | 10 | Local SmolVLM adapter, fixed revision, image input, response logs | Adapter implemented; real benchmark access/status in progress report |
| Q6.a: shared examples, primary metric, validation prompts | — | validation prompt selection; final guard shares full test IDs; failures retained | Implemented; approved-dataset execution NOT RUN |
| Q6.b: evidence-based comparison and extension | 10 | report Q6, recorded latency/token/service dependencies | Qualitative draft; numerical comparison PENDING |
| Single split / test isolation / seeds / complete logs | — | hashes, exclusive test-access guard, seeds 42/123/2026, actual records | Development verified; real full evidence PENDING |
| Code, environment, notebooks, README | — | modular source, pinned environment, four notebooks and tests | Delivered; exact verification in `verification.json` |
| Dataset/sample if data exceeds 500 MB | — | Download procedure and audit; included sample is explicitly artificial | Representative authentic sample PENDING |
| Report at most 25 pages excluding appendices | — | Q1–Q6 draft; PDF generation after results | Draft only; final page-checked report PENDING |
| Demonstration video at most 8 minutes | — | `docs/demonstration-storyboard.md` | Recording PENDING; storyboard is not a video |
| AI-use appendix and individual defence | — | `docs/ai-use-declaration.md`, viva guide | Draft provided; student verification PENDING |

No metric, historical figure, simulated VLM result or unexecuted source file is treated as evidence that a requirement is complete.
