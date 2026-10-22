# Coursework requirement matrix

This matrix was checked directly against the supplied ten-page **AI Coursework 2025 Batch** brief. The brief is authoritative; project notes are evidence only. `COMPLETE` means the reduced worktree retains technical evidence for the requirement. It does not predict a mark or assert the student's personal understanding.

| Brief requirement | Page | Retained evidence | Status |
| --- | ---: | --- | --- |
| Specific application, user, and changed decision/action | 1–3 | README; report introduction/Q1; app inspection recommendations | COMPLETE |
| Lecturer approval before main implementation | 1, 3 | `docs/approval-and-protocol.md` records student-confirmed approval; student must retain the real correspondence | STUDENT EVIDENCE REQUIRED |
| Dataset source, licence, structure, size, target, distribution, and limitations | 3, 5 | README; `data/sample/Readme.txt`; `data/manifests/data-audit.json`; report Q1 | COMPLETE |
| Appropriate preprocessing and every input transformation documented | 3, 5 | `src/fabric_inspector/data/`; checkpoint metadata; report Q1/Q2 | COMPLETE |
| Explicit duplicate, near-duplicate, unavailable-feature, and group leakage controls | 3–5 | fixed split JSON/CSV; audit; manifest code; report Q1 | COMPLETE |
| Comprehensive train-only EDA ending in three modelling constraints | 5 | EDA notebook/script; EDA JSON; two EDA figures; report Q1 | COMPLETE |
| False-positive/false-negative consequences and justified primary metric | 5 | report Q1; defect-class F2 implementation and validation selection | COMPLETE |
| ANN layers, dimensions, activations, parameter count, and diagram | 5 | architecture CSVs, run JSON/checkpoint metadata, and report Q2.a | COMPLETE |
| Architecture assumptions and individual depth/width/normalisation/regularisation/output/loss justifications | 5 | report Q2.a; shared preprocessing and BCE implementation | COMPLETE |
| Alternatives, ANN suitability, conditional choices, and pre-result ANN/classical expectation | 5 | report Q2.b; `docs/pre-evaluation-hypothesis.md` | COMPLETE |
| One fixed split for classical, ANN, and GenAI work | 3–4 | split hash in manifests, every run record, and frozen evaluation manifest | COMPLETE |
| Training/validation-only model, prompt, threshold, and hyperparameter selection | 4 | scripts; selection records; frozen evaluation manifest | COMPLETE |
| Fixed seeds; at least three runs and spread when variation is material | 4 | `artifacts/seed-variability.json`; final comparison | COMPLETE |
| ANN and classical searches with comparable tuning effort | 3–4, 6 | 23 ANN and 28 classical completed rows; declared search spaces and budgets | COMPLETE |
| Complete experiment log with changes, hyperparameters, validation result, lesson, and failures | 1, 4, 6 | `reports/experiment-log.csv`; run JSON; `docs/failed-approaches.json` | COMPLETE |
| Initial ANN, improvements, and interpreted learning curves | 6 | initial/final runs; learning-curve figure; report Q3.a | COMPLETE |
| Trivial reference, tuned classical, initial ANN, and final ANN with primary metric, training time, and inference latency | 3, 6 | shared held-out comparison JSON and report table | COMPLETE |
| Final-model error analysis by meaningful factors | 6 | `reports/error-analysis.json`; origin, mask-size, and recommendation analyses | COMPLETE |
| Three empirical hyperparameter studies and an empirical comparison of two optimisers | 6 | matched experiment rows, optimisation figure, and report Q3.b | COMPLETE |
| At least four unsuccessful attempts, including two substantive approaches, with diagnoses | 1, 6 | four completed run IDs and controls in `docs/failed-approaches.json` | COMPLETE |
| Functional frontend/backend using the trained ANN at request time | 6 | `app/`; selected bundle/checkpoint; measured application verification | COMPLETE |
| Intended-user input, score meaning, limitations, and normal/invalid/OOD tests | 6 | UI/API; application verification results; demonstration | COMPLETE |
| Compare three XAI approaches, including method type, explanation, cost, limitations, and selected-method justification | 7 | report Q5.a: Grad-CAM, Integrated Gradients, and LIME | COMPLETE |
| ANN explanation in the web UI with intuitive presentation and limitation language | 7 | Grad-CAM code, `/explain`, and UI | COMPLETE |
| Explanation sanity checks on at least two cases | 7 | full XAI validation JSON; matched-area top/low/random occlusion | COMPLETE |
| Incorrect predictions analysed with XAI; final checklist asks for three | 7, 9 | three real errors, panels, masks, scores, and perturbation results | COMPLETE |
| Suitable pretrained generative/foundation model; same examples/metric; validation-only prompts | 7 | two validation prompts; pinned SmolVLM; shared test comparison | COMPLETE |
| GenAI model/version/config/prompts/services/limitations/costs documented | 7 | VLM selection/run JSON, report Q6.a, and raw final decisions | COMPLETE |
| Evidence-based ANN/GenAI comparison of performance, latency, cost, data, privacy, dependencies, deployment conditions, and extension | 8 | final comparison and report Q6.b | COMPLETE |
| Reproducible code, environment, README, and executed notebooks | 4 | `src/`, `scripts/`, retained helpers, four notebooks, `pyproject.toml`, requirements, README | COMPLETE |
| Dataset over 500 MB: download procedure and representative testable sample | 4 | download script; authentic 168 patches/84 masks; sample manifests | COMPLETE |
| Report follows Q1–Q6 and has no more than 25 main pages | 4, 9 | final PDF: 24 main pages plus two appendix pages | COMPLETE |
| Demonstration is at most eight minutes and shows live prediction, XAI, and an incorrect prediction | 4, 9 | 130-second MP4 and application verification event log | COMPLETE |
| Generative-AI assistance declared in an appendix | 4, 9 | AI-use declaration and report Appendix A | COMPLETE |
| Final models evaluated only once on held-out test data | 4, 9 | identical frozen protocol was physically rerun after first result files were lost | DISCLOSED EXCEPTION; LECTURER ACCEPTANCE UNVERIFIED |
| Live presentation and ability to defend every submitted item | 5, 8–9 | `docs/viva-guide.md` assists preparation only | HUMAN TASK |

The reduced implementation does not conceal adverse results: the classical and VLM comparisons are weak, explanation checks include adverse outcomes, and the held-out rerun is disclosed. These facts must remain in the submission.
