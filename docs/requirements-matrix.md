# Coursework requirement matrix

The supplied ten-page coursework brief is authoritative. The student confirmed lecturer approval of the proposal. COMPLETE refers to delivered and verified technical evidence; it does not assert individual understanding or a completed live presentation.

| Requirement | Actual evidence | Status |
| --- | --- | --- |
| Q1: source/licence/size/target | data/README.md; data/manifests/data-audit.json; report Q1 | COMPLETE |
| Q1: preprocessing, leakage and groups | data/manifests/split.json / split.csv; shared data/images.py; integrity tests | COMPLETE |
| Q1: EDA and three modelling constraints | docs/eda-statistics.json / eda-modelling-findings.json; real figures | COMPLETE |
| Q1: consequences and primary metric | report Q1; evaluation/metrics.py; validation-only F2 selection | COMPLETE |
| Q2.a: layers, dimensions, parameters and diagram | actual architecture CSVs and run JSON; report Q2.a | COMPLETE |
| Q2.a: separate design/loss justifications | report Q2.a; same preprocessing in training/API/XAI | COMPLETE |
| Q2.b: alternatives, ANN suitability and prior hypothesis | report Q2.b; docs/pre-evaluation-hypothesis.md | COMPLETE |
| Q3.a: initial/final ANN and learning curves | all 21 ANN candidate records; reports/figures/initial-final-learning.png | COMPLETE |
| Q3.a: comparable classical tuning | 24 main configurations + four retained pilots; reports/experiment-log.csv | COMPLETE |
| Q3.a: references, final metrics and latency | reports/final-evaluation/comparison.json and per-model outputs | COMPLETE |
| Q3.a: meaningful full-population error analysis | reports/error-analysis.json; origin/mask-size/recommendation tables | COMPLETE |
| Q3.b: three hyperparameters and two optimisers | reports/optimisation-analysis.json; matched screening runs; report Q3.b | COMPLETE |
| Four unsuccessful attempts; two substantive approaches | docs/failed-approaches.json; completed run IDs and controls | COMPLETE |
| Q4: functional trained request-time ANN frontend/backend | app/; actual bundle/checkpoint; reports/application-verification/results.json | COMPLETE |
| Q4: score meaning, limitations, normal/invalid/OOD testing | UI/API labels; 13 actual HTTP checks; browser checks; unit tests | COMPLETE |
| Q5.a: three methods, cost/limitations and justified choice | report Q5.a: Grad-CAM, Integrated Gradients, LIME | COMPLETE |
| Q5.b: actual explanation UI and sanity checks | reports/xai-validation/; final-evaluation/xai/; matched-area top/low/random occlusion | COMPLETE |
| Three genuine incorrect predictions analysed with XAI | docs/three-xai-errors.json; three inspected original/mask/overlay panels | COMPLETE |
| Q6.a: real VLM, exact prompts and shared examples | experiments/vlm-validation-{0,1}.json; final-evaluation/vlm.json; immutable revision | COMPLETE |
| Q6.b: numerical comparison, costs/privacy/deployment/extension | report Q6.b; actual token/latency/failure logs | COMPLETE |
| Single split, validation-only selection, seeds and complete logs | final-evaluation-manifest.json; exclusive COMPLETE test-access guard; three frozen seeds | COMPLETE |
| Code, environment, executed notebooks and README | pinned pyproject/requirements-tested; four executed notebooks; verification.json | COMPLETE |
| Dataset >500 MB: download script and representative sample | verified publisher downloader; authentic 168 patches/84 masks; sample checksums | COMPLETE |
| Report <=25 main pages | reports/final_report.pdf; reports/pdf-verification.json | COMPLETE |
| Actual demonstration <=8 minutes | reports/demonstration.mp4; duration/decode checks; genuine incorrect prediction | COMPLETE |
| AI-use appendix | docs/ai-use-declaration.md; report appendix | COMPLETE |
| Strictly one physical held-out evaluation | Two identical frozen computations after result loss; docs/engineering-runs/final-results-workspace-reset.json | DISCLOSED EXCEPTION; lecturer acceptance unverified |
| Individual review, understanding and live defence | Student review and live presentation; docs/viva-guide.md | HUMAN TASK |

Updated 2026-10-12T11:28:41.834085+00:00. Real results and adverse findings are retained; weak performance is not concealed.
