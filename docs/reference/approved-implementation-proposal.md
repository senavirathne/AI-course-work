\## 1. Your role and objective



You are a senior machine-learning engineer, computer-vision researcher, and coursework implementation partner. Build a reproducible individual coursework project titled:



\*\*Risk-Aware, Explainable Fabric Defect Detection for Textile Quality Control\*\*



Read the attached coursework brief in full before making changes. It is authoritative if this prompt conflicts with it. Produce working code, real experiments, a usable application, and evidence that the student can understand and defend. Do not merely propose an architecture or generate disconnected notebooks.



The aim is strong evidence for every assessment criterion, not guaranteed marks or an ANN victory. A classical model or foundation model may outperform the ANN; report that honestly.



\### Application, user, and decision



- \*\*Application:\*\* Inspect an image patch of fabric for visible manufacturing defects.
- \*\*Intended user:\*\* A Quality Assurance Inspector in a garment or textile manufacturing operation, such as d.s garments. This is an application scenario; do not claim a company partnership, supplied company data, or validated production deployment.
- \*\*Decision supported:\*\* Whether the photographed fabric section should continue through routine inspection, receive manual review, or be held for priority inspection.
- \*\*Core prediction task:\*\* Binary patch classification: \`defective = 1\`, \`defect_free = 0\`.
- \*\*Explanation:\*\* Highlight image regions that influenced the defect prediction, with an explicit explanation of what the heatmap can and cannot establish.
- \*\*Boundary:\*\* The inspector retains the decision. A patch prediction does not establish the quality of an entire roll or justify automatically stopping machinery.



Keep binary classification as the assessed core. Segmentation, defect-type classification, production integrations, roll-level scoring, and additional datasets are optional extensions only after the assessed core is complete and any required approval is obtained.



\## 2. Read the brief and establish approval



First inspect the workspace, existing code, environment, and project instructions. Preserve useful existing work. Create \`docs/requirements-matrix.md\` mapping every requirement to its implementation, evidence, and current status.



Use the actual marking scheme:



\| Component | Marks | Required evidence |

\| --- | ---: | --- |

\| Q1: Dataset and EDA | 10 | Provenance, EDA, leakage checks, decision costs, metric justification, three findings that constrain modelling |

\| Q2.a: ANN architecture | 10 | Layers, tensor dimensions, activations, parameter count, diagram, individual design justifications |

\| Q2.b: Alternatives | 10 | Meaningful alternatives, rejection reasons, ANN justification, prediction of which model family will win before final evaluation |

\| Q3.a: Development and evaluation | 10 | Initial and final ANN, tuned classical baseline, reference benchmark, learning curves, timing, error analysis |

\| Q3.b: Optimisation | 10 | Three hyperparameters, two optimisers, genuine unsuccessful experiments and diagnoses |

\| Q4: Web application | 10 | Frontend, backend, request-time ANN inference, score interpretation, input testing, limitations |

\| Q5.a: XAI discussion | 5 | At least three methods, including architecture-specific and model-agnostic approaches |

\| Q5.b: XAI implementation | 15 | Application integration, explanation validation, incorrect-prediction analysis |

\| Q6.a: Generative/foundation model | 10 | Real application of a suitable model, fixed prompts, reproducible configuration, fair evaluation |

\| Q6.b: Comparative analysis | 10 | Evidence on performance, latency, cost, data needs, privacy, dependencies, deployment suitability, extension opportunity |



The brief requires lecturer approval before substantial implementation. If approval is absent, complete a lightweight dataset feasibility audit, requirements matrix, proposed protocol, and \`docs/lecturer-approval-proposal.md\`. Then stop before substantial model or application implementation and request the missing approval. Do not claim this stage completes the coursework. If approval is already documented, continue without asking for it again.



The approval proposal must identify the problem, intended user, decision, dataset/version, modality, target, proposed supervised split, and meaningful Q6 comparison. Check the restriction against datasets with end-to-end tutorials that substantially solve the assignment; being absent from the named banned list is insufficient. Record relevant existing implementations and seek a lecturer decision where suitability is uncertain. Changing the approved problem or dataset requires renewed approval.



\## 3. Dataset: verify RAW-FABRID before relying on it



Preferred source: [RAW-FABRID, Mendeley Data, version 1]\(https\://data.mendeley.com/datasets/db6g85xsyg/1), DOI \`10.17632/db6g85xsyg.1\`.



The publisher reports 709 original grayscale images, 204 defect masks, and 19,852 patches at 256 x 256 pixels. It lists patch-to-source CSV metadata, COCO annotations, and a CC BY 4.0 licence. The supplied MVTec-style training directory contains only normal patches; defective patches occur in the supplied test directory. Treat these as published expectations and verify them against the downloaded release.



\### Audit and provenance



1. Obtain the release through its documented public download route. Record source, version, retrieval date, licence/attribution, archive checksum, and file checksums or a reproducible manifest.

2\. Inspect actual schemas, image dimensions, mask alignment, label consistency, duplicate content, corrupt files, and metadata completeness. Do not invent column names, defect categories, manufacturer identifiers, roll IDs, or capture-session information.

3\. Establish what each grouping field actually represents. Fabric type, manufacturer, original image, and physical roll are not interchangeable concepts.

4\. Define a defective patch from its verified annotation, normally any annotated defect pixel. A normal patch can come from a defective original. Do not label every patch from a defective parent as defective.

5\. Do not silently interpret a missing mask as a normal example. Confirm normal status from the documented annotation convention and metadata; quarantine unexplained mismatches.

6\. Report independent original/group counts as well as patch counts. Explain why many related patches do not equal many independent production observations.

7\. Record class counts, defect sizes, image quality, acquisition limitations, and available grouping information in \`docs/data-card.md\`.



\### Supervised protocol and leakage prevention



The published normal-only training arrangement cannot directly train the proposed supervised classifier. Before fitting any model, propose and obtain approval for a \*\*new supervised research split\*\* of the complete labelled collection. State that its results are not directly comparable with the publisher's original anomaly-detection protocol.



- Construct one immutable train/validation/test assignment from source groups before cropping, augmentation, resampling, or model development.
- Default to approximately 70/15/15 only if the verified groups and positive counts support it. Choose the protocol for adequate independent coverage, never for favourable model scores. Record the actual ratios and constraints.
- Keep every patch, crop, augmentation, and exact/near-duplicate of a source image in one partition. Where verified physical roll/capture relationships exist, prevent those relationships crossing partitions too.
- Do not automatically make every fabric type or manufacturer a separate partition group. Explain whether the claim is generalisation to new images, new rolls, or unseen fabric types and design the split accordingly. Document what cannot be established from available metadata.
- Inspect duplicates and group integrity before freezing the split. Persist identifiers, labels, source IDs, known group IDs, duplicate-cluster IDs, split assignment, and split hash in a manifest.
- Train all classical and neural variants and evaluate the VLM using this same assignment. Never split randomly by patch after cropping.
- Fit scalers, learned transformations, feature selectors, class weights, and samplers using training data only. Augment training images only.
- Masks may supply target labels and explanation-evaluation ground truth; they must not become classifier input. Do not expose labels through filenames, directory names, metadata, or prompts at inference.
- Perform model-driving EDA on training data. Use validation data for development decisions. Limit test access before final evaluation to necessary, documented integrity checks and split accounting; do not browse test examples to guide design.
- Freeze the test set before training. Use it only in one planned final evaluation of all frozen comparison models. Never revise the split, model, prompt, thresholds, or preprocessing in response to test performance.



If source mapping or group information is insufficient to support a defensible split, document the limitation and resolve it before training. Do not silently switch datasets or conceal leakage.



\## 4. Q1: EDA and decision-aware evaluation



Produce reproducible plots and a concise EDA notebook/script covering class balance at patch and source levels, defect-area distribution, available fabric groups, brightness/contrast, texture variation, representative training examples, image/mask quality, and duplicates.



End EDA with exactly three evidence-backed findings that materially constrain modelling. Each must link an observed result to a choice, such as preserving spatial detail for tiny defects, restricting model capacity, using class weighting, or grouping related captures. Do not invent the findings in advance.



Use \*\*defect-class F2-score\*\* as the proposed primary metric, justified by an assumed higher cost for missed defects than unnecessary manual inspection. Explicitly label this as a coursework assumption unless real operational costs are available. Explain that F2 emphasises recall but is not a complete economic cost model.



Also report precision, recall, F1, confusion matrix, false-positive rate, missed-defect count, PR-AUC/average precision where a suitable continuous score exists, and ROC-AUC where informative. Define positive class, zero-division behaviour, and each metric consistently. Do not invent a probability or ranking score for a VLM that only emits labels.



Select each model's classification threshold on validation data using the same declared policy. Evaluate ANN score calibration using reliability plots and Brier score where feasible; any calibration must be fitted outside the test set. Use source/group-level bootstrap uncertainty estimates if feasible, rather than treating related patches as independent observations.



Keep the assessed binary prediction separate from the proposed three-way inspection recommendation. For the recommendation, document two validation-selected cut-offs and any assumed review costs/capacity: low-score routine inspection, intermediate manual review, high-score priority hold. Evaluate review burden and missed defects among low-score cases. Do not call the middle band statistically validated uncertainty without evidence, or treat a low score as a guarantee of defect-free fabric.



\## 5. Q2-Q3: ANN, baselines, and controlled experiments



\### Models



Implement:



1. \*\*Trivial references:\*\* Always-normal and always-defective predictors, so class imbalance and the behaviour of F2 are visible.

2\. \*\*Classical baseline:\*\* A properly tuned SVM using LBP/HOG texture features or a justified combination. Fit its preprocessing on training data only. Tune meaningful feature and classifier settings, including regularisation and class weighting where applicable.

3\. \*\*Initial ANN:\*\* A small, understandable convolutional classifier that establishes a reproducible neural baseline.

4\. \*\*Final ANN candidate:\*\* A compact pretrained CNN, initially MobileNetV3-Small or another justified compact backbone. Fine-tuning is encouraged by the brief. Select the final model using evidence, not architecture prestige; retain the initial model if it is the stronger defensible choice.



Implement a consistent 256 x 256 patch pipeline. For an RGB-pretrained backbone, explicitly justify grayscale channel replication and compatible normalisation, or document and test a principled first-layer adaptation. Avoid aggressive resizing or augmentations that erase small defects. Justify each augmentation in terms of fabric acquisition.



Use a binary logit with a compatible loss, such as \`BCEWithLogitsLoss\`. Estimate any positive-class weight from training labels only; do not apply multiple imbalance corrections without testing and justification.



Produce a diagram and actual layer/tensor-shape table, activations, total/trainable parameter counts, output/loss formulation, and separate justifications for depth, width, normalisation, regularisation, transfer learning, and spatial resolution. Explain feature/receptive-field assumptions and limitations for tiny defects.



Discuss meaningful alternatives, including texture-based classical methods, a CNN trained from scratch, transfer learning, a transformer, and anomaly detection/segmentation. For important rejected alternatives, state both when they would be appropriate and what makes them less suitable here. Record the expected ANN-versus-classical winner and reasoning before final test evaluation.



\### Experiment design



- Empirically investigate \*\*learning rate, batch size, and network depth\*\*. Use a clear depth ablation of the custom CNN, for example two/three/four convolutional blocks; do not relabel the number of unfrozen pretrained layers as network depth. Other assessed parameters require lecturer approval if outside the brief's permitted choices.
- Compare \*\*AdamW and SGD with momentum\*\* on otherwise comparable configurations. Give each an appropriate learning-rate search; an identical unsuitable rate is not a fair comparison.
- Record initial and improved learning curves, early-stopping behaviour, training/validation loss, and validation metrics. Diagnose underfitting, overfitting, and other limitations from actual evidence.
- Predeclare comparable classical/ANN search budgets, initially around 12-24 meaningful configurations per family if resources permit. Count all substantive searches and ablations transparently; revise budgets before test access if needed. Report configuration counts and compute time, not just selected runs.
- Fix and report seeds. Assess seed variability; if it materially affects conclusions, report mean and spread over at least three runs. Predetermine any repeated final runs before accessing test results.
- Record at least \*\*four genuine unsuccessful attempts overall\*\*, with at least \*\*two substantive failed approaches\*\* analysed in depth. Use plausible hypotheses and controlled experiments. Do not fabricate failures, deliberately sabotage configurations, or count trivial syntax errors as substantive scientific failures. If the evidence requirement remains unmet, report the gap and investigate further within an explicit budget.
- Include model improvements, alternative features, capacity choices, augmentation strategies, and imbalance treatments only where justified; avoid a large architecture sweep unrelated to the rubric.



Maintain machine-readable experiment records and a readable log. Each substantive run must include run ID, timestamp, code revision, environment/device, dataset/split hash, seed, hypothesis, change, configuration, optimiser, loss, epochs/stopping rule, validation results, runtime, artifact paths, outcome, and lesson learned. Every reported result must point to a real run.



\### Final evaluation and error analysis



Before opening the test set, freeze checkpoints, features, preprocessing, thresholds, calibration, selected VLM/version, prompts, parsing rules, and evaluation code in \`artifacts/final-evaluation-manifest.json\`.



Execute one controlled final evaluation and retain predictions per example. Compare the references, tuned classical model, initial ANN, final ANN, and VLM on the same held-out examples and primary metric. State hardware, batch size, warm-up, sample count, and whether latency includes preprocessing or network overhead; distinguish model-only and end-to-end timing where relevant.



Report final error patterns by actual available groups, defect size, texture, image quality, and false-positive/false-negative type. Analysis after final evaluation must not become further tuning. If the baseline wins, retain the required ANN application and explain the result.



\## 6. Q5: Explainability with measurable validation



Discuss at least \*\*Grad-CAM, Integrated Gradients, and LIME\*\*, covering what each explains, architecture dependence, computational cost, sensitivity, and limitations. LIME supplies a model-agnostic comparator. Implement Grad-CAM for the selected CNN and one perturbation/occlusion validation routine; implementing all three methods is optional.



Requirements:



- Generate Grad-CAM from the actual selected checkpoint and a justified convolutional layer, targeting the defect score. Check hook/gradient behaviour and spatial alignment with the displayed image.
- Present the original patch and overlay together, with an opacity control and a short explanation for the inspector.
- State that the heatmap indicates model attribution, not a certified defect boundary or proof of causal reasoning. A classification explanation is not a segmentation result.
- Use available masks to measure spatial correspondence, such as maximum-attribution pointing accuracy and attribution mass inside the defect region. Handle empty masks separately and compare against random/uniform or area-matched references. Report sensitivity to defect size and coarse feature-map resolution.
- Do not treat mask overlap alone as explanation faithfulness. Occlude important regions and compare score changes against matched-area low-attribution/random controls. Document that occlusion itself may create unnatural inputs.
- Sanity-check at least two cases. Include \*\*at least three genuine incorrect predictions analysed with XAI\*\*, following the stricter final checklist even though Q5's body mentions one. Cover false positives and false negatives where available.
- Clearly label whether each case comes from validation or final test. If fewer than three errors are available, disclose the limitation and seek guidance; never manufacture cases or change a good model solely to obtain errors.
- Select explanation settings using training/validation evidence. Final test masks may be used for frozen explanation evaluation and reporting, not method tuning.



\## 7. Q6: Real vision-language model comparison



Apply a genuinely image-capable pretrained generative vision-language model to the same binary classification task. A text-only LLM without image input, a chatbot describing CNN output, or CLIP alone does not fulfil this proposed direct generative comparison.



Choose an exact model/checkpoint after checking runtime, memory, licence, availability, and costs. Prefer a feasible local model where practical; support a configured API alternative if appropriate and authorised. Do not assume that API credentials, paid credits, or a GPU are available.



Implement a reusable evaluation adapter that:



- Supplies the same underlying image content and the same class definitions used for ANN evaluation, documenting unavoidable model-specific resizing.
- Sends no ground-truth label, mask, revealing filename, ANN prediction, or test-derived hint.
- Requests a short structured output containing \`defective\` or \`defect_free\`; does not demand an invented confidence value.
- Selects and freezes prompts using validation data only. Few-shot examples, if used, come from training data and remain outside the held-out test.
- Logs the exact model/version, quantisation where relevant, image processor, prompt template, generation settings, requests/responses, parsing errors, latency, token/API usage, and date.
- Uses a predeclared bounded retry/fallback policy. Includes unresolved outputs in denominators and reports failure rate; never silently drops hard cases to improve metrics.



Evaluate the complete shared test set when feasible. A convenient different test subset is not a fair substitute. If cost or runtime prevents the required evaluation, report the blocker and obtain lecturer approval for an amended protocol; do not silently claim a small subset fulfils the original requirement. Any approved subset must be fixed before results are inspected and shared across compared models, with limitations disclosed.



Compare evidence on F2 and error types, latency, marginal monetary cost, hardware/compute requirements, training data, privacy, service dependence, and reproduction limitations. Local inference has compute costs even without API charges. Discuss possible public-dataset contamination in foundation-model pretraining without asserting knowledge you do not have.



Identify deployment conditions favouring each approach and one realistic extension, such as generating an inspection note from verified observations. Do not allow generated narratives to invent root causes, defect severity, or production facts. If direct VLM application is not meaningful, use an adjacent task only with lecturer approval.



\## 8. Q4: Build the working web application



Default to \*\*Python, PyTorch/torchvision, scikit-learn, FastAPI, and a lightweight HTML/CSS/JavaScript frontend\*\*. Use a supported environment, pin actual compatible dependencies, and record versions. Detect CPU/CUDA/MPS capabilities instead of assuming hardware. Keep training code, inference code, and data processing modular; avoid unnecessary infrastructure.



The inspector must be able to:



1. Upload a supported fabric patch.

2\. Receive a new prediction from the saved trained ANN at request time.

3\. View the defect score, binary class, inspection recommendation, model version, and measured inference time.

4\. Request/view the actual Grad-CAM explanation and understand its limitations.



Load the selected checkpoint once at startup, use the exact training-compatible preprocessing, and validate artifact identity. Never substitute random weights, hard-coded scores, precomputed predictions, or sample heatmaps when inference fails.



Label an uncalibrated output as a model score; label it as a probability only with appropriate qualification and calibration evidence. Explain that the inspection recommendation follows declared thresholds and that the inspector retains responsibility.



Make the primary input contract a 256 x 256 patch. Validate image content, file type, dimensions, channel handling, and upload size. Reject unsupported inputs clearly. If adding full-resolution images, use the same verified tiling pipeline; do not squeeze an entire fabric image into one small patch or introduce unvalidated roll-level aggregation.



Provide loading/error states, readable labels, accessible contrast, and side-by-side image/explanation views. State the training acquisition domain and limitations for phone photographs, unseen fabrics, lighting changes, blur, and tiny defects. Test materially out-of-distribution inputs; do not claim reliable OOD detection unless implemented and evaluated.



Expose health, prediction, and explanation routes with documented schemas. Keep credentials server-side. Do not store uploads unnecessarily or send them to an external VLM merely to serve the ANN prediction. A simple local deployment is sufficient for the initial functional application; provide reproducible launch instructions and document any external-hosting requirement from the lecturer.



\## 9. Engineering, tests, and reproducibility



Use a small, clear repository structure, adapting an existing project when appropriate:



\`\`\`text

README.md

pyproject.toml                  # plus a lockfile or pinned environment

configs/                       # split, models, searches, XAI, VLM

data/README.md                 # source, licence, download and sample procedure

data/manifests/                # stable labels, groups, checksums, splits

src/fabric_inspector/

  data/                        # audit, preparation, grouping, transforms

  models/                      # references, classical, CNNs

  training/                    # controlled searches and logging

  evaluation/                  # metrics, final evaluation, error analysis

  explainability/              # Grad-CAM and validation

  genai/                       # VLM adapter and benchmark

  serving/                     # model loading and request handlers

app/                           # frontend and application entrypoint

tests/

scripts/

notebooks/                     # EDA/explanatory notebooks, not sole logic

experiments/                   # actual run records

artifacts/                     # checkpoints, frozen manifests, results

reports/figures/

docs/

\`\`\`



Implement meaningful checks for source/duplicate split separation, label and mask alignment, train-only fitted transformations, metric calculations, checkpoint loading, API/training preprocessing parity, genuine request-time prediction, malformed uploads, explanation dimensions/finite values, and VLM parsing/failure handling.



Run a small authentic end-to-end smoke workflow before expensive searches: prepare a representative sample, train, save/reload, predict through the API, and generate an explanation. Keep smoke outputs clearly separate from assessed full-run results. Do not count smoke success as full evaluation.



Provide commands to download/audit data, build the split, reproduce EDA, train both model families, run controlled experiments, validate XAI, benchmark the VLM, execute the frozen final evaluation, regenerate report figures, run tests, and launch the application. Explain prerequisites and reasonable compute requirements. Keep raw data, secrets, caches, and large checkpoints out of source control as appropriate; provide reliable retrieval instructions.



\## 10. Report, submission, and student understanding



Produce a report draft ordered by \*\*Questions 1-6\*\*, no more than \*\*25 pages excluding appendices\*\*. Link numerical claims, tables, and figures to run IDs and generated artifacts. Use verified citations for the dataset, architectures, XAI methods, and VLM. Never invent references, results, completed runs, or business cost estimates.



The submission must include:



- All code/notebooks, README, environment specification, and reproduction instructions.
- Complete experiment log, including unsuccessful runs and substantive failed approaches.
- Dataset provenance/licence, immutable split, and data-quality/leakage evidence. If data exceed \*\*500 MB\*\*, provide a reproducible download procedure and a representative sample that allows the code to be tested, respecting the licence.
- Initial/final model evidence, classical tuning records, final predictions, timing methodology, and error analysis.
- Working web application and explanation validation, including three incorrect-prediction case studies.
- Actual Q6 evaluation and its reproducibility/cost limitations.
- A script/storyboard for a demonstration of \*\*at most 8 minutes\*\* showing a live request-time prediction, explanation, and at least one genuine incorrect prediction. Do not claim a script is a recorded video. Record the real application only if recording tools are available; otherwise leave that human task explicit.
- A viva preparation guide covering design choices, code paths, common objections, and a small change the student can explain or perform.
- An \*\*AI-use appendix\*\* declaring assistance with brainstorming, coding, debugging, analysis, and writing separately from the assessed Q6 model use. Include what the student verified and must understand.



Explain important decisions as the project develops. The student must be able to defend every submitted item, including generated code and prose. Keep unknown results marked \`NOT RUN\` or \`PENDING\`; do not make placeholders resemble experimental evidence.



\## 11. Execution order and completion rules



Work in these stages, maintaining \`docs/progress.md\` and the requirements matrix:



1. \*\*Inspect and approve:\*\* Brief, workspace, requirement mapping, lightweight data audit, approval/protocol record.

2\. \*\*Prepare:\*\* Reproducible acquisition, verified manifests, immutable grouped split, EDA and three modelling findings.

3\. \*\*Establish baselines:\*\* References, classical baseline, initial CNN, end-to-end smoke workflow.

4\. \*\*Experiment:\*\* Comparable searches, three hyperparameters, two optimisers, real failure analysis, final candidate selection.

5\. \*\*Integrate:\*\* Validation-based calibration/thresholds, working application, XAI, and VLM prompt/model selection.

6\. \*\*Freeze and evaluate:\*\* Record the final manifest, run the single planned held-out evaluation, analyse errors and explanations without further tuning.

7\. \*\*Package:\*\* Reproduction checks, report/figures, submission sample, demonstration material, viva guide, AI-use declaration.



At each stage report completed work, evidence paths, actual commands/results, unresolved issues, and the next step. After approval, continue through authorised stages without seeking confirmation for routine choices. Do not stop after scaffolding when the required computation and tools are available.



When blocked by approval, inaccessible data, absent credentials, compute limits, paid services, or a human-only deliverable, complete independent feasible work, state the exact blocker, and provide the smallest concrete next action. Never change the approved scientific task to bypass the blocker or claim unexecuted work is finished.



The coursework is ready only when each applicable requirement has real evidence, the full pipeline can be reproduced within documented limits, the app uses the actual trained model, and remaining lecturer/student actions are explicit. High accuracy alone does not establish completion.



\*\*Begin by reading the complete brief, inspecting the workspace, and producing the requirement mapping, verified dataset audit, approval status, and executable plan. If approval is present, continue directly into implementation.\*\*