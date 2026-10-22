# Question 4 learning material: functional web application

This guide explains how the submitted project answers Question 4 of the coursework brief, why each component is required, and how to describe the implementation accurately. It is a study and viva aid; the assessed Q4 answer is on page 17 of `reports/final_report.pdf`.

## What Question 4 requires

Question 4 requires a functional web application that:

1. has a usable frontend and backend;
2. accepts a new user input;
3. obtains a prediction from the trained model rather than displaying a fixed or simulated answer;
4. communicates the result appropriately to the intended user;
5. is tested with normal, invalid, and materially different inputs; and
6. explains important limitations and appropriate use.

The application itself is the main evidence. A screenshot or interface mock-up would not demonstrate that the backend loads the trained model, processes an uploaded image, and returns a fresh prediction.

## How Questions 1–3 constrain Question 4

Q4 deploys the system already defined and evaluated; it must not silently change its task or preprocessing.

| Earlier evidence | Mandatory Q4 consequence |
|---|---|
| The model was trained on individual 256 × 256 patches. | The interface accepts one 256 × 256 patch, not an arbitrary full fabric image. It does not silently resize or tile full images. |
| The selected model is `ann-search-018`. | The backend loads the corresponding MobileNetV3-Small checkpoint, not a demonstration-only substitute. |
| Training uses grayscale conversion, channel replication, and fixed ImageNet normalisation. | Serving calls the same shared preprocessing implementation used by training. |
| The validation-selected binary threshold is `0.8952339208`. | The application uses that frozen cutoff rather than a convenient UI threshold such as 0.5. |
| Model scores are not calibrated probabilities. | The frontend calls the output an uncalibrated model score, not confidence or probability of a defect. |
| Tiny defects, unseen origins, lighting change, and blur can cause errors. | Those limitations remain visible to the inspector. A low score cannot certify defect-free fabric. |

This consistency is mandatory because a correct model evaluated under one input contract can become an invalid system if deployment changes image size, normalisation, threshold, or score meaning.

## Intended user and task

The intended user is a textile Quality Assurance Inspector. The user uploads one fabric patch and receives decision support for reviewing that patch.

The application does **not** accept a full fabric image. A full-image system would require a separately specified tiling method, overlap policy, patch-to-image aggregation rule, boundary handling, latency study, and validation on full images. None of those should be implied by the current patch-level evidence.

The system is also not an automatic release mechanism. It prioritises human inspection; it does not establish the quality of an entire roll.

## System architecture

```text
Browser frontend
  │ upload one PNG/JPEG patch
  ▼
FastAPI backend
  │ validate bytes, format, mode, size, and upload limit
  ▼
Shared grayscale preprocessing
  │ exact frozen preprocessing contract
  ▼
Selected trained ANN: ann-search-018
  │ fresh model score for this request
  ├─ frozen binary cutoff → defective / defect-free
  ├─ frozen recommendation cutoffs → routine / review / hold
  └─ optional Grad-CAM → visual attribution overlay
  ▼
Browser displays original, score, decision, recommendation,
timing, model version, explanation, and limitations
```

The separation is useful:

- `app/static/index.html`, `app/static/app.js`, and `app/static/style.css` implement the browser interface;
- `app/main.py` provides HTTP routes and request handling;
- `src/fabric_inspector/serving/predictor.py` owns the trained-model inference logic;
- `src/fabric_inspector/data/images.py` provides the shared input validation and preprocessing; and
- `artifacts/serving-bundle.json` binds the selected checkpoint, hashes, preprocessing, thresholds, provenance, and score meaning.

## Requirement 1: usable frontend and backend

### Frontend

The frontend lets the inspector:

- choose a PNG or JPEG patch;
- see the accepted dimensions and size limit before uploading;
- request a prediction with or without a visual explanation;
- inspect the original patch alongside the Grad-CAM overlay;
- vary overlay opacity without rerunning or altering the prediction;
- read the score, binary class, cutoff, and review recommendation;
- see the model version and request timings; and
- read score and deployment limitations.

The inspection button remains disabled until both a file is selected and `/health` confirms that a real model is ready. Server errors are displayed to the user, and the interface remains usable after an invalid upload.

### Backend

FastAPI exposes:

| Route | Purpose |
|---|---|
| `GET /` | Serves the frontend. |
| `GET /health` | Reports model readiness, version, provenance, input contract, and the absence of an OOD detector. |
| `POST /predict` | Validates one image and returns a fresh trained-model prediction. |
| `POST /explain` | Returns the same prediction plus original image and Grad-CAM overlay. |

Model loading happens once during application startup. CPU-heavy prediction is run outside the asynchronous event loop, and an `RLock` prevents concurrent requests from interfering with Grad-CAM hooks and gradients.

This is more than an aesthetic frontend: it is an integrated frontend/backend application with a real inference path.

## Requirement 2: accept a new user input safely

The declared input contract is:

- exactly one image;
- PNG or JPEG content;
- exactly 256 × 256 pixels;
- 8-bit grayscale (`L`) or RGB mode;
- no transparency, palette mode, or animation; and
- no larger than 5 MiB.

The server decodes and inspects the actual bytes. It does not trust the filename extension or client-supplied MIME type. This matters because a file named `patch.png` need not contain a valid PNG.

The system rejects rather than resizes a wrong-size image. Silent resizing would depart from the trained input distribution and could erase tiny defects. After validation, RGB is converted to grayscale and the exact training preprocessing contract is applied.

## Requirement 3: use the actual trained model at request time

The serving bundle identifies:

- run ID `ann-search-018`;
- checkpoint `artifacts/ann-search-018.pt`;
- checkpoint SHA-256 `8726ee0f5dfbc27783f45702c148b95fe757de891a5e41c7f250385e1a6cfe8a`;
- provenance `RAW-FABRID_V1`;
- the frozen split hash;
- the complete preprocessing specification;
- classification threshold `0.8952339208126068`; and
- recommendation cutoffs `0.8447646290063858` and `0.9988340735435486`.

At startup, the backend verifies the checkpoint hash and checks that checkpoint and bundle agree on split, provenance, and preprocessing. Missing or corrupt weights leave the application unready; prediction requests return HTTP 503. There is no random or canned fallback.

For each request, the backend:

1. decodes the uploaded bytes;
2. applies shared preprocessing;
3. runs a forward pass through the loaded ANN;
4. converts the output logit to a score;
5. applies the frozen thresholds; and
6. returns the result and model version.

The live verification confirms direct-checkpoint, HTTP, and browser score parity. Therefore the displayed output is linked to the selected checkpoint rather than to prerecorded demonstration values.

## Requirement 4: communicate a useful result

The API and frontend distinguish three different outputs.

### Uncalibrated defect score

The score ranks evidence for the defective class. It is explicitly labelled `uncalibrated_model_score`; for example, a score of 0.79 must not be stated as a 79% probability that the patch is defective.

### Binary prediction

The class is `defective` when the score is at least the validation-selected cutoff `0.8952339208`; otherwise it is `defect_free`. The cutoff is shown so the decision is not presented as unexplained magic.

### Inspection recommendation

The validation-frozen recommendation bands are:

| Score band | Recommendation |
|---|---|
| Below `0.8447646290` | Routine inspection |
| `0.8447646290` to below `0.9988340735` | Manual review |
| At or above `0.9988340735` | Priority hold |

These are coursework decision rules based on assumed costs, not uncertainty intervals or production-certified actions. The inspector retains responsibility. Keeping the binary class and operational recommendation separate prevents the model class from being mistaken for an automatic factory decision.

## Optional explanation support

Grad-CAM is not needed merely to prove that a prediction web app functions, but it strengthens usability and connects Q4 to the mandatory explainability work in Q5.

When requested, `/explain` targets the defect logit and returns:

- the original decoded patch;
- an overlay of positive Grad-CAM attribution;
- the explanation computation time;
- whether the attribution map is entirely zero; and
- a warning that the map is neither a certified defect boundary nor causal proof.

The original image stays visible, and opacity can be changed in the browser. Changing opacity changes only presentation; it does not change the score or rerun model selection.

## Requirement 5: test normal, invalid, and different inputs

`reports/application-verification/results.json` records 13 live HTTP checks and six browser checks against model version `8726ee0f5dfb`.

### Normal valid inputs

The application was tested with:

- an original grayscale PNG;
- an equivalent RGB PNG; and
- a grayscale JPEG.

For all three, the HTTP result matches a direct call to the same checkpoint. This tests both supported modes/formats and verifies serving parity.

### Invalid inputs

| Invalid case | Expected behaviour |
|---|---|
| Empty bytes | HTTP 422 with a useful validation message |
| Malformed bytes | HTTP 422 |
| 128 × 256 wrong-size image | HTTP 422; no silent resize |
| RGBA image | HTTP 422 |
| Paletted image | HTTP 422 |
| Upload larger than 5 MiB | HTTP 413 |
| Explanation opacity outside 0–1 | HTTP 422 |

These tests are mandatory because a functioning happy path alone does not demonstrate safe handling of realistic user mistakes.

### Materially different inputs

A uniform blank image and seeded random noise both return finite numerical scores without crashing. The correct conclusion is only that the application remains technically stable. Their predicted classes are not meaningful evidence of defect-detection performance.

The API and UI explicitly state that out-of-distribution detection is not implemented or claimed. Returning a finite softmax/sigmoid score does not prove that an unfamiliar input is safe or in distribution.

### Browser interaction

The browser verification checks that:

- the real model is ready and no development banner is shown;
- three authentic cases obtain fresh UI prediction and explanation parity;
- invalid input produces a visible error while the interface remains usable; and
- no browser JavaScript errors occur.

This is stronger than endpoint-only testing because it exercises the user-visible workflow.

## Requirement 6: limitations and appropriate use

The following limitations are visible or returned with the result:

1. The application accepts a patch, not a full fabric image or roll.
2. A low score does not certify defect-free fabric.
3. Scores are uncalibrated and are not guaranteed confidence values.
4. Tiny defects can be missed; the held-out evaluation missed 28 of 96 defects.
5. Unseen weaves, phone photographs, changed lighting, and blur may cause errors.
6. No automatic out-of-distribution detector is implemented.
7. Grad-CAM is a coarse attribution cue, not segmentation or a manufacturing-cause diagnosis.
8. Recommendation thresholds use assumed coursework costs and are not factory safety rules.
9. Uploaded pixels are processed locally and are not sent to the VLM, but the prototype does not establish a complete production security/privacy programme.
10. Human inspection remains necessary.

Limitations are mandatory because the target user could otherwise interpret the output more strongly than the Q3 evaluation supports.

## Demonstration evidence

`reports/demonstration.mp4` is an approximately 130-second recording of the live browser application. It includes:

- a fresh prediction for an authentic normal patch;
- a fresh prediction for an authentic defective patch;
- the original image and adjustable Grad-CAM presentation;
- the score, cutoff, recommendation, and model version;
- a genuine false negative retained rather than hidden;
- invalid-input rejection; and
- limitations, human review, and dataset attribution.

The three cases and expected immutable scores are recorded in `data/demo/cases.json`. Ground-truth labels are used to explain the demonstration; they are not supplied to the model.

The recording is application evidence, not a substitute for the student's individual live presentation or understanding of the implementation.

## Technical-accuracy audit

The Q4 implementation is technically coherent and satisfies the functional application requirements, subject to the following precise qualifications:

1. It is a patch-level application only; full-image tiling is not implemented or validated.
2. The deployed artifact hash matches the serving bundle, and startup checks bind it to the selected run and preprocessing.
3. Fresh direct, HTTP, and browser outputs were checked for parity; no fake fallback is present.
4. Content-based validation covers the documented formats, dimensions, modes, size, and explanation opacity.
5. Blank/noise stability is not OOD-detection accuracy. The project correctly makes no OOD claim.
6. Score and recommendation meanings are communicated accurately, but they remain uncalibrated and dependent on validation-derived cutoffs.
7. Model inference timing excludes wider network/process startup and does not establish production throughput under concurrent load.
8. The in-process lock protects Grad-CAM/model operations within one worker, but this prototype is not a complete distributed concurrency or load test.
9. Local processing avoids sending uploaded pixels to an external VLM; broader authentication, audit logging, retention policy, TLS, and access control would still be needed in production.
10. The verification result is evidence of the tested environment, not a guarantee across every browser, device, or unseen fabric source.

## Evidence map

| Q4 requirement | Primary repository evidence |
|---|---|
| Frontend | `app/static/index.html`, `app/static/app.js`, `app/static/style.css` |
| Backend and HTTP routes | `app/main.py` |
| Real request-time inference | `src/fabric_inspector/serving/predictor.py` |
| Shared input validation/preprocessing | `src/fabric_inspector/data/images.py` |
| Selected artifact and frozen contract | `artifacts/serving-bundle.json`, `artifacts/ann-search-018.pt` |
| HTTP and browser tests | `reports/application-verification/results.json` |
| Reproduction/verification workflow | `tools/verify_live_application.py`, `README.md` |
| Authentic demonstration cases | `data/demo/cases.json` |
| Recorded application demonstration | `reports/demonstration.mp4` |
| Assessed answer | `reports/final_report.pdf`, page 17 |

## How to run it

After installing the environment from the repository root:

```bash
FABRIC_BUNDLE=artifacts/serving-bundle.json \
FABRIC_CPU_THREADS=1 \
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and upload an authentic 256 × 256 PNG or JPEG patch. To reproduce the retained live verification:

```bash
PYTHONPATH=src python tools/verify_live_application.py \
  --cases data/demo/cases.json
```

Browser verification additionally requires the documented Node.js/Playwright setup.

## Concise viva explanation

Question 4 requires a functional application around the trained model, not a mock interface. My browser frontend uploads one 256 × 256 PNG or JPEG patch to FastAPI. The server validates the actual bytes, applies the same preprocessing used during training, runs the selected `ann-search-018` checkpoint, and applies validation-frozen thresholds. The interface shows an uncalibrated score, binary prediction, inspection recommendation, model version, original image, and optional Grad-CAM overlay while keeping limitations visible. I tested valid grayscale, RGB, and JPEG inputs; malformed, wrong-size, unsupported, oversized, and invalid-control inputs; blank and noise inputs; and the complete browser workflow. The system performs fresh local inference and has no canned fallback, but it remains a patch-level decision-support prototype without validated OOD detection or production-safety certification.
