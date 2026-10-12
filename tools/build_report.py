"""Author a page-bounded Q1-Q6 report using only completed real evidence."""

from pathlib import Path
import json
import re
import xml.sax.saxutils as xml

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    Flowable,
)


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


class ArchitectureDiagram(Flowable):
    def __init__(self, labels):
        Flowable.__init__(self)
        self.labels = labels
        self.width = 475
        self.height = 30 * len(labels)

    def draw(self):
        c = self.canv
        for i, label in enumerate(self.labels):
            y = self.height - 30 * (i + 1)
            c.setFillColor(colors.HexColor("#eef5f7"))
            c.setStrokeColor(colors.HexColor("#1d6773"))
            c.roundRect(60, y, 355, 22, 5, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#173a4b"))
            c.setFont("Body", 8.4)
            c.drawCentredString(237, y + 7, label)
            if i + 1 < len(self.labels):
                c.line(237, y, 237, y - 8)
                c.line(237, y - 8, 234, y - 5)
                c.line(237, y - 8, 240, y - 5)


def main():
    comparison = read("reports/final-evaluation/comparison.json")
    assert comparison["status"] == "COMPLETE"
    guard = read("reports/final-evaluation/test-access.json")
    assert guard["status"] == "COMPLETE"
    frozen = read("artifacts/final-evaluation-manifest.json")
    bundle = read("artifacts/serving-bundle.json")
    final = read("experiments/" + bundle["run_id"] + ".json")
    initial = read("experiments/ann-search-000.json")
    classical = frozen["comparisons"]["classical"]
    seed = read("artifacts/seed-variability.json")
    errors = read("docs/three-xai-errors.json")["cases"]
    error_analysis = read("reports/error-analysis.json")
    failures = read("docs/failed-approaches.json")["attempts"]
    eda = read("docs/eda-statistics.json")
    app = read("reports/application-verification/results.json")
    assert app["status"] == "PASS"
    notebooks = read("reports/notebook-verification.json")
    assert notebooks["status"] == "PASS"
    checks = read("reports/execution-checks.json")
    vlm = read("reports/final-evaluation/vlm.json")
    xai = read("reports/final-evaluation/xai/xai-validation.json")
    by_name = {r["model"]: r for r in comparison["comparisons"]}
    winner = max(
        (by_name[n] for n in ["classical", "final_ann"]), key=lambda r: r["metrics"]["f2"]
    )["model"]
    ann_runs = [
        read(p) for p in sorted(Path("experiments").glob("ann-search-[0-9][0-9][0-9].json"))
    ]
    svm_runs = [
        read(p) for p in sorted(Path("experiments").glob("svm-search-[0-9][0-9][0-9].json"))
    ]
    assert all(r["status"] == "COMPLETE" for r in ann_runs + svm_runs)
    pages = []

    def page(title, subtitle=""):
        result = {"title": title, "subtitle": subtitle, "items": []}
        pages.append(result)
        return result["items"]

    def p(items, text):
        items.append(("p", text))

    def table(items, head, rows, widths=None):
        items.append(("table", (head, rows, widths)))

    def figure(items, path, width=6.55):
        items.append(("image", (path, width)))

    items = page(
        "Risk-aware, explainable fabric defect detection",
        "Individual ANN coursework | RAW-FABRID V1 | Questions 1-6",
    )
    p(
        items,
        "A textile Quality Assurance Inspector submits a 256 x 256 fabric patch, reviews the trained model score and binary decision, then uses a Grad-CAM overlay to direct visual inspection. The inspector retains the release/hold decision. This system does not certify a complete roll or operate production machinery.",
    )
    p(
        items,
        "The approved dataset was recovered from the publisher, audited and evaluated using one frozen supervised split. The publisher training folder contains only normal patches; labelled patches were pooled and reassigned by the publisher fabric-origin field, source image and duplicate relationships before training. This is a new supervised protocol, not a benchmark result on the publisher anomaly-detection split.",
    )
    table(
        items,
        ["Evidence", "Completed result"],
        [
            ["Full release", "19,852 patches; 687 defective; publisher SHA-256 verified"],
            [
                "Training search",
                f"{len(ann_runs)} ANN configurations; {len(svm_runs)} classical configurations including four pilots; three selected-model seeds",
            ],
            [
                "Final population",
                "All 2,996 held-out patches; 96 defective; same examples for ANN/SVM/VLM",
            ],
            [
                "Scientific failures",
                "Four actual unsuccessful attempts, including two substantive approaches",
            ],
            ["XAI", "Three genuine incorrect predictions; measured top/low/random occlusion"],
            [
                "Engineering checks",
                f"{checks['pytest_passed']} tests passed; real HTTP/browser checks and four notebooks passed",
            ],
            [
                "Presentation evidence",
                "Actual application recording under eight minutes; individual live defence remains the student task",
            ],
        ],
    )
    p(
        items,
        f"Final ANN: {bundle['run_id']}, {final['architecture']['total_parameters']:,} parameters, test F2 {by_name['final_ann']['metrics']['f2']:.4f}. Tuned classical: {classical['run_id']}, test F2 {by_name['classical']['metrics']['f2']:.4f}. The observed winner between these families is {winner.replace('_', ' ')}. No choice was changed after observing test scores.",
    )
    p(
        items,
        "The purpose is reproducible evidence and defensible reasoning. Predictive weakness, limited origin coverage, uncalibrated scores and adverse XAI findings are reported directly. This is a research prototype, not an operational quality guarantee. The frozen evaluation was physically computed twice after files were lost. This is a disclosed exception to the brief's once-only rule; lecturer acceptance is unverified.",
    )

    items = page("Q1 | Dataset, provenance and target", "Source and audited release structure")
    p(
        items,
        "RAW-FABRID V1 is published by Ruben Perez-Llorens, Teresa Albero-Albero and Javier Silvestre-Blanes, 1 April 2026, DOI 10.17632/db6g85xsyg.1, under CC BY 4.0 [1]. The publisher describes industrial line-scan grayscale imagery. The byte audit, rather than a web description alone, defines this implementation.",
    )
    table(
        items,
        ["Release object", "Audited count / shape"],
        [
            ["Original images", "709; 1792 x 1024; 8-bit grayscale PNG"],
            ["Original masks", "204"],
            ["All image patches", "19,852; 256 x 256 grayscale"],
            ["Publisher train/good", "14,196; normal only"],
            ["Publisher test/good", "4,969; normal"],
            ["Publisher test/defect and patch masks", "687 images; 687 corresponding masks"],
        ],
    )
    p(
        items,
        "The target is patch label 1 if its corresponding annotation contains at least one exact-white (255) defect pixel; label 0 requires explicit normal status from the publisher metadata/folder. A normal patch remains normal when its parent source contains a defect elsewhere. A missing mask is never silently converted into a normal label. Masks, source IDs and revealing paths do not enter the classifier.",
    )
    p(
        items,
        "The patches CSV explicitly joins patch_name to parent_raw. The high-resolution CSV joins filename to roll_id. The publisher README describes this field as anonymised fabric origin: it is conservatively kept intact, but a one-to-one mapping to physical rolls, fabric types or manufacturers is not established.",
    )
    p(
        items,
        "471 patch masks contain intermediate grayscale edges, contrary to a strict two-level interpretation of the publisher description. Exactly-white foreground reproduces every one of the 687 CSV defect areas. Raw mask bytes are preserved. Source/mask alignment and labels are checked before the manifest is frozen.",
    )
    p(
        items,
        "The complete archive is 4,133,046,386 bytes. Its verified SHA-256 is f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9. The submission provides the reproducible range-download script and an authentic 168-patch sample because the archive exceeds 500 MB. Sample defect oversampling is for engineering tests, never assessment metrics.",
    )

    items = page("Q1 | Train-only exploratory analysis", "Image statistics and annotation scale")
    figure(items, "reports/figures/eda-summary.png")
    p(
        items,
        f"Training contains {eda['patches']:,} patches and {eda['positives']} positives ({eda['positive_fraction']:.2%}), from {eda['source_images']} sources but only four publisher-origin groups. {eda['source_images_containing_positive_patches']} sources contain positive patches. Related patches are not independent samples.",
    )
    q = eda["defect_area_fraction_quantiles"]
    p(
        items,
        f"Among positive training patches, the median foreground area is {q['median']:.2%}; quartiles are {q['q25']:.2%} and {q['q75']:.2%}. The smallest annotation is one pixel ({q['min']:.4%}); the largest is {q['max']:.2%}. This label definition includes defects too small for reliable human or coarse CNN recognition.",
    )
    figure(items, "reports/figures/compact-training-examples.png", 6.55)
    p(
        items,
        f"Patch mean intensity spans {eda['brightness_range'][0]:.1f}-{eda['brightness_range'][1]:.1f}; standard-deviation contrast spans {eda['contrast_range'][0]:.1f}-{eda['contrast_range'][1]:.1f}. These are acquisition observations, not verified defect categories. All model-driving EDA uses train only.",
    )

    items = page("Q1 | Leakage controls, consequences and metric", "One frozen supervised protocol")
    table(
        items,
        ["Partition / publisher origins", "Patches", "Positive", "Sources"],
        [
            ["Train: Rollo2A, 4A, 5A, 7A", "13,384", "539", "478"],
            ["Validation: Rollo6A", "3,472", "52", "124"],
            ["Test: Rollo1A, 3A", "2,996", "96", "107"],
        ],
        [240, 65, 65, 65],
    )
    p(
        items,
        "The split uses seed 42; 128 candidate draws are assessed for structural integrity and class availability, with draw 32 selected without model performance. Exact decoded-pixel hashes find zero duplicates. The near-duplicate audit examines 324,109 candidate pairs, performs 7,587 full-pixel checks and finds zero confirmed matches within MAE 3/255. Source/origin/duplicate components are disjoint. Rotations, alternate crops and undocumented relationships can still evade these checks.",
    )
    p(
        items,
        "False negatives can miss a defective patch and waste downstream material; false positives create unnecessary inspection or holds. Those are relative consequences, not measured factory costs. The primary metric is positive-class F2 = 5 TP / (5 TP + FP + 4 FN), emphasizing recall while retaining a precision penalty. Accuracy alone is misleading: the always-normal benchmark benefits from class imbalance. Report precision, recall, F1, false-positive rate, confusion counts, AP and ROC-AUC where continuous scores exist.",
    )
    p(
        items,
        "Each score-producing model selects its cutoff on the same full validation partition: 101 fixed score quantiles plus extremes, maximizing F2; ties prefer precision, then higher cutoff. VLM unresolved decisions (-1) remain in denominators and count as errors for their true class; they are never treated as probabilities. Ranking metrics are omitted for label-only VLM output.",
    )
    p(
        items,
        "Three modelling constraints conclude the EDA: (1) low prevalence requires recall-sensitive evaluation and a train-derived weighted-loss candidate; (2) tiny defect regions require preserving 256 x 256 detail, avoiding crops/blur, and checking spatial explanations; (3) only four training origins require modest capacity, transfer-learning comparison and conservative group-level evaluation. Train/validation prevalence also differs, so raw loss gaps alone cannot prove overfitting.",
    )
    p(
        items,
        "Split SHA-256: af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f. Choices, checkpoints, prompts and code were frozen before any test access. A runtime reset later lost the first completed test files; they were reconstructed by a second physical computation of the identical protocol, without training, reselection or changed cutoffs. This is one benchmark population, not independent repeated validation. The original frozen plan retains its intended count of one; the recovery record discloses the actual two computations. Two test origins limit uncertainty estimates.",
    )

    def architecture_page(title, run):
        items = page(title, run["run_id"])
        cfg = run["config"]
        spec = cfg["model"]
        leaves = run["architecture"]["layers"]
        p(
            items,
            f"Measured total/trainable parameters: {run['architecture']['total_parameters']:,}. The hook-derived leaf-layer table is retained as reports/{run['run_id']}-architecture.csv. The summary below groups those actual modules; batch dimension is omitted. Output is one unbounded defect logit, converted to a sigmoid score for display and validation thresholding.",
        )
        if spec["name"] == "custom":
            labels = ["Input: 1 x 256 x 256"]
            rows = []
            for i in range(spec["depth"]):
                layers = [r for r in leaves if r["layer"].startswith(f"features.{i}.")]
                out = layers[-1]["output"][1:]
                width = spec["width"] * 2**i
                rows.append(
                    [
                        f"Block {i + 1}",
                        "3x3 Conv, BN, ReLU, 2x2 max pool",
                        " x ".join(map(str, out)),
                        str(sum(layer["parameters"] for layer in layers)),
                    ]
                )
                labels.append(
                    f"Block {i + 1}: Conv3x3 / BN / ReLU / Pool -> " + " x ".join(map(str, out))
                )
            labels.append(f"Global average pool / dropout {spec['dropout']} / linear -> 1 logit")
            rows.extend(
                [
                    ["Global average pool", "AdaptiveAvgPool2d(1)", f"{width} x 1 x 1", "0"],
                    [
                        "Classifier",
                        f"Flatten, dropout {spec['dropout']}, Linear({width},1)",
                        "1",
                        str(width + 1),
                    ],
                ]
            )
        else:
            labels = [
                "Input: replicated grayscale, 3 x 256 x 256",
                "Stem: Conv / BN / Hardswish",
                "11 MobileNetV3 inverted residual blocks, including squeeze-excitation",
                "Final 1x1 convolution / BN / Hardswish",
                "Global average pool / 576 -> 1024 / Hardswish / dropout / 1 logit",
            ]
            rows = []
            stages = {}
            for leaf in leaves:
                match = re.match(r"network.features.(\d+)\.", leaf["layer"])
                if match:
                    stages.setdefault(int(match.group(1)), []).append(leaf)
            for stage, ll in sorted(stages.items()):
                activation = "HSw" if any(x["type"] == "Hardswish" for x in ll) else "ReLU"
                se = any(x["type"] == "Hardsigmoid" for x in ll)
                operation = (
                    "Conv3x3 / BN / HSw"
                    if stage == 0
                    else "Conv1x1 / BN / HSw"
                    if stage == max(stages)
                    else "IR / BN / " + activation + (" / SE" if se else "")
                )
                rows.append(
                    [
                        f"Features {stage}",
                        operation,
                        " x ".join(map(str, ll[-1]["output"][1:])),
                        str(sum(layer["parameters"] for layer in ll)),
                    ]
                )
            classifier = [
                leaf for leaf in leaves if not re.match(r"network.features.", leaf["layer"])
            ]
            rows.append(
                [
                    "Pool/classifier",
                    "GAP, Linear 576->1024, Hardswish, dropout 0.2, Linear 1024->1",
                    "1",
                    str(sum(leaf["parameters"] for leaf in classifier)),
                ]
            )
        items.append(("diagram", labels))
        table(items, ["Stage", "Operation", "Output shape", "Parameters"], rows, [70, 190, 110, 65])
        return items

    items = architecture_page("Q2.a | Initial ANN architecture", initial)
    p(
        items,
        "Three local convolutional blocks assume shared local weave cues and translation tolerance. Width 8 limits capacity with few independent origins; progressive width 8/16/32 represents increasingly complex texture combinations. Same padding preserves edge information before pooling. Batch normalisation stabilises activation scale; ReLU introduces nonlinearity. Pooling and global averaging reduce computation and parameters, but can dilute very small defects. Dropout 0.2 regularises the compact classifier.",
    )
    p(
        items,
        "BCEWithLogitsLoss is stable for a single binary logit. The initial positive weight is (13,384-539)/539 = 23.8312, derived from train only. It addresses imbalance but need not calibrate scores or equal the business cost ratio. AdamW, lr 0.001, batch 32, four screening epochs, horizontal/vertical train-only flips. Random seeds are fixed; unsupported deterministic kernels produce warnings rather than an invented reproducibility guarantee.",
    )

    items = architecture_page("Q2.a | Selected final ANN architecture", final)
    cfg = final["config"]
    p(
        items,
        f"Selection used validation F2, then AP, then fewer parameters across all 21 candidates. Final configuration: {cfg['model']}; {cfg['optimizer']}, lr {cfg['learning_rate']}, batch {cfg['batch_size']}, augmentation {cfg['augment']}, weighted loss {cfg['weighted_loss']}; selected epoch {final['selected_epoch']} of {final['epochs_completed']}. The 12-epoch refinement is a predeclared candidate, not automatic promotion.",
    )
    items = page(
        "Q2.a | Final architecture design choices",
        "Component justification and limitations",
    )
    p(
        items,
        "Depth and width are tested as receptive-field hierarchy and capacity choices. Custom input uses (gray/255-0.5)/0.5. MobileNet replicates gray into three channels and uses ImageNet means/stds; every layer is fine-tuned from genuine ImageNet weights [2,3], with its classifier replaced by one output. No resize or centre crop is used. Transfer adds domain mismatch and coarse features despite its data-scarcity motivation.",
    )
    p(
        items,
        "IR means inverted residual: expansion where configured, depthwise convolution and projection; residual addition where shapes permit. BN is BatchNorm; HSw is Hardswish. SE uses ReLU/Hardsigmoid channel gates. GAP and dropout 0.2 limit head capacity. BN statistics use train only; inference/Grad-CAM use eval mode. Leaf-layer CSVs give full detail.",
    )
    table(
        items,
        ["Component", "Reason for retaining it", "Relevant limitation"],
        [
            [
                "Replicated grayscale / ImageNet normalisation",
                "Matches the genuine pretrained three-channel input without changing patch geometry.",
                "Colour pretraining differs from industrial grayscale; replication adds no colour information.",
            ],
            [
                "Depthwise convolution and pointwise projection",
                "Separates spatial filtering from channel mixing with fewer parameters than a similarly wide dense convolution.",
                "Progressive stride reduces tiny-defect detail; the final feature map is coarse.",
            ],
            [
                "Inverted residual / matching-shape shortcut",
                "Expanded features support local combinations; shortcuts preserve information and gradient flow when dimensions match.",
                "Widths/strides are inherited from MobileNetV3, not separately optimised for this dataset.",
            ],
            [
                "BatchNorm",
                "Reuses trainable scale/offset and train-estimated running statistics; eval mode keeps serving consistent.",
                "Origin or illumination shift can invalidate the training statistics.",
            ],
            [
                "ReLU / Hardswish / SE gates",
                "ReLU supplies nonlinearity; Hardswish is x*ReLU6(x+3)/6. SE pools channel context and uses Hardsigmoid gates, as in the transferred architecture.",
                "Channel/global context can favour weave background; positive CAM attribution does not prove defect causality.",
            ],
            [
                "Global average pool",
                "Converts spatial features into a fixed channel vector without a large flattened classifier.",
                "Small defect responses can be diluted by averaging.",
            ],
            [
                "576 -> 1024 -> 1 head; dropout 0.2",
                "Retains the pretrained nonlinear head structure and replaces the ImageNet output with one binary logit. Dropout regularises training only.",
                "The 1024-unit head width is inherited, not a demonstrated optimum; all layers are fine-tuned.",
            ],
        ],
        [100, 185, 180],
    )
    p(
        items,
        "BCEWithLogitsLoss and the train-derived positive weight retain the initial binary objective. The final learning rate, augmentation and epoch are validation-selected settings. A sigmoid score plus frozen cutoff produces the class; it is not calibrated confidence. The complete leaf table and recorded configuration make these design choices inspectable.",
    )

    items = page(
        "Q2.b | Alternatives and pre-evaluation expectation",
        "Suitability depends on the task and available evidence",
    )
    table(
        items,
        ["Alternative", "When appropriate", "Why retain or reject here"],
        [
            [
                "LBP/HOG with SVM",
                "Limited labelled data; texture-dominated patches",
                "Retained and tuned; spatial/directional features may generalise better than learned filters.",
            ],
            [
                "Small scratch CNN",
                "Task-specific local cues, enough independent coverage",
                "Retained as initial model; explicit depth/width help defend design. Only four origins limit learning.",
            ],
            [
                "Pretrained compact CNN",
                "Limited labels and reusable visual features",
                "Genuine fine-tuning tested; domain mismatch, grayscale and tiny defects can weaken transfer.",
            ],
            [
                "Vision transformer",
                "Large diverse data or suitable domain pretraining",
                "Rejected as a primary candidate: compute/data requirements and local-detail tradeoffs exceed this study.",
            ],
            [
                "Normal-only anomaly detection",
                "Representative normal process, rare unknown defects",
                "Publisher split supports it, but it would change the approved supervised comparison and score semantics.",
            ],
            [
                "Supervised segmentation",
                "Reliable dense labels and need for exact defect boundaries",
                "Potential next study; binary patch classification is the approved task. Grad-CAM is not segmentation.",
            ],
            [
                "RNN/LSTM/GRU",
                "True temporal sensor/frame sequences",
                "Rejected: patch pixels are spatial, not a time-ordered production sequence.",
            ],
        ],
        [90, 140, 205],
    )
    p(
        items,
        "An ANN is technically appropriate because unstructured images contain spatial, compositional texture cues. Its value still must be demonstrated against a strong descriptor baseline; modality alone does not guarantee superior F2.",
    )
    p(
        items,
        "Before final evaluation, docs/pre-evaluation-hypothesis.md recorded a conditional expectation that fine-tuned compact CNNs may exceed the SVM on F2 if independent-source coverage is sufficient. Adaptable local filters and transfer were the supporting reasons. Tiny defect dilution, regular weave, colour-domain mismatch and sparse origins were explicit reasons the classical model could win. The final result does not rewrite that prior.",
    )

    items = page(
        "Q3.a | Study design and complete experiment record",
        "Selection uses train and validation only",
    )
    table(
        items,
        ["Family", "Measured budget / search space"],
        [
            [
                "ANN screening",
                "20 configurations, up to four epochs each, early-stop patience 3, seed 42",
            ],
            [
                "ANN refinement",
                "Best screened configuration retrained for up to 12 epochs; remains one candidate",
            ],
            [
                "Seed repetitions",
                "Selected configuration, seeds 42 / 123 / 2026; no seed chosen by test score",
            ],
            [
                "Main classical search",
                "24 configurations: 2 feature sets x 2 kernels x 3 C values x 2 class weights",
            ],
            [
                "Classical pilots",
                "Four earlier genuine libsvm linear fits retained in the full log",
            ],
            [
                "Foundation model",
                "Two exact prompts on every validation example; one selected prompt on every test example",
            ],
        ],
    )
    p(
        items,
        "ANN factors: AdamW learning rates 0.0001/0.0003/0.001/0.003; batch 16/32/64; convolutional depth 2/3/4; momentum SGD rates 0.001/0.01/0.03/0.1; weighted/unweighted BCE; flips on/off; width 8/16; dropout 0/0.2; four genuine MobileNet transfer configurations. These are defined ablations and candidates, not a full combinatorial grid.",
    )
    p(
        items,
        "Classical features are 160-dimensional spatial 4x4 LBP histograms or 8,260-dimensional HOG+LBP. StandardScaler is fitted on train only. Main linear fits use LinearSVC, squared-hinge loss, tol 1e-4, originally dual=auto and max_iter 10,000. A training-only numerical amendment tries dual=True for 2,000 iterations on remaining HOG+LBP fits with C >=1; any convergence warning triggers a fresh fit with the full original solver. Validation scores never determine numerical fallback. RBF SVC uses gamma=scale and a 1,024 MiB kernel cache. C is 0.1/1/10; class_weight is None/balanced. LinearSVC is not mathematically identical to earlier hinge-loss pilots; final parameters, warnings, iterations and each numerical attempt are retained.",
    )
    p(
        items,
        f"Approximate measured training time: ANN screening/refinement {sum(r['training_seconds'] for r in ann_runs) / 3600:.2f} CPU elapsed hours; classical fits {sum(r['training_seconds'] for r in svm_runs) / 3600:.2f} elapsed hours. These are same-order configuration budgets, not equal FLOPs. Classical feature extraction is charged on first use and later reused. Runs share hardware during training, so elapsed-time comparisons have contention limitations.",
    )
    p(
        items,
        "The full reports/experiment-log.csv and JSON entries retain every configuration, change, hypothesis, validation result, time, code/data/checkpoint hash and learned lesson. The workspace reset, interrupted engineering probes and earlier synthetic development checks are separately labelled, never counted as scientific failures. The recovered partial VLM prefix is reused only after exact source/image/revision/parser checks, then scored with the complete population.",
    )

    items = page(
        "Q3.a | Learning curves and model development", "Initial and selected final neural model"
    )
    figure(items, "reports/figures/initial-final-learning.png")
    for run, label in [(initial, "Initial"), (final, "Final")]:
        h = run["history"]
        p(
            items,
            f"{label} {run['run_id']}: train BCE {h[0]['train_loss']:.4f} to {h[-1]['train_loss']:.4f}; validation BCE {h[0]['val_loss']:.4f} to {h[-1]['val_loss']:.4f}; selected epoch {run['selected_epoch']}/{run['epochs_completed']}, validation F2 {run['validation_metrics']['f2']:.4f}, AP {run['validation_metrics']['average_precision']:.4f}. The checkpoint is the best recorded validation epoch, not necessarily the last one.",
        )
    p(
        items,
        "The initial weighted loss changes little over its four epochs and yields many false positives; this supports limited early discrimination rather than a solved defect detector. Flat/improving training loss combined with weak validation ranking is consistent with inadequate convergence, source shift or an architecture that dilutes tiny regions. Those mechanisms are not isolated by one curve. A train/validation loss gap also reflects different prevalence under weighted BCE. A later worsening validation score after training improvement warns against choosing the last epoch.",
    )
    p(
        items,
        "The final model is selected from measured candidates. Improvement or deterioration is judged by validation F2/AP and the held-out comparison, not by decreasing training loss alone. Architecture, optimiser and objective changes require their matched controls; an aggregate initial/final change is not a single-cause experiment.",
    )

    items = page(
        "Q3.b | Three empirical hyperparameter studies",
        "Learning rate, batch size and actual network depth",
    )
    figure(items, "reports/figures/hyperparameter-study.png")
    opt = read("reports/optimisation-analysis.json")
    for c in opt["comparisons"][:3]:
        best = max(zip(c["run_ids"], c["f2"]), key=lambda x: x[1])
        p(
            items,
            f"{c['factor']}: "
            + "; ".join(f"{rid} F2={score:.4f}" for rid, score in zip(c["run_ids"], c["f2"]))
            + f". The strongest observed setting in this matched screen is {best[0]}; this is a validation observation within four epochs.",
        )
    p(
        items,
        "Learning rate changes update size and convergence/stability; a small rate may need longer training and a large rate can oscillate. Batch size changes aggregation and updates per epoch: 16 gives about twice as many updates as 32, so fixed epochs do not isolate gradient noise. Actual depth changes convolutional block count and receptive-field hierarchy, parameter count and CAM resolution; additional pooling can discard tiny signals. These are bounded empirical comparisons, not universal parameter recommendations.",
    )

    items = page(
        "Q3.b | Optimisers and seed variability", "Mechanics interpreted through this study"
    )
    rows = []
    for idx in [0, 1, 2, 3, 8, 9, 10, 11]:
        r = read(f"experiments/ann-search-{idx:03d}.json")
        rows.append(
            [
                r["run_id"],
                r["config"]["optimizer"],
                str(r["config"]["learning_rate"]),
                fmt(r["validation_metrics"]["f2"]),
                fmt(r["validation_metrics"]["average_precision"]),
                str(r["selected_epoch"]),
            ]
        )
    table(
        items,
        ["Run", "Optimiser", "LR", "Val F2", "Val AP", "Best epoch"],
        rows,
        [115, 70, 65, 65, 65, 65],
    )
    adam = max((r for r in ann_runs[:4]), key=lambda r: r["validation_metrics"]["f2"])
    sgd = max(
        (read(f"experiments/ann-search-{i:03d}.json") for i in [8, 9, 10, 11]),
        key=lambda r: r["validation_metrics"]["f2"],
    )
    p(
        items,
        f"Best matched AdamW screen: {adam['run_id']} F2 {adam['validation_metrics']['f2']:.4f}; best momentum SGD screen: {sgd['run_id']} F2 {sgd['validation_metrics']['f2']:.4f}. Each has its own learning-rate range. The comparison holds the scratch architecture, loss, augmentation, batch size, seed and maximum epochs constant, while optimiser/LR vary together as stated.",
    )
    p(
        items,
        "AdamW uses adaptive first/second moments (betas 0.9/0.999, epsilon 1e-8) and decoupled weight decay; SGD uses momentum 0.9 without Nesterov. Both use weight decay 1e-4. SGD may require larger rates and longer training. No scheduler is used: this establishes early-budget behaviour, not superiority at eventual convergence. Training is shuffled with a seeded generator, zero loader workers and no dropped final batch. Full defaults and checkpoint rules are in docs/training-method.json.",
    )
    table(
        items,
        ["Selected-model seed", "Validation F2"],
        [[str(s), fmt(v)] for s, v in zip(seed["seeds"], seed["f2"])],
    )
    p(
        items,
        f"Validation mean {seed['mean']:.4f}, sample SD {seed['sample_std']:.4f}, range {seed['range']:.4f}; material-variation rule: range >0.02, observed {seed['material']}. All three configurations were frozen before test access; artifact reconstruction changes no seed choice. Held-out mean F2 {comparison['test_seed_repetitions']['mean']:.4f}, sample SD {comparison['test_seed_repetitions']['sample_std']:.4f}. The serving seed remains 42 regardless of test rankings.",
    )

    items = page(
        "Q3.b | Two substantive approaches that failed",
        "Actual completed runs with bounded diagnoses",
    )
    for failure in [f for f in failures if f["substantive"]]:
        e = failure["evidence"]
        p(
            items,
            f"<b>{failure['run_id']} versus {failure['control_run_id']}</b>. Hypothesis: {failure['hypothesis']}",
        )
        p(
            items,
            f"Validation F2 {e['attempt_f2']:.4f} versus {e['control_f2']:.4f}; AP {e['attempt_average_precision']:.4f} versus {e['control_average_precision']:.4f}; TP/FP/FN {e['attempt_tp_fp_fn']} versus {e['control_tp_fp_fn']}.",
        )
        p(items, "Control scope: " + failure["controlled_change"])
        p(items, failure["diagnosis"])
        p(items, "Lesson: " + failure["lesson"])
    p(
        items,
        "These are substantive descriptor/whole-approach decisions, not deliberate sabotage or syntax failures. The whole-model ANN/SVM comparison supports rejecting that short-budget approach relative to a useful comparator; it does not isolate one causal architecture property. All other tested candidates remain in the log.",
    )

    items = page(
        "Q3.b | Two further unsuccessful attempts", "Four distinct measured failures in total"
    )
    for failure in [f for f in failures if not f["substantive"]]:
        e = failure["evidence"]
        p(
            items,
            f"<b>{failure['run_id']} versus {failure['control_run_id']}</b>. Hypothesis: {failure['hypothesis']}",
        )
        p(
            items,
            f"Validation F2 {e['attempt_f2']:.4f} versus {e['control_f2']:.4f}; AP {e['attempt_average_precision']:.4f} versus {e['control_average_precision']:.4f}; TP/FP/FN {e['attempt_tp_fp_fn']} versus {e['control_tp_fp_fn']}.",
        )
        p(items, "Control scope: " + failure["controlled_change"])
        p(items, failure["diagnosis"])
        p(items, "Lesson: " + failure["lesson"])
    p(
        items,
        "Each diagnosis is based on train/validation evidence recorded before final evaluation. A low learning rate is unsuccessful within this budget, not intrinsically invalid; more epochs could change the result. A higher rate can add false positives at similar recall, but four epochs cannot establish an eventual optimum. Execution interruptions are separately preserved and do not satisfy the failed-approach count.",
    )

    items = page(
        "Q3.a | Frozen held-out model comparison",
        "All 2,996 shared test examples; validation-frozen decisions",
    )
    rows = []
    for name in [
        "always_normal",
        "always_defective",
        "classical",
        "initial_ann",
        "final_ann",
        "vlm",
    ]:
        r = by_name[name]
        m = r["metrics"]
        timing = r.get("timing")
        latency = (
            timing["mean_end_to_end_ms"]
            if isinstance(timing, dict)
            else r.get("mean_end_to_end_ms")
        )
        train = r.get("training_seconds")
        rows.append(
            [
                name.replace("_", " "),
                fmt(m["f2"]),
                fmt(m["precision"]),
                fmt(m["recall"]),
                f"{m['tp']}/{m['fp']}/{m['fn']}",
                fmt(train, 1) if train is not None else "no task fit",
                fmt(latency, 2),
            ]
        )
    table(
        items,
        ["Model", "F2", "Precision", "Recall", "TP/FP/FN", "Train (s)", "Mean ms"],
        rows,
        [95, 50, 55, 50, 95, 65, 55],
    )
    table(
        items,
        ["Scored model", "AP", "ROC-AUC", "FPR", "P95 total ms"],
        [
            [
                n.replace("_", " "),
                fmt(by_name[n]["metrics"].get("average_precision")),
                fmt(by_name[n]["metrics"].get("roc_auc")),
                fmt(by_name[n]["metrics"]["false_positive_rate"]),
                fmt(by_name[n]["timing"]["p95_end_to_end_ms"], 2),
            ]
            for n in ["classical", "initial_ann", "final_ann"]
        ],
        [130, 70, 70, 70, 125],
    )
    p(
        items,
        (
            "The final ANN and tuned classical model tie on the primary held-out metric. "
            if by_name["final_ann"]["metrics"]["f2"] == by_name["classical"]["metrics"]["f2"]
            else f"The {winner.replace('_', ' ')} has the higher F2 between the ANN and classical families. "
        )
        + (
            "The conditional ANN expectation is supported. "
            if by_name["final_ann"]["metrics"]["f2"] > by_name["classical"]["metrics"]["f2"]
            else "The conditional expectation of ANN superiority is not supported. "
        )
        + "The always-defective reference is retained to expose the benefit of predicting every patch positive under F2. "
        + (
            "Neither learned family exceeds that reference on F2; the scores do not establish a useful production detector. "
            if max(by_name["final_ann"]["metrics"]["f2"], by_name["classical"]["metrics"]["f2"])
            <= by_name["always_defective"]["metrics"]["f2"]
            else "At least one learned family exceeds that reference on F2; this alone does not establish production suitability. "
        )
        + "No test-driven architecture, cutoff, prompt or seed change follows.",
    )
    p(
        items,
        "Timing: Linux x86_64, CPU only, eight allocated cores; common one-thread final inference, batch one, three warmups per model. Per-image time includes decode, preprocessing and prediction; startup/model loading are excluded. VLM parsing/retries are included. Timings here are fresh reconstruction measurements: the first result files were lost during a runtime reset. No model, threshold or prompt changed, and all per-example outputs were genuinely recomputed. Background checkpoint saving and application recording/encoding overlapped part of VLM inference, so these timings are descriptive, not an isolated benchmark. Historical confusion counts are cross-checks only, never substituted for inference. Constant outputs have no meaningful measured latency. VLM task-specific training is absent; pretrained training cost is unknown.",
    )
    ci = by_name["final_ann"]["uncertainty"]
    p(
        items,
        f"Final ANN group-bootstrap nominal 95% F2 interval: {ci.get('f2_95_percent_interval')}, 1,000 resamples, seed 42, {ci.get('independent_groups')} independent test components. With only two origin groups, resampling yields very few distinct compositions. This is descriptive uncertainty, not reliable coverage of a broader production population.",
    )

    items = page(
        "Q3.a | Error patterns and inspection recommendations",
        "Post-evaluation description only; no retuning",
    )
    table(
        items,
        ["Held-out origin", "N", "Positives", "F2", "Recall", "FP / FN"],
        [
            [
                r["publisher_origin"],
                str(r["n"]),
                str(r["tp"] + r["fn"]),
                fmt(r["f2"]),
                fmt(r["recall"]),
                f"{r['fp']} / {r['fn']}",
            ]
            for r in error_analysis["publisher_origin_metrics"]
        ],
    )
    table(
        items,
        ["Positive annotation size", "Positive patches", "Missed", "Recall"],
        [
            [r["size_bin"], str(r["positives"]), str(r["missed"]), fmt(r["recall"])]
            for r in error_analysis["positive_mask_size_recall"]
        ],
    )
    p(
        items,
        f"Recall is {error_analysis['positive_mask_size_recall'][0]['recall']:.2%} for <=1%-area annotations versus {error_analysis['positive_mask_size_recall'][2]['recall']:.2%} for >5%-area annotations: {error_analysis['positive_mask_size_recall'][0]['missed']}/{error_analysis['positive_mask_size_recall'][0]['positives']} small positives are missed. This supports a tiny-region limitation, but size/origin/texture effects are not isolated. Origin differences may reflect acquisition/texture shift; IDs do not establish physical rolls or manufacturers. Mask area is not verified severity. Q5 examines three actual mistakes; all population error counts retain every test patch.",
    )
    table(
        items,
        ["Recommendation", "Patches", "Defects", "Normal"],
        [
            [r["action"].replace("_", " "), str(r["patches"]), str(r["defects"]), str(r["normal"])]
            for r in error_analysis["recommendations"]
        ],
    )
    p(
        items,
        f"Two validation-selected recommendation cutoffs are low={bundle['recommendation_thresholds']['low']:.6f}, high={bundle['recommendation_thresholds']['high']:.6f}. Assumed costs are missed routine defect 10, any manual review 1, unnecessary normal hold 3. Held-out review/hold burden is {error_analysis['review_or_hold_fraction']:.2%}; {error_analysis['missed_defects_in_routine_band']} defective patches fall in routine inspection. These costs are coursework assumptions; weak ranking and rare positives can make a near-all-routine policy numerically cheap. It is not validated operational safety or uncertainty. Keep ordinary human inspection; never automatically release fabric.",
    )
    p(
        items,
        f"Scores remain uncalibrated (temperature 1); held-out Brier score is {error_analysis['brier_score']:.6f}, validation Brier {bundle['calibration']['raw_brier']:.6f}. Reliability bins use selection data and do not independently validate confidence. The binary F2 cutoff and three-way recommendation cutoffs serve different stated objectives.",
    )

    items = page(
        "Q4 | Functional web application and tests",
        "Fresh local requests to the selected trained ANN",
    )
    items.append(
        (
            "diagram",
            [
                "Inspector uploads a PNG/JPEG patch in the browser",
                "FastAPI validates content, dimensions and upload size",
                "Shared grayscale preprocessing -> frozen trained ANN",
                "Fresh score / frozen binary cutoff / inspection recommendation",
                "Optional defect-logit Grad-CAM -> original and adjustable overlay",
            ],
        )
    )
    p(
        items,
        "FastAPI loads the selected checkpoint once, verifies SHA-256/preprocessing/provenance, then executes the model for every prediction or explanation request. /health reports readiness; /predict and /explain receive multipart images. A lock protects model hooks and gradients across concurrent requests. Missing/corrupt weights produce 503. No random/canned fallback exists. Uploaded pixels are not retained or sent to an external VLM.",
    )
    p(
        items,
        "The frontend shows the original and actual overlay, adjustable opacity, uncalibrated score meaning, cutoff, class, three-way inspection recommendation, checkpoint version, measured timing and limitations. It accepts only single-image 8-bit L/RGB 256x256 PNG/JPEG, at most 5 MiB. It decodes content rather than trusting filename/MIME. The inspector uses highlights to inspect, not as certified segmentation.",
    )
    table(
        items,
        ["Executed check group", "Result"],
        [
            ["Real HTTP validation/parity", f"{len(app['http_checks'])} checks passed"],
            ["Browser interaction", f"{len(app['browser']['checks'])} checks passed"],
            [
                "Normal image content",
                "Grayscale PNG, RGB PNG and JPEG match direct checkpoint inference",
            ],
            [
                "Invalid inputs",
                "Empty/malformed/wrong size/alpha/paletted/oversize/invalid opacity rejected",
            ],
            [
                "Materially different inputs",
                "Blank and random noise produce finite scores; no reliable OOD detector is claimed",
            ],
            [
                "Demonstration",
                "Actual browser recording: fresh prediction, opacity controls and a genuine incorrect prediction",
            ],
        ],
    )
    p(
        items,
        "The recorded walkthrough is scripted browser automation of the live local application, with visible captions explaining actual labels and mistakes. It is not the student live presentation. Recording metadata/transcript, case provenance and model version are included; duration and media integrity are checked. Internal browser QA screenshots are excluded from submission.",
    )

    items = page(
        "Q5.a | Three explainability approaches", "Explanation target, cost and limitations"
    )
    table(
        items,
        ["Method", "What it explains / cost", "Important limitations"],
        [
            [
                "Grad-CAM [4]",
                "Architecture-specific conv features; one forward and backward pass for the defect logit",
                "Coarse, target/layer-dependent maps; global averaging and ReLU suppress information; not causal or certified boundaries.",
            ],
            [
                "Integrated Gradients [5]",
                "Input attribution relative to a baseline; many gradient evaluations along a path",
                "Reference choice changes attribution; a black/mean fabric is unnatural; pixel noise and interactions can mislead.",
            ],
            [
                "LIME [6]",
                "Model-agnostic local surrogate over perturbed superpixels; many forward calls",
                "Superpixel/perturbation/sampling choices; local fidelity and instability; cutting periodic weave can make off-distribution images.",
            ],
        ],
        [85, 160, 220],
    )
    p(
        items,
        "Grad-CAM is selected because the ANN has convolutional feature maps and the inspector needs a fast, intuitive spatial cue alongside the original patch. It targets the defect logit even for normal predictions, using gradient averages to weight feature channels, ReLU, bilinear alignment to 256x256 and maximum normalisation. A genuinely zero attribution remains zero: the UI does not fabricate a hotspot.",
    )
    p(
        items,
        "Custom CNN uses the last block ReLU before pooling, retaining twice the final pooled map resolution; MobileNet uses the final feature block. Mask correspondence measures spatial alignment with annotations; faithfulness needs sensitivity tests. Neither constitutes an explanation of the manufacturing cause. The original image, score and frozen cutoff remain visible so a hot overlay cannot replace visual judgement.",
    )
    p(
        items,
        "A suitable next validation would compare different baselines/perturbations and model-parameter randomisation across more independent origins. Those would be new preregistered studies, not post-test retuning of this submission.",
    )

    items = page(
        "Q5.b | Measured explanation sanity checks",
        "Matched-area perturbation controls and genuine mistakes",
    )
    cases = xai["cases"]
    positive = [c for c in cases if c["label"] == 1]
    supported = sum(
        not c["occlusion"]["degenerate_zero_cam"]
        and c["occlusion"]["logit_drops"]["top"] > 0
        and c["occlusion"]["logit_drops"]["top"]
        > max(c["occlusion"]["logit_drops"]["low"], c["occlusion"]["logit_drops"]["random"])
        for c in cases
    )
    hits = [
        c["spatial"]["pointing_hit"] for c in positive if c["spatial"]["pointing_hit"] is not None
    ]
    p(
        items,
        f"The predeclared explanation sample selects up to nine TP/FN/FP/TN cases each, preferring distinct source images then filename order. It contains {len(cases)} cases, {xai['genuine_error_count']} genuine errors; all 2,996 test examples remain in primary metrics. {supported}/{len(cases)} cases have a nonzero CAM and positive top-region logit drop larger than both controls. This does not establish faithfulness for all cases.",
    )
    p(
        items,
        "Occlusion selects the same 10% pixel area from highest CAM, lowest CAM and a seed-42 random mask; each is replaced with its within-image channel mean. The response is original defect logit minus perturbed logit. Positive drop means decreased defect evidence; a negative drop means the score increased. A positive top drop exceeding both controls supports local sensitivity only. Mean-fill occlusion can make implausible texture, and tied/zero CAM ranking can be arbitrary.",
    )
    rows = []
    for c in cases[:2]:
        d = c["occlusion"]["logit_drops"]
        rows.append(
            [
                c["patch_id"],
                str(c["label"]) + "/" + str(c["prediction"]),
                fmt(d["top"], 6),
                fmt(d["low"], 6),
                fmt(d["random"], 6),
            ]
        )
    table(
        items,
        ["Measured case", "Label/pred", "Top drop", "Low drop", "Random drop"],
        rows,
        [155, 65, 75, 75, 75],
    )
    p(
        items,
        f"For positive annotated cases with a nondegenerate map, observed pointing hits are {sum(hits)}/{len(hits)}. Attribution mass in the mask is compared against uniform mask-area mass per case. Negative patches have empty masks, so pointing/mass measures are undefined rather than invented zeros. Mask overlap is correspondence, not proof that the network used the true defect cause.",
    )
    p(
        items,
        "The next three pages analyse three genuinely incorrect predictions selected from this sample, including both error types where available. Numerical checks can contradict visual attractiveness; adverse checks are kept and interpreted. No explanation or error diagnosis is used to revise frozen weights, thresholds or prompts.",
    )

    for c in errors:
        items = page("Q5.b | " + c["case_name"], c["patch_id"] + " | " + c["publisher_origin"])
        figure(items, c["panel"])
        p(items, c["analysis"])
        p(items, "Visual inspection: " + c["visual_analysis"])
        spatial, perturbation = c["spatial"], c["occlusion"]
        point = spatial["pointing_hit"]
        p(
            items,
            f"Spatial correspondence: mask {spatial['mask_status']}; peak-attribution pixel "
            + (
                "not applicable"
                if point is None
                else "inside foreground"
                if point
                else "outside foreground"
            )
            + f"; attribution mass in mask {fmt(spatial.get('attribution_mass_in_mask'), 6)} "
            + f"versus uniform-area reference {fmt(spatial.get('uniform_mass_reference'), 6)}. "
            + "An empty mask has no positive-defect pointing target.",
        )
        p(
            items,
            f"Original defect logit {fmt(perturbation['original_defect_logit'], 6)}; "
            f"occluded area {perturbation['area_fraction']:.2%}; "
            f"top-minus-random drop {fmt(perturbation['top_minus_random_drop'], 6)}; "
            f"top-minus-low drop {fmt(perturbation['top_minus_low_drop'], 6)}; "
            f"zero CAM {perturbation['degenerate_zero_cam']}. Full precision is retained in the case JSON.",
        )
        p(
            items,
            "The publisher annotation is ground truth for this benchmark. It does not identify an independently verified fabric failure mechanism. A reviewer can reproduce the exact original, mask and overlay from the cited manifest row and saved checkpoint.",
        )

    items = page(
        "Q6.a | Genuine local vision-language evaluation",
        "SmolVLM-256M-Instruct, validation-selected prompt",
    )
    p(
        items,
        "The assessed generative model is HuggingFaceTB/SmolVLM-256M-Instruct [7], Apache 2.0, immutable revision 7e3e67edbbed1bf9888184d9df282b700a323964. A compact image-text generative model is meaningful for direct binary patch decisions and feasible on the available CPU. It is a pretrained comparator, not a replacement for the coursework trained ANN.",
    )
    p(
        items,
        "Configuration: transformers 4.57.1; real downloaded safetensors; trust_remote_code=False; bfloat16; CPU one thread; greedy decoding; 24 new tokens maximum; one retry after parsing/inference failure; batch one. The underlying 256x256 pixels are preserved without tiling/upscaling. The processor uses 16 matching visual tokens at this size. This compression may lose tiny defects. No paid API is used; hardware, electricity, local compute time and initial download cost still exist.",
    )
    for index in [0, 1]:
        r = read(f"experiments/vlm-validation-{index}.json")
        p(items, f"<b>Prompt {index}</b>: {r['config']['prompt']}")
        p(
            items,
            f"All {r['metrics']['n']} validation patches: F2 {r['metrics']['f2']:.4f}, precision {r['metrics']['precision']:.4f}, recall {r['metrics']['recall']:.4f}; unresolved {r['metrics']['unresolved_count']} ({r['failure_rate']:.2%}).",
        )
    p(
        items,
        f"Selected prompt index {frozen['vlm']['config']['prompt_index']} by validation F2, then lower failure rate, then smaller index. Both prompts have zero correctly classified normal validation patches; their high recall does not establish useful discrimination. The exact prompt/config/parser and revision are frozen before any test score. The saved validation prefix was completed under the unchanged protocol, with raw responses retained; partial metrics never selected a prompt.",
    )
    p(
        items,
        "Inference receives image pixels and prompt only. PNG metadata is stripped; labels, masks, revealing filenames, source groups, ANN scores and test-derived hints never reach the model. Parsing requires one exact defective/defect_free token or a single canonical decision sentence. Ambiguous prose is unresolved after the fixed retry, retained and counted as error. Responses, attempts, input/output tokens and timings are saved for every example.",
    )

    items = page(
        "Q6.b | Comparison, deployment conditions and extension", "Measured capabilities and limits"
    )
    m = vlm["metrics"]
    am = by_name["final_ann"]["metrics"]
    table(
        items,
        ["Evidence", "Trained ANN", "Generative VLM"],
        [
            ["Test population", "2,996 same frozen patches", "2,996 same frozen patches"],
            ["Defect-class F2", fmt(am["f2"]), fmt(m["f2"])],
            [
                "Precision / recall",
                fmt(am["precision"]) + " / " + fmt(am["recall"]),
                fmt(m["precision"]) + " / " + fmt(m["recall"]),
            ],
            ["FP / FN", f"{am['fp']} / {am['fn']}", f"{m['fp']} / {m['fn']}"],
            [
                "Mean end-to-end latency",
                fmt(by_name["final_ann"]["timing"]["mean_end_to_end_ms"], 2) + " ms",
                fmt(vlm["mean_latency_ms"], 2) + " ms",
            ],
            ["Unresolved output", "0", str(m["unresolved_count"])],
            [
                "Ranking / probability",
                "Continuous uncalibrated model score",
                "Label only; no invented AP/AUC/probability",
            ],
            [
                "Explanation",
                "Measured defect-logit Grad-CAM",
                "Generated label is not a faithful causal explanation",
            ],
        ],
        [135, 160, 170],
    )
    p(
        items,
        "The compact ANN has a small fixed interface, lower dependencies and a reproducible local checkpoint, but needs representative labelled origins and can miss tiny/unseen textures. The VLM can be prompted without task-specific weight fitting and might support adjacent descriptions; it has more dependency/download/compute burden, compressed visual detail and formatting failures. Its pretrained data provenance cannot establish zero benchmark contamination. Neither result guarantees unseen factory performance.",
    )
    p(
        items,
        "Deployment should follow measured F2, false-negative tolerance, inspection capacity and latency, rather than architecture prestige. Local ANN serving keeps uploaded factory imagery in the process; this study also runs the VLM locally. A hosted VLM would add external availability, data-governance, pricing and reproducibility concerns. Those costs are not incurred or numerically invented here.",
    )
    p(
        items,
        "A realistic extension is an inspection-note assistant that converts verified fields - inspector-confirmed defect status, patch ID and observed location - into a short structured note. It must quote the confirmed fields, avoid invented severity/root cause and request human review. A new origin-held-out study could then compare usefulness and unsupported claims. Current Q6 is the direct full-population classification experiment; no adjacent-task exemption is claimed.",
    )

    items = page(
        "Appendix A | AI declaration and reproducibility",
        "Technical verification and individual responsibility",
    )
    for paragraph in Path("docs/ai-use-declaration.md").read_text().split("\n\n"):
        if paragraph.startswith("#"):
            continue
        p(items, paragraph)
    p(
        items,
        f"Executed verification: {checks['pytest_passed']} tests, lint/compilation and actual HTTP/browser checks passed. All actual notebook cells executed in separate IPython processes with rich outputs; the hosted environment blocks the standard Jupyter socket transport. The report is rendered/visually checked and the recording decoded/duration checked. See notebook-verification.json, docs/requirements-matrix.md and verification.json for evidence.",
    )
    p(
        items,
        "Quick verification from the extracted project: install the pinned CPU environment with test/vlm/notebooks extras; run python -m pytest -q; launch uvicorn using FABRIC_BUNDLE=artifacts/serving-bundle.json and FABRIC_CPU_THREADS=1; upload authentic data/demo cases. Full reproduction downloads the verified release and pinned pretrained weights, then uses a fresh output folder. Existing frozen results are never overwritten or silently reevaluated.",
    )
    p(
        items,
        "The student must review the actual code, report, logs and recording, explain the split/metric/architecture/controls/errors/XAI/VLM, and deliver the individual live presentation. Lecturer approval is recorded as the student-confirmed approval of the proposed problem/dataset; no approval letter or completed oral defence is fabricated. The viva guide supplies concrete questions and reproduction pointers.",
    )

    items = page(
        "Appendix B | References and evidence map",
        "Primary sources and local reproducible artifacts",
    )
    refs = [
        "[1] Perez-Llorens, R.; Albero-Albero, T.; Silvestre-Blanes, J. (2026). RAW-FABRID V1. Mendeley Data. DOI 10.17632/db6g85xsyg.1. https://data.mendeley.com/datasets/db6g85xsyg/1",
        "[2] Howard, A. et al. (2019). Searching for MobileNetV3. https://arxiv.org/abs/1905.02244",
        "[3] PyTorch torchvision. MobileNetV3 Small ImageNet weights and model documentation. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html",
        "[4] Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. https://arxiv.org/abs/1610.02391",
        "[5] Sundararajan, M.; Taly, A.; Yan, Q. (2017). Axiomatic Attribution for Deep Networks. https://arxiv.org/abs/1703.01365",
        "[6] Ribeiro, M. T.; Singh, S.; Guestrin, C. (2016). Why Should I Trust You?: Explaining the Predictions of Any Classifier. https://arxiv.org/abs/1602.04938",
        "[7] Hugging Face. SmolVLM-256M-Instruct model card; pinned commit recorded in this report. https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
        "[8] Scikit-learn. LinearSVC and SVC documentation. https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
    ]
    for ref in refs:
        p(items, ref)
    table(
        items,
        ["Claim / output", "Exact evidence"],
        [
            ["Dataset, split, labels", "data/manifests/split.json / split.csv / data-audit.json"],
            [
                "EDA and modelling constraints",
                "docs/eda-statistics.json / eda-modelling-findings.json",
            ],
            ["Every training run", "experiments/*.json; reports/experiment-log.csv"],
            ["Architecture and curves", "reports/*-architecture.csv; reports/figures/"],
            ["Four failures", "docs/failed-approaches.json"],
            [
                "Frozen protocol / reconstruction",
                "artifacts/final-evaluation-manifest.json; reports/final-evaluation/test-access.json",
            ],
            [
                "Per-example final scores",
                "reports/final-evaluation/{initial_ann,final_ann,classical,vlm}.json",
            ],
            [
                "Three genuine XAI errors",
                "docs/three-xai-errors.json; reports/final-evaluation/xai/",
            ],
            [
                "Request-time application",
                "reports/application-verification/results.json / browser-results.json",
            ],
            [
                "Tests, notebooks and video",
                "reports/execution-checks.json / notebook-verification.json / demonstration-metadata.json",
            ],
        ],
        [135, 330],
    )
    assert len(pages) == 26, len(pages)
    assert all(page["title"].startswith("Q3.a") for page in pages[14:16])
    assert all(page["title"].startswith("Q3.b") for page in pages[10:14])
    pages = pages[:10] + pages[14:16] + pages[10:14] + pages[16:]

    # Consistent portable text typography; chart pixels retain their scientific labels.
    def plain_hyphens(value):
        return str(value).translate({ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})

    for page_data in pages:
        page_data["title"] = plain_hyphens(page_data["title"])
        page_data["subtitle"] = plain_hyphens(page_data["subtitle"])
        cleaned = []
        for kind, value in page_data["items"]:
            if kind == "p":
                value = plain_hyphens(value)
            elif kind == "table":
                head, rows, widths = value
                value = (
                    [plain_hyphens(v) for v in head],
                    [[plain_hyphens(v) for v in row] for row in rows],
                    widths,
                )
            elif kind == "diagram":
                value = [plain_hyphens(v) for v in value]
            cleaned.append((kind, value))
        page_data["items"] = cleaned
    # Output the same numerical/prose content as a reviewable text source.
    md = []
    for page_data in pages:
        md.extend(["# " + page_data["title"], "", page_data["subtitle"], ""])
        for kind, value in page_data["items"]:
            if kind == "p":
                md.extend([value.replace("<b>", "**").replace("</b>", "**"), ""])
            elif kind == "table":
                head, rows, _ = value
                md.extend(
                    ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
                )
                md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
                md.append("")
            elif kind == "image":
                md.extend(["![](" + value[0].replace("reports/", "") + ")", ""])
            elif kind == "diagram":
                md.extend(["Architecture: " + " -> ".join(value), ""])
    Path("reports/final_report.md").write_text("\n".join(md))
    font = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("Body", "DejaVuSans.ttf"), ("Strong", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font / file)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Strong", italic="Body", boldItalic="Strong"
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyText2",
            fontName="Body",
            fontSize=9,
            leading=13.1,
            spaceAfter=9,
            textColor=colors.HexColor("#243746"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Title2",
            fontName="Strong",
            fontSize=19,
            leading=24,
            spaceAfter=9,
            textColor=colors.HexColor("#153949"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Sub2",
            fontName="Body",
            fontSize=9.5,
            leading=13,
            spaceAfter=15,
            textColor=colors.HexColor("#257d87"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell2",
            fontName="Body",
            fontSize=7.4,
            leading=10.0,
            textColor=colors.HexColor("#243746"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Head2", fontName="Strong", fontSize=7.5, leading=10, textColor=colors.white
        )
    )
    story = []
    for index, page_data in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(xml.escape(page_data["title"]), styles["Title2"]))
        story.append(Paragraph(xml.escape(page_data["subtitle"]), styles["Sub2"]))
        for kind, value in page_data["items"]:
            if kind == "p":
                safe = xml.escape(value).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
                story.append(Paragraph(safe, styles["BodyText2"]))
            elif kind == "image":
                from PIL import Image as PILImage

                w, h = PILImage.open(value[0]).size
                width = value[1] * inch
                story.append(Image(value[0], width=width, height=width * h / w))
                story.append(Spacer(1, 10))
            elif kind == "diagram":
                story.append(ArchitectureDiagram(value))
                story.append(Spacer(1, 6))
            elif kind == "table":
                head, rows, widths = value
                data = [[Paragraph(xml.escape(str(v)), styles["Head2"]) for v in head]] + [
                    [Paragraph(xml.escape(str(v)), styles["Cell2"]) for v in row] for row in rows
                ]
                t = Table(
                    data,
                    colWidths=widths or [465 / len(head)] * len(head),
                    repeatRows=1,
                    hAlign="LEFT",
                )
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#185969")),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.HexColor("#f0f5f7"), colors.white],
                            ),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 4 if len(rows) > 10 else 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4 if len(rows) > 10 else 6),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#bfd1d9")),
                        ]
                    )
                )
                story.extend([t, Spacer(1, 12)])

    def decorate(c, doc):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#bfd1d9"))
        c.line(48, 790, 547, 790)
        c.setFont("Body", 7.2)
        c.setFillColor(colors.HexColor("#617889"))
        c.drawString(
            48, 802, "RAW-FABRID  |  Individual ANN coursework  |  Frozen supervised protocol"
        )
        c.drawString(48, 30, "Real measured evidence; no test-driven tuning")
        c.drawRightString(547, 30, str(doc.page))
        c.restoreState()

    path = Path("reports/final_report.pdf")
    doc = SimpleDocTemplate(
        str(path),
        pagesize=(595.28, 841.89),
        leftMargin=48,
        rightMargin=48,
        topMargin=63,
        bottomMargin=52,
        title="Risk-aware, explainable fabric defect detection",
        author="Individual coursework technical report",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    from pypdf import PdfReader

    actual = PdfReader(path)
    assert len(actual.pages) == 26, f"Unexpected overflow: {len(actual.pages)} PDF pages"
    print("Wrote 24 main-report pages plus 2 appendix pages:", path)


if __name__ == "__main__":
    main()
