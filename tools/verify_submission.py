"""Check technical completion against the measured coursework evidence without rerunning test selection."""

import csv
import math
from pathlib import Path

import nbformat
from PIL import Image

from fabric_inspector.common import code_hash, now, read_json, resolve_under, sha256, write_json
from fabric_inspector.data.manifest import REAL, load_manifest, read_mask
from fabric_inspector.evaluation.metrics import metrics
from fabric_inspector.genai.vlm import parse_decision


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def main():
    checks = []

    def passed(name, evidence):
        checks.append({"check": name, "status": "PASS", "evidence": evidence})

    split, rows, root = load_manifest(Path("data/manifests/split.json"), verify_files=True)
    assert split["provenance"] == REAL and len(rows) == 19852
    assert sum(int(r["label"]) for r in rows) == 687
    for field in ["source_id", "group_id", "verified_extra_group"]:
        partitions = {}
        for row in rows:
            partitions.setdefault(row[field], set()).add(row["split"])
        assert all(len(values) == 1 for values in partitions.values()), field
    assert len({r["verified_extra_group"] for r in rows}) == 7
    passed(
        "Full authentic data, publisher labels and source/origin isolation",
        "data/manifests/split.json and split.csv",
    )

    sample, sample_rows, sample_root = load_manifest(
        Path("data/sample-manifests/split.json"), verify_files=True
    )
    assert (
        len(sample_rows) == 168 and sample["sample_of_full_split_sha256"] == split["split_sha256"]
    )
    by_id = {r["patch_id"]: r for r in rows}
    for row in sample_rows:
        original = by_id[row["patch_id"]]
        assert row["sha256"] == original["sha256"] and row["label"] == original["label"]
        assert row["split"] == original["split"]
        with Image.open(resolve_under(sample_root, row["image_path"])) as image:
            assert image.size == (256, 256) and image.mode == "L"
        if row["mask_path"]:
            mask = read_mask(
                resolve_under(sample_root, row["mask_path"]), row.get("mask_foreground") or "binary"
            )
            assert int((mask != 0).sum()) == int(row["defect_pixels"])
            assert int(mask.any()) == int(row["label"])
    passed(
        "Authentic engineering sample, paths and mask-label agreement",
        "data/sample-manifests/split.json: 168 images, original assignments",
    )

    ann = [
        read_json(p) for p in sorted(Path("experiments").glob("ann-search-[0-9][0-9][0-9].json"))
    ]
    svm = [
        read_json(p) for p in sorted(Path("experiments").glob("svm-search-[0-9][0-9][0-9].json"))
    ]
    assert len(ann) == 21 and len(svm) == 28
    assert all(r["status"] == "COMPLETE" for r in ann + svm)
    assert all(
        r["provenance"] == REAL and r["split_sha256"] == split["split_sha256"] for r in ann + svm
    )
    assert all(r["validation_metrics"]["n"] == 3472 for r in ann + svm)
    assert all("PENDING" not in r["lesson"] for r in ann + svm)
    records = {r["run_id"]: r for r in ann + svm}
    failures = read_json(Path("docs/failed-approaches.json"))["attempts"]
    assert len(failures) >= 4 and len({r["run_id"] for r in failures}) >= 4
    assert sum(bool(r["substantive"]) for r in failures) >= 2
    for failure in failures:
        attempt = records[failure["run_id"]]
        control = records[failure["control_run_id"]]
        assert attempt["validation_metrics"]["f2"] < control["validation_metrics"]["f2"]
        close(failure["evidence"]["attempt_f2"], attempt["validation_metrics"]["f2"])
        close(failure["evidence"]["control_f2"], control["validation_metrics"]["f2"])
        assert failure["diagnosis"] and failure["lesson"]
    passed(
        "Full ANN/classical searches and four genuine unsuccessful attempts",
        "21 ANN candidates, 24 main classical configurations + 4 pilots; docs/failed-approaches.json",
    )

    frozen_path = Path("artifacts/final-evaluation-manifest.json")
    frozen = read_json(frozen_path)
    assert frozen["code_sha256"] == code_hash(Path("."))
    assert frozen["split_sha256"] == split["split_sha256"]
    assert frozen["manifest_sha256"] == sha256(Path("data/manifests/split.json"))
    for path, digest in frozen["artifact_sha256"].items():
        assert sha256(resolve_under(frozen_path.parent, path)) == digest
    guard = read_json(Path("reports/final-evaluation/test-access.json"))
    assert guard["status"] == "COMPLETE" and guard["frozen_manifest_sha256"] == sha256(frozen_path)
    test_rows = [r for r in rows if r["split"] == "test"]
    ids = [r["patch_id"] for r in test_rows]
    assert len(ids) == 2996 and ids == frozen["test_patch_ids"]
    comparison = read_json(Path("reports/final-evaluation/comparison.json"))
    assert comparison["status"] == "COMPLETE" and comparison["provenance"] == REAL
    compared = {r["model"]: r for r in comparison["comparisons"]}
    assert set(compared) == {
        "always_normal",
        "always_defective",
        "classical",
        "initial_ann",
        "final_ann",
        "vlm",
    }
    assert all(r["metrics"]["n"] == 2996 for r in compared.values())
    for name in [
        "initial_ann",
        "final_ann",
        "classical",
        "seed_repeat_0",
        "seed_repeat_1",
        "seed_repeat_2",
    ]:
        result = read_json(Path("reports/final-evaluation") / (name + ".json"))
        predictions = result["predictions"]
        assert [p["patch_id"] for p in predictions] == ids
        assert all(p["label"] == int(row["label"]) for p, row in zip(predictions, test_rows))
        recomputed = metrics(
            [p["label"] for p in predictions],
            [p["prediction"] for p in predictions],
            [p["score"] for p in predictions],
        )
        for key, value in recomputed.items():
            if value is None:
                assert result["metrics"][key] is None
            else:
                close(result["metrics"][key], value)
    passed(
        "Frozen single held-out evaluation with shared examples and reproducible metrics",
        "final-evaluation-manifest.json; test-access.json; all 2,996 IDs and per-example metrics",
    )

    seeds = read_json(Path("artifacts/seed-variability.json"))
    assert seeds["seeds"] == [42, 123, 2026] and len(seeds["run_ids"]) == 3
    for run_id in seeds["run_ids"]:
        r = read_json(Path("experiments") / (run_id + ".json"))
        assert r["status"] == "COMPLETE" and r["split_sha256"] == split["split_sha256"]
    passed(
        "Three genuine selected-model repetitions",
        "artifacts/seed-variability.json and frozen seed-repeat test results",
    )

    for i in range(2):
        result = read_json(Path(f"experiments/vlm-validation-{i}.json"))
        assert result["status"] == "COMPLETE" and result["metrics"]["n"] == 3472
        assert len(result["examples"]) == 3472 and result["provenance"] == REAL
    vlm = read_json(Path("reports/final-evaluation/vlm.json"))
    assert vlm["status"] == "COMPLETE" and [e["patch_id"] for e in vlm["examples"]] == ids
    assert all(e["image_sha256"] == r["sha256"] for e, r in zip(vlm["examples"], test_rows))
    assert all(e["attempts"] for e in vlm["examples"])
    assert vlm["config"]["revision"] == "7e3e67edbbed1bf9888184d9df282b700a323964"
    assert "average_precision" not in vlm["metrics"] and "roc_auc" not in vlm["metrics"]
    assert vlm["config"] == frozen["vlm"]["config"]
    recomputed_vlm = metrics(
        [int(r["label"]) for r in test_rows], [e["prediction"] for e in vlm["examples"]]
    )
    for key, value in recomputed_vlm.items():
        close(vlm["metrics"][key], value)
    for entry in vlm["examples"]:
        if entry["prediction"] != -1:
            assert (
                parse_decision(
                    entry["attempts"][-1]["raw_response"], vlm["config"]["response_format"]
                )
                == entry["prediction"]
            )
    raw = Path("reports/final-evaluation/vlm.responses.jsonl")
    assert len(raw.read_text().splitlines()) == 2996
    import json

    assert [json.loads(line) for line in raw.read_text().splitlines()] == vlm["examples"]
    if comparison.get("workspace_recovery"):
        recovery = read_json(Path(comparison["workspace_recovery"]))
        assert recovery["status"] == "COMPLETE"
        assert recovery["physical_test_computation_count"] == 2
        assert recovery["predictive_source_sha256"] == frozen["code_sha256"]
        assert recovery["frozen_manifest_sha256"] == sha256(frozen_path)
        assert recovery["model_selection_changed"] is False
        assert recovery["training_repeated"] is False
        assert recovery["threshold_prompt_split_changed"] is False
        assert recovery["original_guard_deleted_by_assistant"] is False
    passed(
        "Actual pinned generative model, full validation/test outputs, no simulated scores",
        "vlm-validation-0/1.json; final-evaluation/vlm.json and raw responses",
    )

    errors = read_json(Path("docs/three-xai-errors.json"))
    assert errors["run_id"] == frozen["serving_bundle"]["run_id"] and len(errors["cases"]) >= 3
    final_predictions = {
        p["patch_id"]: p
        for p in read_json(Path("reports/final-evaluation/final_ann.json"))["predictions"]
    }
    for case in errors["cases"]:
        prediction = final_predictions[case["patch_id"]]
        assert case["incorrect"] and prediction["label"] != prediction["prediction"]
        close(case["score"], prediction["score"])
        assert case["analysis"] and case.get("visual_analysis"), (
            "Each panel needs an actual visual analysis"
        )
        assert Path(case["panel"]).is_file()
        assert all(math.isfinite(v) for v in case["occlusion"]["logit_drops"].values())
    passed(
        "Three genuine wrong predictions, actual visual analysis and XAI sensitivity",
        "docs/three-xai-errors.json; three rendered image/mask/CAM panels",
    )

    app_path = Path("reports/application-verification/results.json")
    app = read_json(app_path)
    assert app["status"] == "PASS" and len(app["http_checks"]) == 13
    assert app["browser"]["status"] == "PASS"
    assert app["run_id"] == frozen["serving_bundle"]["run_id"]
    assert any(e.get("incorrect") for e in app["browser"]["events"])
    passed(
        "Working application, fresh prediction/explanation and browser verification",
        "13 live HTTP checks plus browser checks and real erroneous prediction",
    )

    video = read_json(Path("reports/demonstration-metadata.json"))
    assert video["status"] == "PASS" and 0 < video["duration_seconds"] <= 480
    assert video["video_sha256"] == sha256(Path(video["path"]))
    assert video["application_verification_sha256"] == sha256(app_path)
    assert video["genuine_incorrect_prediction_shown"] and video["scripted_automation_declared"]
    assert video.get("visual_verification") == "PASS", (
        "Inspect actual recorded frames before completion"
    )
    passed(
        "Actual application demonstration, <=8 minutes, full-frame decode and visual QA",
        "reports/demonstration.mp4 and demonstration-metadata.json",
    )

    execution = read_json(Path("reports/execution-checks.json"))
    assert execution["status"] == "PASS" and execution["pytest_passed"] >= 52
    assert execution["code_sha256"] == frozen["code_sha256"]
    assert execution["lint"] == "PASS" and execution["compilation"] == "PASS"
    notebooks = read_json(Path("reports/notebook-verification.json"))
    assert notebooks["status"] == "PASS" and len(notebooks["notebooks"]) == 4
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None
                assert not any(output.output_type == "error" for output in cell.outputs)
    passed(
        "All automated tests, compilation/lint and four executed notebooks",
        "reports/execution-checks.json; notebook-verification.json; notebooks/*.ipynb",
    )

    pdf = read_json(Path("reports/pdf-verification.json"))
    assert pdf["status"] == "PASS" and pdf["main_pages"] <= 25 and pdf["pages"] == 26
    assert pdf["pdf_sha256"] == sha256(Path("reports/final_report.pdf"))
    assert pdf["visually_checked_pages"] == list(range(1, 27))
    assert "PENDING" not in Path("reports/final_report.md").read_text()
    for path in [
        "README.md",
        "docs/requirements-matrix.md",
        "docs/ai-use-declaration.md",
        "docs/viva-guide.md",
    ]:
        assert Path(path).is_file() and Path(path).stat().st_size > 500
    with Path("reports/experiment-log.csv").open(newline="") as stream:
        logged = list(csv.DictReader(stream))
    assert len(logged) >= 51
    passed(
        "Question-ordered report, all pages visually checked, complete logs and truthful documentation",
        "24 main pages + 2 appendices; final_report.pdf; experiment-log.csv; requirements matrix",
    )

    write_json(
        Path("verification.json"),
        {
            "technical_status": "COMPLETE",
            "verified_at": now(),
            "provenance": REAL,
            "code_sha256": frozen["code_sha256"],
            "split_sha256": split["split_sha256"],
            "checks": checks,
            "pytest_passed": execution["pytest_passed"],
            "dataset": {
                "patches": 19852,
                "source_images": 709,
                "positive_patches": 687,
                "test_examples": 2996,
            },
            "experiments": {
                "ann_candidates": len(ann),
                "classical_configurations_including_pilots": len(svm),
                "selected_model_seeds": 3,
                "vlm_validation_prompts": 2,
            },
            "report": {"pages": pdf["pages"], "main_pages": pdf["main_pages"]},
            "demonstration_seconds": video["duration_seconds"],
            "pending_technical": [],
            "human_assessment": "Student must personally review, understand and defend the work in the individual live presentation. Technical verification does not certify a grade or completed oral defence.",
        },
    )
    print(f"COMPLETE: {len(checks)} technical evidence checks passed")


if __name__ == "__main__":
    main()
