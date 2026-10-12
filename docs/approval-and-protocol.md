# Approval and supervised protocol

The student stated on 11 October 2026 that the suggested proposal was approved by the lecturer. This is recorded as student-confirmed approval, not an invented lecturer email or signature. Retain the actual approval correspondence for assessment.

The approved scope is **Risk-Aware, Explainable Fabric Defect Detection for Textile Quality Control**: an industrial fabric image patch, a Quality Assurance Inspector, a binary patch label, and inspection prioritisation with the human decision retained. The supplied proposal identifies RAW-FABRID V1 and a new supervised research split using source/duplicate grouping. The implemented real-data workflow follows that proposal. No new dataset or problem is substituted.

The full official release was acquired, checksum-verified and imported on 11 October 2026. The supplied historical development textures remain artificial engineering history. The source image, physical roll, capture event and manufacturer are different concepts. The real publisher source CSV supplies seven `roll_id` values, described in its README as anonymised fabric origins. Every such origin group is kept intact; its exact relationship to physical rolls, weave types and manufacturers is not asserted.

The new supervised assignment combines the fully labelled publisher patch collection. The original publisher train/good arrangement supports anomaly detection and does not contain positive supervised training examples. Reassigning labelled patches by source before development is therefore intentional and must be reported. It does not support a direct score comparison against the publisher's original anomaly-detection split.

The immutable assignment was created before fitting: 13,384 train, 3,472 validation and 2,996 test patches. The 128 declared integrity-only draws select feasible class/group coverage and ratios without fitting or evaluating any model. Source mapping, publisher origin groups and conservative exact/near duplicate controls prevent observed relationships crossing partitions. Unknown additional physical links remain a limitation. The full split hash is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`.

All preprocessing fits, training, capacity/optimiser searches, calibration, classification thresholds, recommendation cutoffs, VLM prompts and XAI settings use train/validation only. Test access before final scoring is limited to integrity checks and split accounting. The planned final evaluation scores every frozen comparison on the same held-out examples, logs partial failures and permits no silent second attempt.

If later evidence requires changing the approved problem, dataset or comparison protocol, record the change and obtain the lecturer's decision before implementing it. The intended supervised model-selection protocol is unchanged. The subsequent result-recovery exception is documented below.

## Recorded recovery exception

The once-only final-evaluation plan remains the original pre-test record. After the first completed result files were lost in a temporary-workspace reset, the identical frozen inference was physically recomputed to reconstruct them. The original trained weights, model selections, thresholds, prompts and split were unchanged. `docs/engineering-runs/final-results-workspace-reset.json` records both computations and the actual outputs. This is an exception to the coursework brief's strict once-only checklist; lecturer acceptance of the exception has not been verified. The proposal approval does not establish acceptance of this later event. No further population evaluation is performed to finish the submission.
