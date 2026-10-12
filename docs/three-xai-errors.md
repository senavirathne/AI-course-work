# Three genuine incorrect predictions with XAI

## Error 1: false negative

This genuine false negative has ground-truth label 1 and prediction 0; score 0.790539 versus validation-frozen cutoff 0.895234. The publisher mask covers 103 pixels (0.157%). Equal-area top/low/random occlusion changes the defect logit by +1.591740/+2.704260/+0.498417. This case does not support stronger sensitivity of the highlighted region than both controls. Mean-fill perturbation may create unnatural weave; the map does not identify a causal production fault or establish a certified defect boundary.

Visual inspection: The original is predominantly repeating horizontal weave. The publisher foreground is a very small region at the bottom edge toward the right, whereas the brightest Grad-CAM region lies higher in the lower-right interior. The maximum misses the mask and the mask receives 0.00134% of total attribution, below its 0.15717% uniform-area reference. This is consistent with poor attention to tiny boundary evidence, but does not isolate downsampling as the cause. The 0.790539 score falls below the frozen 0.895234 cutoff. Top-region deletion reduces the defect logit, yet low-attribution deletion reduces it more (2.704260 versus 1.591740); the explanation therefore fails the stronger-than-both-controls sensitivity check. A future study could test boundary-preserving features on new validation groups, without changing this evaluated model.

![](../reports/figures/xai-error-1.png)

## Error 2: false negative

This genuine false negative has ground-truth label 1 and prediction 0; score 0.204645 versus validation-frozen cutoff 0.895234. The publisher mask covers 1764 pixels (2.692%). Equal-area top/low/random occlusion changes the defect logit by -0.427480/-0.421962/-0.641229. This case does not support stronger sensitivity of the highlighted region than both controls. Mean-fill perturbation may create unnatural weave; the map does not identify a causal production fault or establish a certified defect boundary.

Visual inspection: The publisher annotation traces an irregular narrow region down the left edge. The original shows subtle left-edge weave/intensity variation; the coarse CAM brightens the lower left and partly overlaps the annotation. Its 21.3191% attribution mass in a 2.69165%-area mask exceeds the uniform reference, but its single maximum remains outside the exact-white mask. Spatial overlap alone does not explain the low 0.204645 score or certify a boundary. All three occlusions increase the defect logit (negative drops); top deletion is -0.427480 versus low -0.421962 and random -0.641229. Thus the highlighted region is not validated as selectively supporting the defect decision by this perturbation. The mean-fill change may itself disrupt the weave; neither a production cause nor annotation severity is established.

![](../reports/figures/xai-error-2.png)

## Error 3: false positive

This genuine false positive has ground-truth label 0 and prediction 1; score 0.925760 versus validation-frozen cutoff 0.895234. The publisher mask covers 0 pixels (0.000%). Equal-area top/low/random occlusion changes the defect logit by +2.779529/+3.000914/-0.592691. This case does not support stronger sensitivity of the highlighted region than both controls. Mean-fill perturbation may create unnatural weave; the map does not identify a causal production fault or establish a certified defect boundary.

Visual inspection: The original has regular weave and a short narrow dark vertical feature in the upper-right region. Grad-CAM peaks nearby, while the publisher foreground is empty and the explicit patch label is normal. The model score 0.925760 exceeds the frozen 0.895234 cutoff, creating a genuine false positive under the publisher target. The highlight is compatible with sensitivity to a visually distinctive texture feature, but does not establish a real unannotated defect or an incorrect publisher label. Empty-mask overlap and pointing-hit scores are undefined, not zero-quality localisation measures. Top deletion lowers the logit by 2.779529, but low-attribution deletion lowers it more (3.000914); random deletion raises it. This again fails the stronger-than-both-controls check. Independent inspector annotation would be needed before revising any label.

![](../reports/figures/xai-error-3.png)
