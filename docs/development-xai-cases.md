# Three genuine development errors analysed with XAI

**Synthetic development evidence only. These are not the approved-dataset assessed case studies.** All three are actual false positives from the saved three-epoch `ann-smoke` checkpoint and frozen development validation partition. The normal labels have empty defect masks; no false-negative case is invented. The complete run has 13 false positives and no false negatives at its validation-selected classification cutoff of approximately 0.500314.

Original images and computed overlays were inspected together. See `development-run/reports/xai-validation/` for the individual overlays and measured case records. All three largely repeat the expected regular texture. Grad-CAM is concentrated in a thin strip along the lower image boundary rather than a clear local fabric defect. This suggests a hypothesis of boundary/texture sensitivity, possibly involving convolutional padding; it does not establish a cause.

| Case | True / predicted | Model score | Top-area logit drop | Low-area drop | Random-area drop |
| --- | --- | ---: | ---: | ---: | ---: |
| `RAW_FAB_0001_p0_0` | normal / defective | 0.501859 | -0.000532 | -0.000734 | -0.000586 |
| `RAW_FAB_0001_p0_1` | normal / defective | 0.501113 | -0.000580 | -0.000826 | -0.000115 |
| `RAW_FAB_0001_p0_5` | normal / defective | 0.500572 | -0.000763 | -0.001033 | -0.000837 |

For case 1, the score is only slightly above the cutoff and the overlay favours the border despite a normal annotation. Occluding its top-attribution area **increases** the defect logit (negative drop). The top-minus-random difference is small, about +0.000054, and does not supply strong evidence of useful defect sensitivity.

For case 2, top-attribution deletion increases the logit more than random deletion: top-minus-random drop is about -0.000464. This goes against the simple expectation that removing highlighted evidence should reduce the defect score. The explanation/perturbation combination should be reported as an unconvincing sensitivity check, not called successful solely because an overlay was produced.

For case 3, the score lies especially close to the cutoff. Top-minus-random drop is about +0.000074, but both top and random deletion increase the logit. The relative comparison is weak and the absolute direction does not support the intended faithfulness argument. A narrowly selected threshold and poorly separated scores can produce a false positive even when the visual attribution looks orderly.

Empty normal masks make positive-defect pointing and mass-inside-mask metrics inapplicable; they are recorded as unavailable rather than treated as perfect overlap. These cases demonstrate why Grad-CAM is not a certified defect boundary or causal proof. Mean-fill occlusion can itself change texture unnaturally. The approved real study must inspect at least two sanity-check cases and analyse three genuine incorrect predictions from its actual frozen validation/test evidence, including false positives/negatives where available, without manufacturing errors or changing a good model to obtain them.
