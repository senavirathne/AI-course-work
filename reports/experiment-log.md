# Complete measured experiment log

All numbers below use the frozen RAW-FABRID validation partition. Full configurations, histories, environment, timestamps and hashes are in the referenced JSON records. Engineering interruptions are retained separately and are not scientific failures.

| Run | Configuration | Val F2 | Val AP | TP / FP / FN | Seconds |
| --- | --- | ---: | ---: | --- | ---: |
| ann-search-000 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.0923 | 0.0188 | 41 / 1973 / 11 | 841.2 |
| ann-search-001 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.0001; batch=32; seed=42; aug=True; weighted=True | 0.0804 | 0.0140 | 48 / 2729 / 4 | 443.1 |
| ann-search-002 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.0003; batch=32; seed=42; aug=True; weighted=True | 0.0941 | 0.0165 | 17 / 678 / 35 | 589.6 |
| ann-search-003 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.003; batch=32; seed=42; aug=True; weighted=True | 0.0881 | 0.0153 | 41 / 2077 / 11 | 450.8 |
| ann-search-004 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=16; seed=42; aug=True; weighted=True | 0.0895 | 0.0170 | 41 / 2042 / 11 | 455.8 |
| ann-search-005 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=64; seed=42; aug=True; weighted=True | 0.0939 | 0.0170 | 15 / 576 / 37 | 405.7 |
| ann-search-006 | {'name': 'custom', 'depth': 2, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.0793 | 0.0139 | 38 / 2149 / 14 | 393.8 |
| ann-search-007 | {'name': 'custom', 'depth': 4, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.1799 | 0.0368 | 15 / 194 / 37 | 481.2 |
| ann-search-008 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; sgd lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.0823 | 0.0180 | 16 / 748 / 36 | 455.9 |
| ann-search-009 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; sgd lr=0.01; batch=32; seed=42; aug=True; weighted=True | 0.0860 | 0.0150 | 40 / 2078 / 12 | 453.1 |
| ann-search-010 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; sgd lr=0.03; batch=32; seed=42; aug=True; weighted=True | 0.0834 | 0.0162 | 33 / 1738 / 19 | 461.2 |
| ann-search-011 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; sgd lr=0.1; batch=32; seed=42; aug=True; weighted=True | 0.1274 | 0.0333 | 8 / 98 / 44 | 466.4 |
| ann-search-012 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=False | 0.0960 | 0.0180 | 38 / 1733 / 14 | 451.9 |
| ann-search-013 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=False; weighted=True | 0.0909 | 0.0226 | 41 / 2007 / 11 | 419.2 |
| ann-search-014 | {'name': 'custom', 'depth': 3, 'width': 16, 'dropout': 0.2}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.0997 | 0.0188 | 18 / 677 / 34 | 633.8 |
| ann-search-015 | {'name': 'custom', 'depth': 3, 'width': 8, 'dropout': 0}; adamw lr=0.001; batch=32; seed=42; aug=True; weighted=True | 0.1097 | 0.0446 | 16 / 505 / 36 | 356.4 |
| ann-search-016 | {'name': 'mobilenet'}; adamw lr=0.0001; batch=32; seed=42; aug=False; weighted=True | 0.2749 | 0.1659 | 21 / 153 / 31 | 639.5 |
| ann-search-017 | {'name': 'mobilenet'}; adamw lr=0.0001; batch=32; seed=42; aug=True; weighted=True | 0.3355 | 0.2962 | 21 / 84 / 31 | 657.7 |
| ann-search-018 | {'name': 'mobilenet'}; adamw lr=0.0003; batch=32; seed=42; aug=False; weighted=True | 0.4856 | 0.4991 | 27 / 43 / 25 | 625.3 |
| ann-search-019 | {'name': 'mobilenet'}; adamw lr=0.0003; batch=32; seed=42; aug=True; weighted=True | 0.4137 | 0.3627 | 23 / 47 / 29 | 670.7 |
| ann-search-020 | {'name': 'mobilenet'}; adamw lr=0.0003; batch=32; seed=42; aug=False; weighted=True | 0.4856 | 0.4991 | 27 / 43 / 25 | 815.2 |
| ann-repeat-123 | {'name': 'mobilenet'}; adamw lr=0.0003; batch=32; seed=123; aug=False; weighted=True | 0.3704 | 0.3623 | 18 / 17 / 34 | 660.1 |
| ann-repeat-2026 | {'name': 'mobilenet'}; adamw lr=0.0003; batch=32; seed=2026; aug=False; weighted=True | 0.4527 | 0.4523 | 22 / 13 / 30 | 693.0 |
| svm-search-000 | lbp; SVC pilot; linear; C=0.1; weight=None | 0.0707 | 0.0132 | 52 / 3420 / 0 | 285.7 |
| svm-search-001 | lbp; SVC pilot; linear; C=0.1; weight=balanced | 0.0784 | 0.0151 | 37 / 2116 / 15 | 13.7 |
| svm-search-002 | lbp; SVC pilot; linear; C=1.0; weight=None | 0.0720 | 0.0123 | 52 / 3350 / 0 | 5.6 |
| svm-search-003 | lbp; SVC pilot; linear; C=1.0; weight=balanced | 0.0786 | 0.0150 | 36 / 2047 / 16 | 36.4 |
| svm-search-024 | lbp; LinearSVC; linear; C=0.1; weight=None | 0.0740 | 0.0148 | 38 / 2323 / 14 | 217.8 |
| svm-search-025 | lbp; LinearSVC; linear; C=0.1; weight=balanced | 0.0823 | 0.0154 | 36 / 1943 / 16 | 0.8 |
| svm-search-026 | lbp; LinearSVC; linear; C=1.0; weight=None | 0.0749 | 0.0148 | 39 / 2356 / 13 | 0.6 |
| svm-search-027 | lbp; LinearSVC; linear; C=1.0; weight=balanced | 0.0819 | 0.0154 | 29 / 1533 / 23 | 0.8 |
| svm-search-028 | lbp; LinearSVC; linear; C=10.0; weight=None | 0.0749 | 0.0148 | 39 / 2356 / 13 | 0.6 |
| svm-search-029 | lbp; LinearSVC; linear; C=10.0; weight=balanced | 0.0819 | 0.0154 | 29 / 1533 / 23 | 0.8 |
| svm-search-030 | lbp; SVC; rbf; C=0.1; weight=None | 0.0713 | 0.0140 | 52 / 3385 / 0 | 3.9 |
| svm-search-031 | lbp; SVC; rbf; C=0.1; weight=balanced | 0.0745 | 0.0151 | 46 / 2835 / 6 | 11.2 |
| svm-search-032 | lbp; SVC; rbf; C=1.0; weight=None | 0.0713 | 0.0140 | 52 / 3385 / 0 | 3.7 |
| svm-search-033 | lbp; SVC; rbf; C=1.0; weight=balanced | 0.0707 | 0.0238 | 52 / 3420 / 0 | 9.4 |
| svm-search-034 | lbp; SVC; rbf; C=10.0; weight=None | 0.0720 | 0.0139 | 52 / 3350 / 0 | 4.4 |
| svm-search-035 | lbp; SVC; rbf; C=10.0; weight=balanced | 0.0707 | 0.0135 | 52 / 3420 / 0 | 6.8 |
| svm-search-036 | hog_lbp; LinearSVC; linear; C=0.1; weight=None | 0.1252 | 0.0331 | 20 / 571 / 32 | 667.6 |
| svm-search-037 | hog_lbp; LinearSVC; linear; C=0.1; weight=balanced | 0.1252 | 0.0332 | 20 / 571 / 32 | 1283.4 |
| svm-search-038 | hog_lbp; LinearSVC; linear; C=1.0; weight=None | 0.1252 | 0.0312 | 20 / 571 / 32 | 1672.1 |
| svm-search-039 | hog_lbp; LinearSVC; linear; C=1.0; weight=balanced | 0.1303 | 0.0297 | 19 / 502 / 33 | 1864.1 |
| svm-search-040 | hog_lbp; LinearSVC; linear; C=10.0; weight=None | 0.1314 | 0.0302 | 21 / 570 / 31 | 991.3 |
| svm-search-041 | hog_lbp; LinearSVC; linear; C=10.0; weight=balanced | 0.1309 | 0.0296 | 20 / 536 / 32 | 940.9 |
| svm-search-042 | hog_lbp; SVC; rbf; C=0.1; weight=None | 0.1646 | 0.0965 | 8 / 27 / 44 | 169.3 |
| svm-search-043 | hog_lbp; SVC; rbf; C=0.1; weight=balanced | 0.1079 | 0.0554 | 9 / 200 / 43 | 493.4 |
| svm-search-044 | hog_lbp; SVC; rbf; C=1.0; weight=None | 0.1441 | 0.0962 | 10 / 129 / 42 | 179.2 |
| svm-search-045 | hog_lbp; SVC; rbf; C=1.0; weight=balanced | 0.1118 | 0.0640 | 7 / 98 / 45 | 337.4 |
| svm-search-046 | hog_lbp; SVC; rbf; C=10.0; weight=None | 0.1440 | 0.0825 | 7 / 28 / 45 | 200.8 |
| svm-search-047 | hog_lbp; SVC; rbf; C=10.0; weight=balanced | 0.1259 | 0.0775 | 7 / 63 / 45 | 208.1 |

## Recorded lessons

### ann-search-000

Validation F2=0.0923, AP=0.0188, TP/FP/FN=41/1973/11. Difference from initial ANN F2=+0.0000; best epoch 1 of 4. Recorded mean training loss 1.2777 to 1.2564. Assess learning rate and convergence within the recorded budget; a short-run difference does not establish eventual learnability. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-001

Validation F2=0.0804, AP=0.0140, TP/FP/FN=48/2729/4. Difference from initial ANN F2=-0.0119; best epoch 3 of 4. Recorded mean training loss 1.3150 to 1.2657. Assess learning rate and convergence within the recorded budget; a short-run difference does not establish eventual learnability. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-002

Validation F2=0.0941, AP=0.0165, TP/FP/FN=17/678/35. Difference from initial ANN F2=+0.0019; best epoch 1 of 4. Recorded mean training loss 1.2825 to 1.2584. Assess learning rate and convergence within the recorded budget; a short-run difference does not establish eventual learnability. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-003

Validation F2=0.0881, AP=0.0153, TP/FP/FN=41/2077/11. Difference from initial ANN F2=-0.0041; best epoch 2 of 4. Recorded mean training loss 1.2757 to 1.2616. Assess learning rate and convergence within the recorded budget; a short-run difference does not establish eventual learnability. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-004

Validation F2=0.0895, AP=0.0170, TP/FP/FN=41/2042/11. Difference from initial ANN F2=-0.0028; best epoch 1 of 4. Recorded mean training loss 1.2772 to 1.2619. Batch size changes gradient aggregation and updates per epoch; fixed epochs are not fixed update counts. Avoid attributing the difference solely to gradient noise. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-005

Validation F2=0.0939, AP=0.0170, TP/FP/FN=15/576/37. Difference from initial ANN F2=+0.0016; best epoch 2 of 4. Recorded mean training loss 1.2779 to 1.2531. Batch size changes gradient aggregation and updates per epoch; fixed epochs are not fixed update counts. Avoid attributing the difference solely to gradient noise. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-006

Validation F2=0.0793, AP=0.0139, TP/FP/FN=38/2149/14. Difference from initial ANN F2=-0.0129; best epoch 1 of 4. Recorded mean training loss 1.2841 to 1.2807. Capacity/depth/dropout changed; use the recorded layer table and matched controls. More parameters are not evidence of better generalisation. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-007

Validation F2=0.1799, AP=0.0368, TP/FP/FN=15/194/37. Difference from initial ANN F2=+0.0876; best epoch 4 of 4. Recorded mean training loss 1.2838 to 1.2033. Capacity/depth/dropout changed; use the recorded layer table and matched controls. More parameters are not evidence of better generalisation. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-008

Validation F2=0.0823, AP=0.0180, TP/FP/FN=16/748/36. Difference from initial ANN F2=-0.0100; best epoch 4 of 4. Recorded mean training loss 1.2862 to 1.2737. Momentum SGD is judged at its own stated learning rate; compare all four SGD rates with AdamW rather than treating one unsuccessful rate as an optimiser verdict. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-009

Validation F2=0.0860, AP=0.0150, TP/FP/FN=40/2078/12. Difference from initial ANN F2=-0.0063; best epoch 2 of 4. Recorded mean training loss 1.3210 to 1.2790. Momentum SGD is judged at its own stated learning rate; compare all four SGD rates with AdamW rather than treating one unsuccessful rate as an optimiser verdict. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-010

Validation F2=0.0834, AP=0.0162, TP/FP/FN=33/1738/19. Difference from initial ANN F2=-0.0089; best epoch 3 of 4. Recorded mean training loss 1.3294 to 1.3104. Momentum SGD is judged at its own stated learning rate; compare all four SGD rates with AdamW rather than treating one unsuccessful rate as an optimiser verdict. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-011

Validation F2=0.1274, AP=0.0333, TP/FP/FN=8/98/44. Difference from initial ANN F2=+0.0351; best epoch 1 of 4. Recorded mean training loss 1.4101 to 1.3857. Momentum SGD is judged at its own stated learning rate; compare all four SGD rates with AdamW rather than treating one unsuccessful rate as an optimiser verdict. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-012

Validation F2=0.0960, AP=0.0180, TP/FP/FN=38/1733/14. Difference from initial ANN F2=+0.0037; best epoch 1 of 4. Recorded mean training loss 0.1770 to 0.1651. Unweighted BCE changes the objective and score distribution. Raw loss magnitude is not comparable with weighted BCE; inspect F2, AP and missed defects. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-013

Validation F2=0.0909, AP=0.0226, TP/FP/FN=41/2007/11. Difference from initial ANN F2=-0.0014; best epoch 1 of 4. Recorded mean training loss 1.2700 to 1.2621. Removing flips tests whether orientation variation helps the held-out acquisition; one origin cannot establish a universal augmentation policy. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-014

Validation F2=0.0997, AP=0.0188, TP/FP/FN=18/677/34. Difference from initial ANN F2=+0.0074; best epoch 2 of 4. Recorded mean training loss 1.2780 to 1.2604. Capacity/depth/dropout changed; use the recorded layer table and matched controls. More parameters are not evidence of better generalisation. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-015

Validation F2=0.1097, AP=0.0446, TP/FP/FN=16/505/36. Difference from initial ANN F2=+0.0175; best epoch 2 of 4. Recorded mean training loss 1.2645 to 1.2524. Capacity/depth/dropout changed; use the recorded layer table and matched controls. More parameters are not evidence of better generalisation. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-016

Validation F2=0.2749, AP=0.1659, TP/FP/FN=21/153/31. Difference from initial ANN F2=+0.1826; best epoch 2 of 4. Recorded mean training loss 1.0103 to 0.2121. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-017

Validation F2=0.3355, AP=0.2962, TP/FP/FN=21/84/31. Difference from initial ANN F2=+0.2432; best epoch 2 of 4. Recorded mean training loss 0.9933 to 0.5071. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-018

Validation F2=0.4856, AP=0.4991, TP/FP/FN=27/43/25. Difference from initial ANN F2=+0.3934; best epoch 2 of 4. Recorded mean training loss 0.9116 to 0.2142. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-019

Validation F2=0.4137, AP=0.3627, TP/FP/FN=23/47/29. Difference from initial ANN F2=+0.3214; best epoch 3 of 4. Recorded mean training loss 0.8838 to 0.4818. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-search-020

Validation F2=0.4856, AP=0.4991, TP/FP/FN=27/43/25. Difference from initial ANN F2=+0.3934; best epoch 2 of 5. Recorded mean training loss 0.9116 to 0.1955. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-repeat-123

Validation F2=0.3704, AP=0.3623, TP/FP/FN=18/17/34. Difference from initial ANN F2=+0.2781; best epoch 2 of 4. Recorded mean training loss 0.8574 to 0.1951. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### ann-repeat-2026

Validation F2=0.4527, AP=0.4523, TP/FP/FN=22/13/30. Difference from initial ANN F2=+0.3604; best epoch 2 of 4. Recorded mean training loss 0.8900 to 0.1990. Real ImageNet transfer is tested with all layers fine-tuned, replicated grayscale and no crop. Domain mismatch and this early-training budget limit the conclusion. Train/validation prevalence differs, so a raw loss gap alone does not diagnose overfitting.

### svm-search-000

Validation F2=0.0707, AP=0.0132, TP/FP/FN=52/3420/0. Matched counterpart svm-search-001 differs in class_weight (balanced -> None): F2 difference -0.0077, AP difference -0.0018. These descriptive differences do not establish a universal setting effect. lbp with SVC pilot, kernel=linear, C=0.1, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-001

Validation F2=0.0784, AP=0.0151, TP/FP/FN=37/2116/15. Matched counterpart svm-search-000 differs in class_weight (None -> balanced): F2 difference +0.0077, AP difference +0.0018. These descriptive differences do not establish a universal setting effect. lbp with SVC pilot, kernel=linear, C=0.1, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-002

Validation F2=0.0720, AP=0.0123, TP/FP/FN=52/3350/0. Matched counterpart svm-search-000 differs in C (0.1 -> 1.0): F2 difference +0.0014, AP difference -0.0010. These descriptive differences do not establish a universal setting effect. lbp with SVC pilot, kernel=linear, C=1.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-003

Validation F2=0.0786, AP=0.0150, TP/FP/FN=36/2047/16. Matched counterpart svm-search-001 differs in C (0.1 -> 1.0): F2 difference +0.0002, AP difference -0.0000. These descriptive differences do not establish a universal setting effect. lbp with SVC pilot, kernel=linear, C=1.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-024

Validation F2=0.0740, AP=0.0148, TP/FP/FN=38/2323/14. Matched counterpart svm-search-036 differs in features (hog_lbp -> lbp): F2 difference -0.0512, AP difference -0.0184. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=0.1, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-025

Validation F2=0.0823, AP=0.0154, TP/FP/FN=36/1943/16. Matched counterpart svm-search-024 differs in class_weight (None -> balanced): F2 difference +0.0083, AP difference +0.0006. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=0.1, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-026

Validation F2=0.0749, AP=0.0148, TP/FP/FN=39/2356/13. Matched counterpart svm-search-024 differs in C (0.1 -> 1.0): F2 difference +0.0010, AP difference -0.0000. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=1.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-027

Validation F2=0.0819, AP=0.0154, TP/FP/FN=29/1533/23. Matched counterpart svm-search-025 differs in C (0.1 -> 1.0): F2 difference -0.0004, AP difference -0.0000. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=1.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-028

Validation F2=0.0749, AP=0.0148, TP/FP/FN=39/2356/13. Matched counterpart svm-search-024 differs in C (0.1 -> 10.0): F2 difference +0.0010, AP difference -0.0000. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=10.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-029

Validation F2=0.0819, AP=0.0154, TP/FP/FN=29/1533/23. Matched counterpart svm-search-025 differs in C (0.1 -> 10.0): F2 difference -0.0004, AP difference -0.0000. These descriptive differences do not establish a universal setting effect. lbp with LinearSVC, kernel=linear, C=10.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-030

Validation F2=0.0713, AP=0.0140, TP/FP/FN=52/3385/0. Matched counterpart svm-search-042 differs in features (hog_lbp -> lbp): F2 difference -0.0933, AP difference -0.0824. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=0.1, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-031

Validation F2=0.0745, AP=0.0151, TP/FP/FN=46/2835/6. Matched counterpart svm-search-030 differs in class_weight (None -> balanced): F2 difference +0.0031, AP difference +0.0010. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=0.1, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-032

Validation F2=0.0713, AP=0.0140, TP/FP/FN=52/3385/0. Matched counterpart svm-search-030 differs in C (0.1 -> 1.0): F2 difference +0.0000, AP difference -0.0001. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=1.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-033

Validation F2=0.0707, AP=0.0238, TP/FP/FN=52/3420/0. Matched counterpart svm-search-031 differs in C (0.1 -> 1.0): F2 difference -0.0038, AP difference +0.0087. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=1.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-034

Validation F2=0.0720, AP=0.0139, TP/FP/FN=52/3350/0. Matched counterpart svm-search-030 differs in C (0.1 -> 10.0): F2 difference +0.0007, AP difference -0.0002. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=10.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-035

Validation F2=0.0707, AP=0.0135, TP/FP/FN=52/3420/0. Matched counterpart svm-search-031 differs in C (0.1 -> 10.0): F2 difference -0.0038, AP difference -0.0015. These descriptive differences do not establish a universal setting effect. lbp with SVC, kernel=rbf, C=10.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-036

Validation F2=0.1252, AP=0.0331, TP/FP/FN=20/571/32. Matched counterpart svm-search-024 differs in features (lbp -> hog_lbp): F2 difference +0.0512, AP difference +0.0184. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=0.1, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-037

Validation F2=0.1252, AP=0.0332, TP/FP/FN=20/571/32. Matched counterpart svm-search-036 differs in class_weight (None -> balanced): F2 difference +0.0000, AP difference +0.0000. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=0.1, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-038

Validation F2=0.1252, AP=0.0312, TP/FP/FN=20/571/32. Matched counterpart svm-search-036 differs in C (0.1 -> 1.0): F2 difference +0.0000, AP difference -0.0019. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=1.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-039

Validation F2=0.1303, AP=0.0297, TP/FP/FN=19/502/33. Matched counterpart svm-search-037 differs in C (0.1 -> 1.0): F2 difference +0.0052, AP difference -0.0035. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=1.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-040

Validation F2=0.1314, AP=0.0302, TP/FP/FN=21/570/31. Matched counterpart svm-search-036 differs in C (0.1 -> 10.0): F2 difference +0.0063, AP difference -0.0029. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=10.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-041

Validation F2=0.1309, AP=0.0296, TP/FP/FN=20/536/32. Matched counterpart svm-search-037 differs in C (0.1 -> 10.0): F2 difference +0.0057, AP difference -0.0036. These descriptive differences do not establish a universal setting effect. hog_lbp with LinearSVC, kernel=linear, C=10.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-042

Validation F2=0.1646, AP=0.0965, TP/FP/FN=8/27/44. Matched counterpart svm-search-030 differs in features (lbp -> hog_lbp): F2 difference +0.0933, AP difference +0.0824. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=0.1, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-043

Validation F2=0.1079, AP=0.0554, TP/FP/FN=9/200/43. Matched counterpart svm-search-042 differs in class_weight (None -> balanced): F2 difference -0.0567, AP difference -0.0410. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=0.1, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-044

Validation F2=0.1441, AP=0.0962, TP/FP/FN=10/129/42. Matched counterpart svm-search-042 differs in C (0.1 -> 1.0): F2 difference -0.0205, AP difference -0.0002. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=1.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-045

Validation F2=0.1118, AP=0.0640, TP/FP/FN=7/98/45. Matched counterpart svm-search-043 differs in C (0.1 -> 1.0): F2 difference +0.0039, AP difference +0.0086. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=1.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-046

Validation F2=0.1440, AP=0.0825, TP/FP/FN=7/28/45. Matched counterpart svm-search-042 differs in C (0.1 -> 10.0): F2 difference -0.0206, AP difference -0.0140. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=10.0, class_weight=None. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

### svm-search-047

Validation F2=0.1259, AP=0.0775, TP/FP/FN=7/63/45. Matched counterpart svm-search-043 differs in C (0.1 -> 10.0): F2 difference +0.0180, AP difference +0.0221. These descriptive differences do not establish a universal setting effect. hog_lbp with SVC, kernel=rbf, C=10.0, class_weight=balanced. Descriptor extraction and scaling use train only; the score cutoff uses validation. Interpret kernel/weighting with matched feature/C controls. Only one validation origin is represented, so these are descriptive selection results.

## Actual VLM validation

| Prompt | Examples | F2 | Unresolved | Mean total latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3472 | 0.0707 | 74 | 675.0 |
| 1 | 3472 | 0.0707 | 23 | 735.8 |

Raw VLM decisions, exact prompts, processor/model revision, attempts and token counts are retained in JSON and JSONL. All completed prompts are scored on every validation example. Partial execution logs never select the prompt.

## Engineering history

`docs/engineering-runs/` preserves actual CPU profiling, solver amendments, interrupted attempts and control-resolution notes. `development-run/` preserves the earlier artificial sample checks, explicitly excluded from assessed results.
