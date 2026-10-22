# Question 2 learning material: solution design

This guide explains how the submitted project answers Question 2 of the coursework brief and why each part is required. It is a study aid for understanding and defending the work. The assessed Q2 answer itself is on pages 5–8 of `reports/final_report.pdf`.

## What Question 2 requires

Question 2 is worth 20 marks and has two equally weighted parts.

### Q2.a: design and architect a suitable ANN — 10 marks

The answer must:

1. present the architecture, including layers, dimensions, activations, parameter count, and a clear architecture diagram;
2. explain the architecture's assumptions about the data and why they suit the problem;
3. justify the major decisions individually, including depth, width, normalisation, regularisation, output layer, and loss; and
4. ensure that the loss and output formulation match the task and the decision consequences identified in Question 1.

### Q2.b: discuss alternative approaches — 10 marks

The answer must:

1. discuss meaningful alternatives and explain why the selected approach was chosen;
2. justify using an ANN from the modality, volume, structure, and task rather than merely asserting that an ANN is appropriate;
3. state, before final evaluation, whether the ANN or classical family is expected to perform better on the primary metric and explain why; and
4. explain when each important rejected alternative would have been appropriate and which property of this project makes it less suitable.

The brief does not prescribe specific filenames. The report provides the directly assessed argument, while the code, architecture tables, experiment records, and frozen hypothesis make that argument verifiable.

## How Question 1 constrains Question 2

Q2 must respond to evidence established in Q1 rather than introduce an arbitrary architecture.

| Q1 finding | Required Q2 consequence |
|---|---|
| Only 539 of 13,384 training patches are defective. | Use a binary formulation, compare train-derived class weighting, evaluate with defect-class F2, and avoid treating accuracy as the design objective. |
| Defects can occupy only one pixel of a 256 × 256 patch. | Preserve input geometry, avoid destructive resizing and cropping, use local spatial features, and disclose the risk from internal stride and global pooling. |
| Training has 13,384 patches but only four independent publisher-origin components. | Control capacity, compare a small scratch model with transfer learning, regularise, and evaluate against a strong classical texture baseline. |

This connection is mandatory because Q2 design choices should follow from the data and user decision established in Q1.

## Q2.a requirement 1: present a complete architecture

The project presents both the initial ANN and the validation-selected final ANN. This is useful because the initial model shows the deliberate starting design, while the final model records what was selected after controlled train/validation experiments.

### Initial ANN: `ann-search-000`

The initial model is a compact custom convolutional neural network with 5,977 total and trainable parameters.

```text
Input: 1 × 256 × 256 grayscale
    ↓
3×3 Conv 1→8, same padding → BatchNorm → ReLU → 2×2 MaxPool
Output: 8 × 128 × 128
    ↓
3×3 Conv 8→16, same padding → BatchNorm → ReLU → 2×2 MaxPool
Output: 16 × 64 × 64
    ↓
3×3 Conv 16→32, same padding → BatchNorm → ReLU → 2×2 MaxPool
Output: 32 × 32 × 32
    ↓
Adaptive global average pooling → 32 × 1 × 1
    ↓
Flatten → Dropout(0.2) → Linear 32→1
    ↓
One unbounded defect logit
```

| Stage | Trainable parameters |
|---|---:|
| Block 1 | 88 |
| Block 2 | 1,184 |
| Block 3 | 4,672 |
| Global average pooling | 0 |
| Binary classifier | 33 |
| Total | 5,977 |

The count is not estimated from a diagram. Forward hooks record the actual leaf-layer shapes and parameter counts in `reports/ann-search-000-architecture.csv`, and the same information is embedded in `experiments/ann-search-000.json`.

### Selected final ANN: `ann-search-018`

The final model is a genuine ImageNet-pretrained MobileNetV3-Small whose classifier was replaced with one binary output. Every layer is fine-tuned. It has 1,518,881 total and trainable parameters.

```text
Input grayscale patch: 256 × 256
    ↓ replicate grayscale channel three times
Input tensor: 3 × 256 × 256
    ↓ ImageNet channel normalisation
Conv stem → BatchNorm → Hardswish
Output: 16 × 128 × 128
    ↓
11 MobileNetV3 inverted-residual blocks
ReLU/Hardswish, depthwise and pointwise convolutions,
squeeze-excitation gates, and matching-shape residual links
Spatial progression: 128 → 64 → 32 → 16 → 8
Channel progression ends at 96 × 8 × 8
    ↓
Final 1×1 Conv → BatchNorm → Hardswish
Output: 576 × 8 × 8
    ↓
Global average pooling → 576
    ↓
Linear 576→1024 → Hardswish → Dropout(0.2) → Linear 1024→1
    ↓
One unbounded defect logit
```

The report groups the network into readable stages. The complete 141-leaf-module record is stored in `reports/ann-search-018-architecture.csv`, and its parameter sum is exactly 1,518,881.

The selected configuration is:

- model: pretrained MobileNetV3-Small;
- optimiser: AdamW;
- learning rate: `0.0003`;
- batch size: `32`;
- augmentation: disabled for this candidate;
- weighted binary cross-entropy: enabled;
- seed: `42`;
- maximum screening epochs: `4`; and
- validation-selected checkpoint: epoch `2`.

Selection used validation F2, then validation average precision, then smaller parameter count across 21 ANN candidates. Test results were not used to choose the model.

### Evidence files and why they matter

| Evidence | Role in satisfying Q2.a |
|---|---|
| `src/fabric_inspector/models/neural.py` | Defines the custom CNN and MobileNetV3 binary model and identifies the Grad-CAM layer. |
| `src/fabric_inspector/training/runner.py` | Creates the model, initialises pretrained weights, trains all parameters, records actual architecture shapes, applies the loss, and selects the best epoch using validation data. |
| `reports/ann-search-000-architecture.csv` | Verifies every initial-model leaf layer, tensor shape, and parameter count. |
| `reports/ann-search-018-architecture.csv` | Verifies every selected-model leaf layer, tensor shape, and parameter count. |
| `experiments/ann-search-000.json` | Preserves the initial architecture, configuration, hypothesis, learning history, threshold, and checkpoint hash. |
| `experiments/ann-search-018.json` | Preserves the selected architecture, genuine pretraining flag, configuration, history, validation metrics, and checkpoint hash. |
| `scripts/select_models.py` | Implements the declared validation-only model-selection rule. |
| `reports/final_report.pdf`, pages 5–7 | Provides the directly assessed diagrams, tables, justifications, and limitations. |

## Q2.a requirement 2: explain the architecture's assumptions

An architecture is not suitable merely because it is popular. Its operations encode assumptions about the input.

### Spatial locality

Convolution assumes that useful evidence occurs in local neighbourhoods. This is suitable because fabric faults appear as local disruptions, holes, stains, or irregular yarn patterns rather than arbitrary relationships between distant flattened pixels.

### Shared features across position

The same convolutional filters are applied across the patch. This assumes a useful local defect cue can matter at different locations. That is appropriate for patch classification because the defect is not restricted to a fixed coordinate.

### Hierarchical composition

Successive layers combine edges and texture changes into broader patterns. This is suitable for repeated weave structure, where a defect may be defined by how local texture differs from its surroundings.

### Some translation tolerance

Stride, pooling, and global pooling reduce sensitivity to exact location. This helps patch-level classification but creates a limitation: very small defects may be diluted as spatial maps become coarser.

### Transferability of visual features

The final model assumes ImageNet features provide a useful starting point despite the domain difference. This may help when independent industrial origins are limited, but natural-image colour pretraining does not perfectly match grayscale fabric. Replicating grayscale into three channels satisfies the pretrained input contract but adds no colour information.

### Limited independent diversity

The design assumes capacity must be controlled because thousands of patches do not represent thousands of independent acquisitions. MobileNetV3 is compact relative to many pretrained CNNs, while the custom starting model is deliberately very small.

These assumptions and their limitations must both be stated. Suitability is a reasoned hypothesis to test, not proof that the ANN will outperform the baseline.

## Q2.a requirement 3: justify each major design decision

### Depth

- The initial three-block depth provides a hierarchy of local receptive fields without starting with a large network.
- Two- and four-block custom variants were tested, so depth was an explicit design factor rather than an unexplained constant.
- The final MobileNet depth is inherited from MobileNetV3-Small and was selected as a whole transfer candidate; its individual block count was not claimed to be independently optimal.

### Width

- Initial channels progress from 8 to 16 to 32, allowing deeper blocks to represent more feature combinations.
- The small starting width responds to limited independent-origin coverage.
- Width variants were tested, but the final MobileNet widths are inherited from the pretrained architecture rather than separately optimised for RAW-FABRID.

### Normalisation

- BatchNorm follows convolutional operations, stabilising activation scale during optimisation.
- Its learned scale and offset are trainable; running statistics come from training data and are fixed in evaluation mode.
- BatchNorm can still fail under unseen-origin or illumination shift, so it is not a guarantee of domain robustness.
- The custom CNN uses `(gray/255 - 0.5) / 0.5`.
- MobileNet replicates grayscale into three channels and uses the pretrained ImageNet means and standard deviations. No resize or centre crop is applied.

### Activations

- ReLU supplies nonlinearity in the custom model and early MobileNet blocks.
- Hardswish is part of MobileNetV3 and provides a computationally efficient smooth gating behaviour.
- Squeeze-excitation blocks use ReLU and Hardsigmoid to adjust channel importance from pooled context.

### Spatial reduction and pooling

- Max pooling in the custom model reduces computation and adds some location tolerance.
- MobileNet uses progressive stride and efficient depthwise convolutions.
- Global average pooling avoids a large flattened classifier and reduces parameter count.
- These choices can dilute tiny defects. The final MobileNet feature map is 8 × 8, so the limitation is explicitly carried into Q5 explanation validation.

### Regularisation

- Dropout `0.2` is applied in the classifier during training and disabled during evaluation.
- Weight decay is applied by AdamW.
- Limited training-only flips were candidates rather than assumptions that augmentation must help. The selected final candidate has augmentation disabled.
- Early stopping retains the best validation epoch rather than automatically using the last epoch.

### Output layer

One linear output produces one unbounded logit because the task is binary patch classification. A sigmoid converts the logit into a score for thresholding and display. The score is explicitly described as uncalibrated, not guaranteed confidence.

### Loss function

`BCEWithLogitsLoss` combines the sigmoid-related binary objective with a numerically stable logit formulation. The weighted candidate uses the training-only positive weight:

`(13,384 - 539) / 539 = 23.8312`

Weighted and unweighted loss were compared. The class weight compensates for imbalance during fitting; it is not a measured factory-cost ratio and does not calibrate the output score.

## Q2.a requirement 4: connect output and loss to Q1 consequences

This is mandatory because model mathematics must match the target and the user's decision.

- The target has two states—normal and defective—so one binary logit is appropriate.
- Q1 establishes that missed defects are more consequential than unnecessary reviews, so positive-class F2 is used for validation model and threshold selection.
- Class weighting gives defective training examples more influence, but it is treated as a candidate because training imbalance and operational cost are different concepts.
- The validation-selected classification threshold is separate from the loss. Training learns a score; thresholding turns that score into a decision.
- The final threshold is approximately `0.895234`, selected on validation F2. It is not fixed at 0.5 and is not selected on test data.
- The binary classification cutoff and three-band inspection recommendation thresholds serve different objectives and must not be conflated.

## Q2.b requirement 1: discuss meaningful alternatives

Each alternative must be discussed using two questions: when would it be appropriate, and why is it retained or rejected here?

| Alternative | When it would be appropriate | Decision for this project |
|---|---|---|
| HOG/LBP with SVM | Limited labelled data and texture-dominated images | Retained and tuned as the mandatory classical baseline. Fixed local texture and directional features may generalise well with few independent origins. |
| Small scratch CNN | Task-specific local cues with enough representative labelled coverage | Retained as the initial ANN and used for interpretable depth, width, optimiser, and regularisation experiments. It may struggle with only four training origins. |
| Pretrained compact CNN | Limited labels but transferable visual features | Retained and selected after validation. Transfer can improve data efficiency, but grayscale domain mismatch and coarse spatial maps remain limitations. |
| Vision transformer | Large, diverse training data or suitable domain-specific pretraining and sufficient compute | Rejected as a primary candidate because data independence, compute budget, and tiny-local-detail risks are unfavourable here. |
| Normal-only anomaly detection | Representative normal production data and rare or previously unseen defects | Meaningful because the publisher provides a normal-only training folder, but it would change the approved supervised task, output semantics, split, and comparison protocol. It is discussed rather than silently mixed into the supervised benchmark. |
| Supervised segmentation | Reliable dense annotations and a need for validated defect boundaries | A reasonable future study, but the approved task is binary patch classification. Grad-CAM is an explanation method and must not be presented as segmentation. |
| RNN, LSTM, or GRU | Genuine temporal or ordered sensor sequences | Rejected because pixels in a fabric patch form a two-dimensional spatial structure, not a production-time sequence. |

The rejected approaches are not described as universally inferior. Each becomes appropriate under different data, task, compute, or output requirements.

Evidence:

- `src/fabric_inspector/models/classical.py` implements the HOG/LBP SVM family.
- `src/fabric_inspector/models/neural.py` implements both retained ANN families.
- `reports/final_report.pdf`, page 8, provides the directly assessed alternatives table.
- `docs/pre-evaluation-hypothesis.md` records the model-family expectation before final test evaluation.

## Q2.b requirement 2: justify why an ANN is appropriate

The justification must use all four requested dimensions.

### Modality

The inputs are unstructured two-dimensional images. Convolution preserves spatial neighbourhoods and is more appropriate than flattening raw pixels into an ordinary tabular model.

### Volume

There are 13,384 training patches, enough to fit compact neural candidates, but only four independent publisher-origin components. This supports compact architectures and transfer learning rather than a very large model trained from scratch.

### Structure

Fabric contains repeated local textures, edges, yarn directions, and spatial disruptions. CNN filters and hierarchical receptive fields match these properties. The classical HOG/LBP baseline remains important because the same regular texture may also suit engineered descriptors.

### Task

The approved task is binary patch classification with request-time prediction and Grad-CAM explanation. A convolutional classifier supplies both a defect score and spatial feature maps suitable for the selected architecture-specific explanation method.

An ANN is therefore technically defensible, but modality alone does not prove that it will achieve higher F2. The fair baseline comparison is essential.

## Q2.b requirement 3: state a pre-evaluation expectation

Before final test evaluation, `docs/pre-evaluation-hypothesis.md` recorded this conditional expectation:

> The fine-tuned compact CNN may exceed the texture SVM on defect-class F2 if independent-source coverage is sufficient.

The reasoning was:

- learned convolutional filters can adapt to texture disruptions and shape cues that fixed HOG/LBP descriptors cannot;
- ImageNet transfer may reduce the amount of task-specific data needed; but
- tiny-defect dilution, regular weave, grayscale/natural-image mismatch, and sparse origins could allow the classical model to win.

This wording is intentionally conditional. The coursework awards marks for defensible prior reasoning, not for guessing the winner correctly. The final result must not be used to rewrite the prior expectation.

## Q2.b requirement 4: explain rejection conditions honestly

A strong alternatives section avoids statements such as “transformers are bad” or “anomaly detection is less accurate.” Those claims were not established. Instead, it identifies the missing condition:

- a transformer becomes more attractive with larger independent data, suitable pretraining, and more compute;
- anomaly detection becomes more attractive when only normal training data are trusted and unseen defects are the deployment concern;
- segmentation becomes more attractive when exact boundaries are required and dense labels are reliable;
- sequence models become more attractive when the observations have genuine temporal order; and
- full-image detection or tiling becomes more attractive when the application and evaluation labels operate at full-image level rather than patch level.

This conditional reasoning is mandatory because it demonstrates model-selection judgement rather than architecture preference.

## Technical accuracy validation

The principal Q2 claims were checked directly against the repository:

| Claim | Validation result |
|---|---|
| Initial model has 5,977 parameters | Confirmed by experiment JSON and the sum of 16 architecture-table rows. |
| Final model has 1,518,881 parameters | Confirmed by experiment JSON and the sum of 141 architecture-table rows. |
| All final-model parameters are trainable | Confirmed: total and trainable counts are both 1,518,881. |
| Final model uses genuine pretrained initialisation | Confirmed by `pretrained: true` and the MobileNetV3 weights-loading implementation. |
| Final configuration is AdamW, lr 0.0003, batch 32, no augmentation, weighted loss | Confirmed in `experiments/ann-search-018.json`. |
| Selected checkpoint is epoch 2 of 4 | Confirmed in the experiment record. |
| Selection uses validation F2, then AP, then fewer parameters | Confirmed in `scripts/select_models.py` and the serving bundle. |
| Final validation F2 is 0.4856 and AP is 0.4991 | Confirmed in the experiment record and serving bundle. |
| Positive weight is 23.8312 and is derived from training only | Confirmed in the training code and both retained ANN records. |
| No resize or centre crop is used for MobileNet input | Confirmed in `artifacts/serving-bundle.json`. |
| Anomaly detection is discussed in Q2.b | Confirmed on report page 8. |
| The ANN-versus-classical expectation predates final test evaluation | Confirmed by the explicit frozen statement in `docs/pre-evaluation-hypothesis.md`. |

## Important wording corrections

Use these precise descriptions when presenting Q2:

- Say “fine-tuned pretrained MobileNetV3-Small,” not merely “a CNN” or “a model trained from scratch.”
- Say grayscale is replicated into three channels; do not imply that the data contain colour information.
- Say the 256 × 256 input geometry is preserved; do not claim the network performs no internal downsampling.
- Say all layers are fine-tuned; do not describe the MobileNet backbone as frozen.
- Say the positive class weight addresses training imbalance; do not describe `23.8312` as the business cost of a missed defect.
- Say sigmoid produces an uncalibrated score; do not call it guaranteed confidence.
- Say validation F2 selected the model, epoch, and threshold; do not imply the test set selected them.
- Say the MobileNet block widths and strides are inherited and empirically selected as a whole; do not claim every internal dimension was independently optimised.
- Say Grad-CAM uses spatial features from the classifier; do not call it a segmentation model.
- Say normal-only anomaly detection is a meaningful alternative that changes the task and protocol; do not dismiss it as technically invalid.

## End-to-end Q2 evidence flow

1. Q1 establishes imbalance, tiny defects, and limited independent origins.
2. `src/fabric_inspector/models/neural.py` defines the compact scratch CNN and MobileNetV3 transfer candidate.
3. `src/fabric_inspector/training/runner.py` trains candidates using train data and evaluates candidate epochs using validation only.
4. Each `experiments/ann-search-*.json` record preserves the hypothesis, configuration, actual architecture, learning history, and validation outcome.
5. The architecture CSV files provide exact layer-level shapes and parameter counts.
6. `scripts/select_models.py` applies the declared validation-only selection rule and freezes the serving configuration.
7. `docs/pre-evaluation-hypothesis.md` preserves the ANN-versus-classical expectation before final test evaluation.
8. `notebooks/02_ann_and_optimisation.ipynb` provides the executed walkthrough and supporting experimental evidence.
9. `reports/final_report.pdf`, pages 5–8, presents the concise assessed Q2 answer.

## Viva self-check

You should be able to answer these questions without reading the code:

1. Why is convolution appropriate for fabric patches?
2. What assumptions do shared convolutional filters make?
3. Why was the initial CNN deliberately small?
4. How do the initial model's shapes change through its three blocks?
5. Why was MobileNetV3-Small a reasonable transfer candidate?
6. What does replicating grayscale into three channels achieve, and what does it not achieve?
7. Why can stride and global average pooling be harmful for one-pixel defects?
8. Why is one binary logit preferable to a two-output softmax for this implementation?
9. What is the difference between the training loss, sigmoid score, and decision threshold?
10. Why is the positive class weight not the same as a business-cost ratio?
11. Why was the final threshold not fixed at 0.5?
12. Which design choices were inherited from MobileNetV3 rather than independently optimised?
13. When would normal-only anomaly detection be preferable?
14. Why is segmentation meaningful but outside the approved patch-classification task?
15. Why must a classical texture baseline still be tested when a CNN is structurally appropriate?
16. What model family was expected to win before final evaluation, and why was that expectation conditional?
