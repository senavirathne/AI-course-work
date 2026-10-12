# Pre-evaluation model-family hypothesis

Recorded before any approved-dataset final test evaluation. **Expectation: the fine-tuned compact CNN may exceed the texture SVM on defect-class F2, provided the verified independent-source coverage is sufficient.** Its local convolutional filters can combine texture disruptions with broader shape cues that fixed LBP/HOG descriptors cannot adapt to. Transfer learning may reduce the data required to learn useful visual features.

This is a conditional prior, not a result. Small defects and regular periodic weave may favour local LBP/HOG features. Limited independent sources, pretrained colour/natural-image domain mismatch, global-average dilution of small defects, or biased source coverage could allow the classical model to win. The initial CNN is particularly vulnerable to limited-data learning. Classical victory would be reported without altering the split or weakening its tuning budget.

The prototype's synthetic scores are not used to predict real factory performance. No real-data final test score was accessed when this hypothesis was written.
