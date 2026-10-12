# Teaching notebook inspiration and changes

The six supplied notebooks were inspected. They inform coding style and student-facing explanations; their tutorial datasets are not used as assessed fabric data.

| Supplied notebook | Useful inspiration | Adaptation for this task |
| --- | --- | --- |
| `02_fully_connected_mnist.ipynb` | `nn.Module`, device handling, loss/optimiser loop and learning plots | Spatial CNN rather than flattening the fabric image; one logit and BCE |
| `03_optimize_nn_exercises.ipynb` | Learning rate, optimisers, depth, regularisation and evaluation exercises | Controlled empirical search, logged changes and matched budgets |
| `04_cnn_concepts.ipynb` | Convolution, feature maps, padding, stride, pooling, dropout and BatchNorm | Actual fabric CNN layer/tensor table and small-defect resolution reasoning |
| `04_cnn_mnist.ipynb` | CNN training, incorrect predictions and confusion matrix | Validation-only development, preserved 256x256 inputs, XAI on genuine errors |
| `04_cnn_cifar_10.ipynb` | Training augmentation, custom CNN and train/evaluation separation | No aggressive 32x32 resize; no epoch-by-epoch test tuning |
| `05_rnn_concepts.ipynb` | RNN/LSTM/GRU and sequential-state intuition | Explain why independent still patches do not justify a sequence model |

The new notebooks call tested package functions instead of maintaining a second, diverging training implementation. They inspect actual full-data artifacts and run authentic-sample checkpoint/explanation checks. Reproduction commands are explicit; notebook review does not rerun the held-out evaluation. MNIST, Fashion-MNIST and unmodified CIFAR-10 are restricted by the coursework brief; no results on those datasets are presented as the project.
