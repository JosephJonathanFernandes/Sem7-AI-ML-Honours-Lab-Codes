# Viva Q&A — Detailed (Questions + Answers)

This file contains detailed question-and-answer entries for viva preparation. Each entry includes: question, type (short / derivation / coding / practical), and a thorough answer with intuition and follow-ups.

1) Q: What is the difference between supervised and unsupervised learning? (Type: short)

Answer:
- Supervised learning uses labelled data (input x paired with target y). The goal is to learn a mapping f: x -> y to predict labels for new inputs. Examples: classification (CIFAR-10), regression (house prices).
- Unsupervised learning uses unlabelled data and seeks structure: clustering, density estimation, dimensionality reduction (PCA, t-SNE). Example: K-means clusters images by visual similarity without labels.
- Intuition/follow-up: Supervised methods provide direct feedback via a loss computed against labels, while unsupervised methods rely on intrinsic data properties.

2) Q: Define bias and variance. How do they affect model performance? (Type: short)

Answer:
- Bias: error from erroneous assumptions in the learning algorithm (underfitting). High-bias models are too simple to capture true patterns.
- Variance: error from sensitivity to small fluctuations in training data (overfitting). High-variance models fit noise in training data.
- Total expected test error = irreducible noise + bias^2 + variance. The trade-off: increasing model complexity typically reduces bias but increases variance.

3) Q: Derive the gradient of the cross-entropy loss with softmax outputs. (Type: derivation)

Answer:
- Setup: For a single input, let z be logits (vector), softmax outputs: s_i = exp(z_i)/sum_j exp(z_j). For a one-hot target y (y_k=1), cross-entropy loss L = -sum_i y_i log s_i = -log s_k.
- Compute dL/dz: using chain rule, dL/dz_i = s_i - y_i. Derivation sketch: dL/dz_i = -1/s_k * ds_k/dz_i. Using softmax derivative ds_k/dz_i = s_k(1 - s_k) if i=k else -s_k s_i. Plug and simplify to get s_i - y_i.
- Intuition: gradient is prediction minus target; optimization pushes logits to reduce difference.

4) Q: Explain backpropagation at a high level for a two-layer network. (Type: derivation)

Answer:
- Consider network: h = f(W1 x + b1), yhat = g(W2 h + b2), loss L(yhat, y).
- Forward pass computes intermediate activations. Backprop: compute dL/dyhat, then propagate to dL/dW2 = dL/dyhat * dyhat/dW2. Next compute dL/dh = (W2^T)(dL/dyhat) * g'..., then dL/dW1 = dL/dh * dh/dW1. Use chain rule repeatedly.
- Implementation note: frameworks compute and accumulate gradients automatically using autograd.

5) Q: Why do we use ReLU instead of sigmoid in deep networks? (Type: short)

Answer:
- ReLU (rectified linear unit) is simple: max(0, x). Advantages: sparse activation, reduced vanishing gradient (derivative is 1 for positive inputs), faster convergence in practice. Sigmoid saturates to 0/1 for large magnitude inputs causing gradients near zero, slowing learning in deep nets.

6) Q: What is batch normalization and why is it used? (Type: short)

Answer:
- BatchNorm normalizes layer inputs using mini-batch statistics (mean and variance), then scales and shifts with learned parameters. It reduces internal covariate shift, stabilizes and often speeds training, allows higher learning rates, and acts as a regularizer.

7) Q: How do you save and load a PyTorch model properly? (Type: coding)

Answer:
- Recommended: save `state_dict` rather than the full model object to avoid path/dependency issues.
- Save: torch.save(model.state_dict(), 'model.pth')
- Load: model = MyModel(...); model.load_state_dict(torch.load('model.pth')); model.eval() if inference.
- If saving optimizer state (to resume training): also save optimizer.state_dict() and the epoch number.

8) Q: In `train_cifar10.py`, how would you add early stopping? (Type: practical)

Answer:
- Basic early stopping logic: track validation loss each epoch; if validation loss hasn't improved for `patience` epochs, stop training. Optionally restore best model.
- Pseudocode:
  - best_loss = inf; epochs_since_improve = 0
  - for epoch in range(max_epochs):
    - train(); val_loss = validate()
    - if val_loss < best_loss: best_loss = val_loss; save_checkpoint(); epochs_since_improve = 0
    - else: epochs_since_improve += 1
    - if epochs_since_improve >= patience: break

9) Q: How do learning rate schedulers help? Give an example. (Type: short)

Answer:
- LR schedulers change the learning rate during training to improve convergence. Examples: StepLR (reduce every N epochs), ReduceLROnPlateau (reduce when metric plateaus), CosineAnnealingLR (cosine schedule). They help escape bad minima and allow large steps early then fine-tuning later.

10) Q: If training loss decreases but validation loss increases, what might be happening and how would you fix it? (Type: practical)

Answer:
- Likely overfitting: model is fitting training data noise.
- Fixes: increase regularization (weight decay, dropout), use data augmentation, reduce model capacity, get more data, use early stopping, or adjust learning rate.

11) Q: How do you set random seeds in PyTorch for reproducibility? (Type: coding)

Answer:
- Set seeds for Python, NumPy, and PyTorch:
  - import random, numpy as np, torch
  - random.seed(s); np.random.seed(s); torch.manual_seed(s)
  - if using CUDA: torch.cuda.manual_seed_all(s)
  - Optionally set torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False — note this may slow training and not guarantee full reproducibility across different hardware/drivers.

12) Q: Explain the role of the DataLoader and Dataset in PyTorch. (Type: short)

Answer:
- Dataset provides access to individual data samples and labels (implements __len__ and __getitem__). DataLoader wraps a Dataset and provides batching, shuffling, and parallel data loading with multiple workers.

13) Q: How would you implement transfer learning using a pretrained ResNet for CIFAR-10? (Type: coding)

Answer:
- Steps:
  - from torchvision import models
  - model = models.resnet18(pretrained=True)
  - Replace final fully-connected layer: model.fc = nn.Linear(model.fc.in_features, 10)
  - Optionally freeze early layers by setting param.requires_grad = False for those parameters and train only the final layers with a smaller lr.

14) Q: What is weight decay and how does it differ from L2 regularization? (Type: short)

Answer:
- Weight decay and L2 regularization are closely related: weight decay directly subtracts a scaled parameter value from gradients during update (w <- w - lr * (dL/dw + lambda * w)), while L2 regularization adds lambda/2 * ||w||^2 to the loss. In many optimizers they produce equivalent updates.

15) Q: What is the confusion matrix and how do you interpret it for CIFAR-10? (Type: short)

Answer:
- A confusion matrix is a KxK table where rows are true classes and columns are predicted classes (or vice versa). Each cell (i,j) counts examples of true class i predicted as j. For CIFAR-10, it helps identify which classes are commonly confused (e.g., cat vs dog) and guides targeted augmentation or class-specific tweaks.

16) Q: How does dropout behave differently during training and evaluation? (Type: short)

Answer:
- During training, dropout randomly zeros activations with probability p to prevent co-adaptation. During evaluation, dropout is disabled (model.eval()), and activations are scaled or left as-is depending on implementation; PyTorch's dropout scales during training so no scaling is needed at eval.

17) Q: Explain the vanishing gradient problem and one remedy. (Type: short)

Answer:
- Vanishing gradients occur when gradients shrink exponentially through layers (common with sigmoid/tanh activations), preventing early layers from learning. Remedies: use ReLU activations, proper weight initialization (He/Xavier), batch normalization, or residual connections.

18) Q: If model's accuracy is stuck at chance level, what debugging steps would you take? (Type: practical)

Answer:
- Quick checklist:
  - Check data and labels: ensure labels align with inputs and loading code is correct.
  - Reduce model to a tiny overfitting test: can it overfit 10 samples? If not, there's a bug.
  - Check learning rate and optimizer: lr too low or incorrect optimizer state.
  - Ensure gradients are non-zero (inspect parameter.grad after backprop).
  - Verify loss function and final activation (e.g., using CrossEntropyLoss expects raw logits, not softmax outputs).

19) Q: Show how you would compute accuracy and per-class precision for CIFAR-10. (Type: coding)

Answer:
- Accuracy: (preds == targets).sum() / total
- Per-class precision: for each class c, precision_c = TP_c / (TP_c + FP_c). Use confusion matrix counts to compute TP and FP per class.

20) Q: What is the role of optimizer.zero_grad() in PyTorch? (Type: short)

Answer:
- PyTorch accumulates gradients by default; optimizer.zero_grad() clears previous gradients before computing new ones each iteration. If you forget it, gradients accumulate and updates will be wrong.

21) Q: Explain ReduceLROnPlateau and when to use it. (Type: short)

Answer:
- ReduceLROnPlateau reduces learning rate when a monitored metric (often validation loss) stops improving for a specified patience. Use when training plateaus; it allows finer convergence without manual lr scheduling.

22) Q: How do you resume training from a checkpoint (model + optimizer) in PyTorch? (Type: coding)

Answer:
- Save: torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': optimizer.state_dict()}, path)
- Load:
  - checkpoint = torch.load(path)
  - model.load_state_dict(checkpoint['model_state'])
  - optimizer.load_state_dict(checkpoint['opt_state'])
  - start_epoch = checkpoint['epoch'] + 1

23) Q: What is gradient clipping and why use it? (Type: short)

Answer:
- Gradient clipping constrains gradient norms (or values) to prevent exploding gradients (common in RNNs or when large losses occur). Use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) before optimizer.step().

24) Q: Describe a minimal one-epoch smoke test to verify training code works. (Type: practical)

Answer:
- Use a tiny synthetic dataset: create random tensors for inputs and labels (e.g., 32 samples, 32x32x3 images), run the training loop for 1 epoch with a small model, verify loss decreases and no exceptions thrown. This confirms shapes, device usage, and basic backward pass.

25) Q: How would you explain the CIFAR-10 pipeline in `train_cifar10.py` to an examiner? (Type: practical)

Answer:
- Describe data loading (Dataset and DataLoader), transformations/augmentations applied, model architecture (layers and final output), loss function used, optimizer choice, scheduler (if any), training loop steps (forward, loss, backward, optimizer.step, zero_grad), validation and checkpointing, and final evaluation metrics and plots (accuracy, training curves).

---


