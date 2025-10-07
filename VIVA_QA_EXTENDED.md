# VIVA Q&A Extended — 50+ Questions with Answers and Code References

This document expands the viva material to 50+ Q&A pairs. Each item shows: question, type (short / derivation / coding / practical), a concise but detailed answer, and where relevant references to functions or snippets in `expt1/train_cifar10.py` (class/function names and short excerpts). Use this for rapid viva prep and quick code lookup.

NOTE: Instead of absolute line numbers (which may change if the file is edited), I reference function and class names and include short code excerpts so you can quickly open `expt1/train_cifar10.py` and locate the relevant lines.

-----

1) Q: What does the `SimpleCNN` class implement and why are BatchNorm + ReLU + MaxPool used together? (Type: short / coding)

Answer:
- `SimpleCNN` defines a small convolutional network with three conv blocks and a small classifier. BatchNorm stabilizes learning by normalizing activations; ReLU introduces non-linearity and mitigates vanishing gradients; MaxPool reduces spatial dimensions and provides translation invariance.
- Code reference: `class SimpleCNN(nn.Module):` and the `self.features = nn.Sequential(...)` block in `expt1/train_cifar10.py`.

2) Q: In the model's classifier `nn.Linear(128 * 4 * 4, 256)`, where does `128 * 4 * 4` come from? (Type: short)

Answer:
- It's the flattened feature map size after three max-pooling operations on 32x32 input images. Each MaxPool2d(2) halves spatial dimensions: 32 -> 16 -> 8 -> 4, with 128 channels, leading to 128*4*4.

3) Q: Show how training data augmentation is configured. Why use RandomCrop + RandomHorizontalFlip? (Type: coding / practical)

Answer:
- Data augmentations are set in `get_data_loaders(...)` via `transforms.RandomHorizontalFlip()` and `transforms.RandomCrop(32, padding=4)`.
- Horizontal flip increases invariance to left-right orientation; random cropping with padding simulates small translations and increases robustness.

4) Q: What normalization values are applied to CIFAR-10 and why? (Type: short)

Answer:
- Normalize with mean (0.4914, 0.4822, 0.4465) and std (0.247, 0.243, 0.261). These are per-channel dataset statistics for CIFAR-10; normalization centers inputs and scales them to similar ranges for stable optimization.

5) Q: Where and how is the CIFAR-10 dataset loaded? (Type: coding)

Answer:
- See `trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)` inside `get_data_loaders`. The DataLoader wraps it with batching and shuffling.

6) Q: Explain the training loop structure in `main()`. Which functions are orchestrating training and evaluation? (Type: practical)

Answer:
- `main()` loops over epochs, calling `train_epoch(model, device, trainloader, criterion, optimizer)` for training and `eval_model(...)` for validation. After each epoch the scheduler is stepped and `save_checkpoint(...)` persists state.

7) Q: In `train_epoch`, what are the core steps per batch and why are they ordered that way? (Type: coding)

Answer:
- Steps: move inputs/targets to device, `optimizer.zero_grad()`, forward pass `outputs = model(inputs)`, compute `loss = criterion(outputs, targets)`, `loss.backward()`, `optimizer.step()`. Zeroing grads prevents accumulation; backward computes gradients; step updates parameters.

8) Q: How is the evaluation different from training in `eval_model`? (Type: short)

Answer:
- `eval_model` sets `model.eval()` and wraps iteration in `torch.no_grad()` to disable gradient computations and dropout, reducing memory and ensuring deterministic behavior.

9) Q: What loss function and optimizer are used, and why are they reasonable defaults? (Type: short)

Answer:
- Loss: `nn.CrossEntropyLoss()` (suitable for multi-class classification). Optimizer: `optim.Adam(model.parameters(), lr=args.lr)` — Adam is adaptive, robust to reasonable hyperparameters and works well for CNNs in many cases.

10) Q: What scheduler is used and how does it change the learning rate? (Type: short)

Answer:
- `torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)` reduces lr by factor `gamma` every `step_size` epochs (here every 10 epochs), allowing larger steps early and fine-tuning later.

11) Q: Explain the checkpointing scheme used in `save_checkpoint`. What is saved and how is the 'best' checkpoint determined? (Type: coding)

Answer:
- `save_checkpoint(state, is_best, checkpoint_dir)` saves a `checkpoint.pth` each epoch and if `is_best` copies state to `best.pth`. The saved `state` includes epoch, `state_dict`, `best_acc`, and optimizer state.

12) Q: How would you resume training from a saved checkpoint in this project? (Type: coding)

Answer:
- Load checkpoint: `ckpt = torch.load(path)`, restore `model.load_state_dict(ckpt['state_dict'])`, `optimizer.load_state_dict(ckpt['optimizer'])`, and set `start_epoch = ckpt['epoch'] + 1`. Then resume the epoch loop.

13) Q: The script calls `scheduler.step()` after validation. Is this the recommended place for StepLR? Why? (Type: short)

Answer:
- Placing `scheduler.step()` after validation (end-of-epoch) is common for StepLR since it is epoch-based. For metric-based schedulers (ReduceLROnPlateau) you would pass the validation metric to step; for epoch-based StepLR this is fine.

14) Q: Where does the script set random seeds and what additional steps are needed for full determinism? (Type: coding)

Answer:
- Seeds set in `main()` with `torch.manual_seed(args.seed)` and `torch.cuda.manual_seed(args.seed)` if CUDA. For more determinism set `random.seed`, `np.random.seed`, `torch.backends.cudnn.deterministic=True`, and `torch.backends.cudnn.benchmark=False` (but note this may slow training and doesn't guarantee bitwise reproducibility across hardware/drivers).

15) Q: How is accuracy computed in `train_epoch`/`eval_model`? (Type: short)

Answer:
- Predictions: `_, predicted = outputs.max(1)` then `predicted.eq(targets).sum().item()` to count correct predictions. Accuracy = 100.0 * correct / total.

16) Q: Explain why `optimizer.zero_grad()` is required before `loss.backward()`. (Type: short)

Answer:
- PyTorch accumulates gradients; without zeroing, gradients from previous batches would accumulate leading to incorrect updates.

17) Q: If training is slow due to data loading, where in the code would you change to improve throughput? (Type: practical)

Answer:
- Increase `num_workers` in `DataLoader` construction in `get_data_loaders(...)`, adjust `pin_memory=True` if using CUDA, and optimize transforms to avoid expensive CPU ops.

18) Q: The code uses `num_workers=2`. What are trade-offs when increasing it to, say, 8 or more? (Type: short)

Answer:
- More workers can increase data throughput but consume more CPU/memory and may cause contention. On Windows, `num_workers` > 0 spawns processes differently; monitor system load and memory.

19) Q: Where are training curves plotted and saved? (Type: coding)

Answer:
- Function `plot_curves(history, out_dir)` uses matplotlib and saves `training_curves.png` in the checkpoint directory.

20) Q: How is early stopping implemented currently? If not present, describe how to add it. (Type: practical)

Answer:
- The script does not currently implement early stopping; it saves the best model but trains for all epochs. Add a `patience` counter that tracks epochs since last improvement and break the epoch loop when patience is exceeded; optionally restore best checkpoint.

21) Q: Show a minimal code snippet to add MixUp augmentation. Where would you integrate it? (Type: coding)

Answer:
- MixUp combines pairs of images and labels; implement a helper that mixes inputs and targets and call it inside `train_epoch` before forward pass. Or implement as a Dataset wrapper. Example (conceptual): `inputs, targets = mixup(inputs, targets, alpha=0.4)`.

22) Q: Why does `plot_curves` multiply loss by batch size when accumulating `running_loss`? (Type: short)

Answer:
- It accumulates `loss.item() * inputs.size(0)` so that final loss is the average across all samples (summing per-sample losses then dividing by total), correctly handling possible unequal batch sizes.

23) Q: Suggest three small model changes to improve CIFAR-10 accuracy quickly. (Type: practical)

Answer:
- Replace classifier with a larger FC layer or add another conv block; use deeper backbone like ResNet18 from torchvision and fine-tune; add stronger augmentations (cutout, random erasing), or use learning rate warmup and cosine annealing.

24) Q: How would you incorporate transfer learning using torchvision's ResNet? Provide the key lines. (Type: coding)

Answer:
- `from torchvision import models`
- `model = models.resnet18(pretrained=True)`
- `model.fc = nn.Linear(model.fc.in_features, 10)`
- Optionally freeze early layers: `for param in model.parameters(): param.requires_grad=False` then unfreeze final layers.

25) Q: Demonstrate how to save both model and optimizer state for a clean resume. (Type: coding)

Answer:
- Save: `torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)`
- Load and restore optimizer with `optimizer.load_state_dict(checkpoint['optimizer'])`.

26) Q: What debugging steps would you take if the model's loss is NaN? (Type: practical)

Answer:
- Check for exploding gradients (inspect grads), lower the learning rate, enable gradient clipping, inspect input data for NaNs or infinities, verify loss target ranges, and ensure numerical stability in custom ops.

27) Q: Why is `torch.no_grad()` used in `eval_model`? (Type: short)

Answer:
- It disables gradient computation and reduces memory; necessary for inference/validation to avoid storing computation graph.

28) Q: Where does the script print epoch statistics and what is included? (Type: short)

Answer:
- After saving the checkpoint each epoch `print(f"Epoch {epoch}/{args.epochs} - train_loss: ... val_acc: ...")` prints train/val loss and accuracy for the epoch.

29) Q: How would you add per-class accuracy logging? (Type: coding)

Answer:
- Keep per-class counters: `class_correct = np.zeros(10)` and `class_total = np.zeros(10)`; after prediction update counters per class index using masks; compute per-class accuracy.

30) Q: What is the purpose of `model.train()` and `model.eval()`? (Type: short)

Answer:
- `model.train()` enables training behavior (dropout, batchnorm update), while `model.eval()` disables dropout and uses running stats in batchnorms for evaluation.

31) Q: Explain CrossEntropyLoss input expectations and a common pitfall when using softmax. (Type: short)

Answer:
- `nn.CrossEntropyLoss` expects raw logits (unnormalized scores). Do not apply `F.log_softmax` or `softmax` before passing to CrossEntropyLoss since it internally computes log-softmax.

32) Q: How to implement gradient clipping in this training script? Where to place it? (Type: coding)

Answer:
- After `loss.backward()` and before `optimizer.step()`, add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)`.

33) Q: How to switch optimizer from Adam to SGD with momentum and why might you do that? (Type: short)

Answer:
- Replace `optim.Adam(...)` with `optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)`. SGD with momentum often generalizes better for large-scale vision tasks; learning-rate scheduling interacts differently with SGD.

34) Q: How is the `history` object structured and how is it saved at the end? (Type: coding)

Answer:
- `history` is a dict with keys `train_loss`, `val_loss`, `train_acc`, `val_acc` (lists per epoch). It's saved with `np.savez(os.path.join(args.checkpoint_dir, 'history.npz'), **history)`.

35) Q: Where would you add TensorBoard logging? (Type: practical)

Answer:
- Create `SummaryWriter` at the start of `main()`, then inside the epoch loop log scalars: `writer.add_scalar('Loss/train', train_loss, epoch)` and similar for val metrics. Close writer at end.

36) Q: How would you profile whether the GPU is the bottleneck or data loading is? (Type: practical)

Answer:
- Monitor GPU utilization (nvidia-smi), check CPU usage, increase `num_workers` to see improvement; add timers around data loading vs forward/backward to measure time per batch.

37) Q: Show a minimal test to confirm `train_epoch` can overfit a few samples. (Type: coding)

Answer:
- Create a small DataLoader with 10 samples and labels, run `train_epoch` for several iterations with high lr; verify training loss becomes very small — this ensures code paths are correct.

38) Q: How would you add label smoothing to CrossEntropyLoss? (Type: coding / short)

Answer:
- Use `torch.nn.functional.cross_entropy` with soft labels or implement `LabelSmoothingCrossEntropy` where target distribution is mixed: `y_smooth = (1 - eps) * y + eps / num_classes` and compute KL or negative log-likelihood against logits.

39) Q: The DataLoader uses `shuffle=True` for training. Why is shuffling important? (Type: short)

Answer:
- Shuffling prevents the model from learning spurious order-based patterns and ensures batches are more representative of the dataset distribution.

40) Q: Where in the code would you add a validation set separate from test if you wanted to tune hyperparameters? (Type: practical)

Answer:
- Modify `get_data_loaders` or write a helper to split the training set into `train` and `val` (e.g., with `torch.utils.data.random_split`), and use the val loader for `val_loss` instead of the `testloader`.

41) Q: How could you implement learning rate warmup in this script? (Type: short)

Answer:
- Add a warmup scheduler (or manual scaling) for the first few epochs to increase lr from a small value to target lr, then switch to StepLR; PyTorch has utilities or custom lambda schedulers.

42) Q: What are common data augmentation improvements for CIFAR-10 beyond RandomCrop and Flip? (Type: practical)

Answer:
- RandomErasing, Cutout, Color jitter, AutoAugment, MixUp, CutMix — these often improve robustness and generalization.

43) Q: How to evaluate model calibration? (Type: short)

Answer:
- Compute reliability diagrams and expected calibration error (ECE) comparing predicted probabilities to empirical accuracy, using softmax probabilities from logits.

44) Q: Suggest a simple hyperparameter search strategy for this experiment. (Type: practical)

Answer:
- Start with grid or random search over learning rate and weight decay; use a small subset and fewer epochs to quickly evaluate; for more advanced, use Bayesian optimization (Optuna) or Sweep tools.

45) Q: Where to modify the batch size and why might you tune it? (Type: short)

Answer:
- Batch size is a CLI arg `--batch-size` (parse_args). Larger batches give more stable gradient estimates but require more memory and may affect generalization.

46) Q: How would you export the model to ONNX for inference? (Type: coding)

Answer:
- In `main()` after training: `dummy = torch.randn(1, 3, 32, 32).to(device)` then `torch.onnx.export(model, dummy, 'model.onnx', input_names=['input'], output_names=['output'], opset_version=11)`.

47) Q: What safety checks would you add before saving checkpoints to disk? (Type: short)

Answer:
- Ensure `checkpoint_dir` exists (function already calls `os.makedirs`), check disk space, optionally atomically write (save to temp file then move), and avoid overwriting best checkpoint unintentionally.

48) Q: How would you change the script to train on multiple GPUs (DataParallel)? (Type: coding)

Answer:
- Wrap model with `torch.nn.DataParallel(model)` after moving to device (or better: use `DistributedDataParallel`); ensure DataLoader uses `sampler=DistributedSampler` for distributed mode.

49) Q: Explain why dropout rate 0.5 may be used in the classifier. What side-effects should you monitor? (Type: short)

Answer:
- Dropout 0.5 is strong regularization to prevent co-adaptation in the dense layer; monitor training/validation gap — too much dropout may underfit.

50) Q: Create 3 viva follow-ups that test deep understanding of the training script. (Type: practical)

Answer (examples):
- a) Explain how you would profile the code to determine if the forward or backward pass is the bottleneck, and what tools you'd use.
- b) Suppose validation accuracy stalls — propose a prioritized list of debugging and improvement steps with expected impact.
- c) Describe how you would port this training script to a cloud GPU instance and ensure reproducibility.

---

Appendix — quick code references in `expt1/train_cifar10.py` (where to look)
- Model definition: `class SimpleCNN(nn.Module):` (top of file)
- Data loaders: `def get_data_loaders(batch_size=..., data_dir=...)`
- Batch training: `def train_epoch(model, device, dataloader, criterion, optimizer)`
- Validation: `def eval_model(model, device, dataloader, criterion)`
- Checkpointing: `def save_checkpoint(state, is_best, checkpoint_dir)`
- Main loop and orchestration: `def main()`


