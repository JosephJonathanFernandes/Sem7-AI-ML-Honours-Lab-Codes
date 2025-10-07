# Viva Questions — AI / ML Honours Lab

This file contains a curated list of viva (oral exam) questions useful for the course experiments in this repository. Questions are grouped by topic and include follow-ups that probe understanding.

1. Core Machine Learning Theory
- What is the difference between supervised, unsupervised and reinforcement learning? Follow-up: give an example dataset or task for each.
- Define bias and variance. How do they affect model performance?
- Explain the bias-variance trade-off and how model complexity interacts with it.
- What is overfitting and how can you detect it in training curves?
- Describe regularization techniques (L1, L2, dropout) and when to use them.
- Explain cross-validation. Why is it useful compared to a single train/test split?

2. Optimization and Losses
- What is gradient descent? Explain variants: batch, stochastic and mini-batch.
- How does learning rate affect training? What happens if it is too high or too low?
- Describe momentum, RMSprop, Adam. When would you choose one over the others?
- What is a loss function? Give examples used in classification and regression.
- Explain softmax and cross-entropy for multi-class classification.

3. Neural Networks and Architectures
- Explain how a feedforward neural network computes outputs from inputs.
- What is backpropagation? Give a high-level derivation of weight updates.
- Why do we use activation functions? Compare ReLU, Sigmoid, and Tanh.
- What are convolutional layers and why are they suited for images?
- Explain pooling (max/average). What are pros and cons?

4. Convolutional Networks & CIFAR-10 specifics
- Describe CIFAR-10 dataset: classes, input size, and splits.
- For CIFAR-10, why might data augmentation improve generalization? Give examples of augmentations.
- How do you design an architecture for CIFAR-10? What depth/width trade-offs matter?
- How would you modify the final layer and loss for binary vs. multi-class classification?
- Explain transfer learning. How could you apply it to CIFAR-10?

5. PyTorch Practical Questions
- How do you define a model in PyTorch? Show the structure of a simple nn.Module.
- Explain the difference between torch.Tensor and torch.nn.Parameter.
- How does autograd work in PyTorch? What does requires_grad=True do?
- How do you switch a model between train and eval modes? Why is it important?
- How do you save and load model checkpoints in PyTorch? What parts do you save (state_dict)?
- In `train_cifar10.py`, what parts of the code handle data loading, training loop, and checkpointing? (Look for DataLoader, optimizer.step, torch.save)

6. Experiment Reproducibility & Debugging
- How do you set random seeds to make experiments reproducible? What limitations remain?
- If training loss decreases but validation loss increases, what might be happening and how would you fix it?
- If a model's accuracy is stuck at random-chance level, list debugging steps you would take.
- How do you debug exploding or vanishing gradients?
- How would you profile training to find bottlenecks (CPU vs GPU)?

7. Mathematics (Short Derivations)
- Derive the gradient of the cross-entropy loss with softmax outputs.
- Show how the chain rule applies in backpropagation for a two-layer network.
- Explain why ReLU mitigates vanishing gradients compared to sigmoid.

8. Evaluation Metrics & Calibration
- Precision, recall, F1 — define and when to use each.
- Confusion matrix — explain and interpret for a multi-class problem.
- What is calibration of a classifier and how can you test it?

9. Advanced Topics (short answers)
- What is batch normalization and what problem does it address?
- Explain dropout: what does it do at training time vs inference time?
- What are residual connections (ResNet) and why are they useful?
- Describe a modern optimization trick (learning rate schedulers, warm restarts, cosine annealing).

10. Practical viva prompts (ask candidate to explain or modify code)
- Open `expt1/train_cifar10.py`. Explain the main training loop line by line.
- Show how you would add early stopping to the training script.
- Modify the script to use a pretrained ResNet from torchvision and fine-tune it on CIFAR-10.
- Show how to log training metrics to TensorBoard or a CSV file.

11. Soft questions — project understanding
- What did you learn from this experiment? Which hyperparameter had the biggest effect?
- If you had one more week to improve the model, what would you try?

Further reading and tips
- Be ready to write short snippets on the whiteboard: e.g., softmax formula, ReLU derivative, or pseudo-code for a training loop.
- Practice explaining trade-offs and intuition — viva examiners often ask "why" repeatedly.

Appendix — sample rapid-fire questions
- What is the dimensionality of CIFAR-10 images?
- What does SGD stand for?
- Name two activation functions other than ReLU.
- What is early stopping?
- What is the purpose of a validation set?
