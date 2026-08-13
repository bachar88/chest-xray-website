"""
train.py
--------
The training loop: this is where the model actually learns.

WHAT HAPPENS DURING TRAINING (in plain terms):
For each batch of images:
  1. Feed images into the model -> get predictions
  2. Compare predictions to the true labels using a "loss function"
     (a number that says "how wrong were you?")
  3. Compute gradients (how much each weight contributed to the error)
  4. Update the weights slightly to reduce the error (this is what the
     "optimizer" does)
Repeat this for every batch, for several passes over the full dataset
("epochs"). After each epoch, we check performance on the validation set
(data the model never trains on) to make sure it's actually learning
general patterns, not memorizing the training images.
"""

import torch
import torch.nn as nn
import copy
import time


def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3, class_weights=None):
    """
    class_weights: optional tensor to up-weight rare classes in the loss
                   function. Medical datasets are often imbalanced (e.g.
                   fewer COVID images than Normal images), so this helps
                   the model not just default to predicting the majority
                   class.
    """
    model = model.to(device)

    # CrossEntropyLoss is the standard loss function for multi-class
    # classification problems.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    # Adam optimizer: an efficient, widely-used method for updating weights.
    # We only pass parameters that require gradients (i.e. aren't frozen).
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Reduces learning rate if validation loss stops improving -- helps
    # the model converge more precisely in later epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # ---------------- TRAINING PHASE ----------------
        model.train()  # tells layers like Dropout to behave in "training mode"
        running_loss, running_correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # clear old gradients
            outputs = model(images)             # forward pass -> predictions
            loss = criterion(outputs, labels)   # compute error
            loss.backward()                     # compute gradients (backpropagation)
            optimizer.step()                    # update weights

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels.data)
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct.double() / total

        # ---------------- VALIDATION PHASE ----------------
        model.eval()  # tells layers like Dropout to turn off (use full network)
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():  # no need to track gradients during evaluation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_correct += torch.sum(preds == labels.data)
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct.double() / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc.item())
        history["val_acc"].append(val_acc.item())

        elapsed = time.time() - start
        print(f"Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f}  Val acc:   {val_acc:.4f}")
        print(f"Time: {elapsed:.1f}s")

        # Keep the best-performing version of the model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print("New best model saved (in memory).")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_weights)
    return model, history
