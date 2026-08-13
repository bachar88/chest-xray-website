"""
evaluate.py
-----------
Proper evaluation of the trained model on the test set.

WHY NOT JUST "ACCURACY":
In medical imaging, accuracy alone is misleading. If 90% of your images
are "Normal", a model that always predicts "Normal" gets 90% accuracy
while being medically useless (it misses every real case).

Instead we look at, PER CLASS:
  - Precision: of the times the model predicted a class, how often was it right?
  - Recall (a.k.a. sensitivity): of all the real cases of a class, how many did
    the model catch? THIS IS THE MOST IMPORTANT METRIC FOR DISEASE DETECTION
    -- missing a real case (a false negative) is the costly mistake in medicine.
  - F1-score: balance between precision and recall.
  - Confusion matrix: a table showing exactly which classes get confused
    with which other classes.
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    return all_preds, all_labels, cm


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
