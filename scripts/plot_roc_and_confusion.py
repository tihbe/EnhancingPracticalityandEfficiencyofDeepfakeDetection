import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

sns.set(font_scale=2)

actual_labels = torch.load("actual_labels.pt")
predicted_labels = torch.load("predicted_labels.pt")

# ROC Curve

fpr, tpr, thresholds = roc_curve(actual_labels, predicted_labels)

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
ax.plot(fpr, tpr, label="ROC curve")
ax.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

fig.savefig("roc_curve.eps", bbox_inches="tight")
fig.savefig("roc_curve.png", bbox_inches="tight")

# Confusion Matrix
cm = confusion_matrix(actual_labels, predicted_labels.round())
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
df_cm = pd.DataFrame(cm, range(2), range(2))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues", fmt="d", ax=ax)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

fig.savefig("confusion_matrix.eps", bbox_inches="tight")
fig.savefig("confusion_matrix.png", bbox_inches="tight")

plt.show()
