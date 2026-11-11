import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import (
    top_k_accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    hamming_loss
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
DATASET_PATH = Path("data/dataset-resized")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)
image_paths = []
labels = []

for label, category in enumerate(class_names):

    category_path = DATASET_PATH / category

    for image_path in category_path.iterdir():

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            image_paths.append(str(image_path))
            labels.append(label)

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

print("Test images:", len(test_df))
def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label
test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["label"].values
    )
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
model = tf.keras.models.load_model(
    "models/best_mobilenetv2_garbage.keras"
)

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = test_df["label"].to_numpy()


# ==========================================
# ADDITIONAL EVALUATION METRICS
# ==========================================

num_classes = len(class_names)
class_labels = np.arange(num_classes)

# 1. Top-2 Accuracy
top2_accuracy = top_k_accuracy_score(
    y_true,
    predictions,
    k=2,
    labels=class_labels
)

# 2. Balanced Accuracy
balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

# 3. Cohen's Kappa
kappa = cohen_kappa_score(
    y_true,
    y_pred
)

# 4. Matthews Correlation Coefficient
mcc = matthews_corrcoef(
    y_true,
    y_pred
)

# 5. Macro ROC-AUC
roc_auc = roc_auc_score(
    y_true,
    predictions,
    multi_class="ovr",
    average="macro",
    labels=class_labels
)

# 6. Macro PR-AUC (Average Precision)
y_true_binary = label_binarize(
    y_true,
    classes=class_labels
)

pr_auc = average_precision_score(
    y_true_binary,
    predictions,
    average="macro"
)

# 7. Log Loss
loss = log_loss(
    y_true,
    predictions,
    labels=class_labels
)

# 8. Hamming Loss
h_loss = hamming_loss(
    y_true,
    y_pred
)

# 9. Inference Time
# Warm-up has already occurred through the earlier prediction.
start_time = time.perf_counter()

model.predict(test_ds, verbose=0)

end_time = time.perf_counter()

inference_time = (
    (end_time - start_time) / len(y_true)
)

# 10. Macro False Positive Rate
cm_for_metrics = confusion_matrix(
    y_true,
    y_pred,
    labels=class_labels
)

total = np.sum(cm_for_metrics)

fpr_values = []

for i in range(num_classes):

    tp = cm_for_metrics[i, i]

    fp = np.sum(cm_for_metrics[:, i]) - tp

    fn = np.sum(cm_for_metrics[i, :]) - tp

    tn = total - tp - fp - fn

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    fpr_values.append(fpr)

macro_fpr = np.mean(fpr_values)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)
print(f"Top-2 Accuracy       : {top2_accuracy:.4f}")
print(f"Balanced Accuracy    : {balanced_acc:.4f}")
print(f"Cohen's Kappa        : {kappa:.4f}")
print(f"Matthews Corr. Coeff.: {mcc:.4f}")
print(f"Macro ROC-AUC        : {roc_auc:.4f}")
print(f"Macro PR-AUC         : {pr_auc:.4f}")
print(f"Log Loss             : {loss:.4f}")
print(f"Hamming Loss         : {h_loss:.4f}")
print(f"Inference Time/Image : {inference_time:.6f} seconds")
print(f"Macro FPR            : {macro_fpr:.4f}")

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.title("MobileNetV2 Confusion Matrix")

plt.tight_layout()

plt.savefig("reports/mobilenetv2_confusion_matrix.png")

plt.show()