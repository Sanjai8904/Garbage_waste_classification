
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from pathlib import Path
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    top_k_accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    hamming_loss,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = Path("data/dataset-resized")
MODEL_PATH = "models/best_mobilenetv2_garbage.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

Path("reports").mkdir(exist_ok=True)


# ==========================================
# 2. GET CLASS NAMES
# ==========================================

class_names = sorted(
    [folder.name for folder in DATASET_PATH.iterdir()
     if folder.is_dir()]
)

print("Classes:", class_names)


# ==========================================
# 3. COLLECT IMAGE PATHS AND LABELS
# ==========================================

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


# ==========================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

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

print("\nDataset Distribution:")
print("Training Images  :", len(train_df))
print("Validation Images:", len(val_df))
print("Testing Images   :", len(test_df))


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

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


# ==========================================
# 6. CREATE DATASETS
# ==========================================

def create_dataset(dataframe):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["image_path"].values,
            dataframe["label"].values
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


# ==========================================
# 7. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)


def calculate_metrics(dataset, dataframe, split_name):

    # Get actual labels
    y_true = dataframe["label"].values

    # Get predicted probabilities
    predictions = model.predict(dataset, verbose=0)

    # Convert probabilities into class predictions
    y_pred = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100

    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ) * 100

    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ) * 100

    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ) * 100

    mcc = matthews_corrcoef(y_true, y_pred) * 100

    return {
        "Split": split_name,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2),
        "MCC": round(mcc, 2)
    }

# ==========================================
# ROC CURVE GENERATION FUNCTION
# ==========================================

def generate_roc_curve(dataset, dataset_name):

    # Get actual labels
    y_true = np.concatenate([
        labels.numpy() for images, labels in dataset
    ])

    # Get predicted probabilities
    predictions = model.predict(dataset, verbose=0)

    # Convert actual labels into binary format
    y_true_binary = label_binarize(
        y_true,
        classes=class_labels
    )

    plt.figure(figsize=(10, 8))

    class_auc_scores = []

    # Plot all six classes in the same graph
    for i, class_name in enumerate(class_names):

        fpr, tpr, thresholds = roc_curve(
            y_true_binary[:, i],
            predictions[:, i]
        )

        class_auc = auc(fpr, tpr)
        class_auc_scores.append(class_auc)

    #     plt.plot(
    #         fpr,
    #         tpr,
    #         linewidth=2,
    #         label=f"{class_name} (AUC = {class_auc:.3f})"
    #     )

    # # Macro-average AUC
    # macro_auc = np.mean(class_auc_scores)

    # # Random classifier reference line
    # plt.plot(
    #     [0, 1],
    #     [0, 1],
    #     linestyle="--",
    #     color="black",
    #     label="Random Classifier"
    # )

    # plt.title(f"MobileNetV2 ROC Curve - {dataset_name}")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")

    # plt.xlim([0, 1])
    # plt.ylim([0, 1.05])

    # plt.legend(loc="lower right")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()

    # # Save graph
    # filename = f"reports/mobilenetv2_{dataset_name.lower()}_roc_curve.png"

    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()
    # plt.close()

    # print(f"\n{dataset_name} ROC-AUC Scores:")

    # for class_name, score in zip(class_names, class_auc_scores):
    #     print(f"{class_name}: {score:.4f}")

    # print(f"Macro Average AUC: {macro_auc:.4f}")
    # print(f"ROC Curve saved: {filename}")


# ==========================================
# 8. TEST SET EVALUATION
# ==========================================

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n========== TEST RESULTS ==========")

print("Test Loss    :", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# 9. PREDICTIONS
# ==========================================

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = test_df["label"].to_numpy()

num_classes = len(class_names)

class_labels = np.arange(num_classes)


# ==========================================
# 10. ADDITIONAL EVALUATION METRICS
# ==========================================

# Top-2 Accuracy
top2_accuracy = top_k_accuracy_score(
    y_true,
    predictions,
    k=2,
    labels=class_labels
)

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

# Cohen's Kappa
kappa = cohen_kappa_score(
    y_true,
    y_pred
)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(
    y_true,
    y_pred
)

# Macro ROC-AUC
roc_auc = roc_auc_score(
    y_true,
    predictions,
    multi_class="ovr",
    average="macro",
    labels=class_labels
)

# Macro PR-AUC
y_true_binary = label_binarize(
    y_true,
    classes=class_labels
)

pr_auc = average_precision_score(
    y_true_binary,
    predictions,
    average="macro"
)

# Log Loss
loss = log_loss(
    y_true,
    predictions,
    labels=class_labels
)

# Hamming Loss
h_loss = hamming_loss(
    y_true,
    y_pred
)


# ==========================================
# 11. INFERENCE TIME
# ==========================================

# Warm-up prediction
model.predict(test_ds, verbose=0)

start_time = time.perf_counter()

model.predict(test_ds, verbose=0)

end_time = time.perf_counter()

inference_time = (
    (end_time - start_time) / len(y_true)
)


# ==========================================
# 12. MACRO FALSE POSITIVE RATE
# ==========================================

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


# ==========================================
# 13. PRINT TEST CONFUSION MATRIX
# ==========================================

print("\n========== TEST CONFUSION MATRIX ==========")

print(cm_for_metrics)


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ==========================================
# 15. PRINT ADDITIONAL METRICS
# ==========================================

# print("\n========== ADDITIONAL EVALUATION METRICS ==========")

# print(f"Top-2 Accuracy       : {top2_accuracy:.4f}")
# print(f"Balanced Accuracy    : {balanced_acc:.4f}")
# print(f"Cohen's Kappa        : {kappa:.4f}")
# print(f"Matthews Corr. Coeff.: {mcc:.4f}")
# print(f"Macro ROC-AUC        : {roc_auc:.4f}")
# print(f"Macro PR-AUC         : {pr_auc:.4f}")
# print(f"Log Loss             : {loss:.4f}")
# print(f"Hamming Loss         : {h_loss:.4f}")
# print(f"Inference Time/Image : {inference_time:.6f} seconds")
# print(f"Macro FPR            : {macro_fpr:.4f}")


# ==========================================
# 16. CONFUSION MATRICES FOR ALL DATASETS
# ==========================================

def generate_confusion_matrix(dataset, dataset_name):

    true_labels = []
    predicted_labels = []

    for images, labels_batch in dataset:

        probabilities = model(images, training=False)

        predicted_classes = tf.argmax(
            probabilities,
            axis=1
        )

        true_labels.extend(labels_batch.numpy())
        predicted_labels.extend(predicted_classes.numpy())

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_labels
    )

    # print(f"\n========== {dataset_name.upper()} CONFUSION MATRIX ==========")
    # print(cm)

    # plt.figure(figsize=(9, 7))

    # sns.heatmap(
    #     cm,
    #     annot=True,
    #     fmt="d",
    #     cmap="Blues",
    #     xticklabels=class_names,
    #     yticklabels=class_names
    # )

    # plt.xlabel("Predicted Category")
    # plt.ylabel("Actual Category")

    # plt.title(f"MobileNetV2 {dataset_name} Confusion Matrix")

    # plt.tight_layout()

    # output_path = (
    #     Path("reports") /
    #     f"mobilenetv2_{dataset_name.lower()}_confusion_matrix.png"
    # )

    # plt.savefig(
    #     output_path,
    #     dpi=300,
    #     bbox_inches="tight"
    # )

    # plt.show()
    # plt.close()

    # print(f"{dataset_name} confusion matrix saved: {output_path}")


# ==========================================
# 17. GENERATE THREE CONFUSION MATRICES
# ==========================================

generate_confusion_matrix(train_ds, "Training (70%)")

generate_confusion_matrix(val_ds, "Validation (15%)")

generate_confusion_matrix(test_ds, "Testing (15%)")

# ==========================================
# GENERATE THREE ROC CURVES
# ==========================================

class_labels = np.arange(len(class_names))

generate_roc_curve(train_ds, "Training")
generate_roc_curve(val_ds, "Validation")
generate_roc_curve(test_ds, "Testing")

# Calculate metrics for all three splits

train_metrics = calculate_metrics(
    train_ds, train_df, "Train"
)

test_metrics = calculate_metrics(
    test_ds, test_df, "Test"
)

val_metrics = calculate_metrics(
    val_ds, val_df, "Validation"
)

# Create comparison table
comparison_df = pd.DataFrame([
    train_metrics,
    test_metrics,
    val_metrics
])

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison_df.to_string(index=False))
