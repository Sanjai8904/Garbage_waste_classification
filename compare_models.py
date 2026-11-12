
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

DATASET_PATH = Path("data/dataset-resized")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

CNN_MODEL_PATH = "models/best_garbage_classifier.keras"
MOBILENET_MODEL_PATH = "models/best_mobilenetv2_garbage.keras"

Path("reports").mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. LOAD DATASET
# ==========================================

class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

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

# Recreate the same 70/15/15 split
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

print("Dataset loaded successfully!")
print("Classes:", class_names)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def load_image(image_path, label, model_type):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    if model_type == "cnn":
        image = image / 255.0

    elif model_type == "mobilenet":
        image = preprocess_input(image)

    return image, label


def create_dataset(dataframe, model_type):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["image_path"].values,
            dataframe["label"].values
        )
    )

    dataset = dataset.map(
        lambda image_path, label: load_image(
            image_path, label, model_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Create test datasets using model-specific preprocessing
cnn_test_ds = create_dataset(test_df, "cnn")

mobilenet_test_ds = create_dataset(test_df, "mobilenet")

# ==========================================
# 4. LOAD TRAINED MODELS
# ==========================================

print("\nLoading CNN model...")

cnn_model = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("CNN model loaded successfully!")

print("\nLoading MobileNetV2 model...")

mobilenet_model = tf.keras.models.load_model(
    MOBILENET_MODEL_PATH
)

print("MobileNetV2 model loaded successfully!")

# ==========================================
# 5. MODEL PREDICTIONS
# ==========================================

y_true = test_df["label"].values

print("\nGenerating CNN predictions...")

cnn_probabilities = cnn_model.predict(
    cnn_test_ds,
    verbose=1
)

cnn_predictions = np.argmax(
    cnn_probabilities,
    axis=1
)

print("\nGenerating MobileNetV2 predictions...")

mobilenet_probabilities = mobilenet_model.predict(
    mobilenet_test_ds,
    verbose=1
)

mobilenet_predictions = np.argmax(
    mobilenet_probabilities,
    axis=1
)

# ==========================================
# 6. CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(y_true, y_pred, model_name):

    accuracy = accuracy_score(
        y_true, y_pred
    ) * 100

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    ) * 100

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    ) * 100

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    ) * 100

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2)
    }


cnn_metrics = calculate_metrics(
    y_true,
    cnn_predictions,
    "CNN"
)

mobilenet_metrics = calculate_metrics(
    y_true,
    mobilenet_predictions,
    "MobileNetV2"
)

# ==========================================
# 7. CREATE COMPARISON TABLE
# ==========================================

comparison_df = pd.DataFrame([
    cnn_metrics,
    mobilenet_metrics
])

print("\n======================================")
print("MODEL PERFORMANCE COMPARISON")
print("======================================")

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# ==========================================
# 8. GENERATE COMPARISON BAR GRAPH
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

cnn_scores = [
    cnn_metrics[metric] for metric in metrics
]

mobilenet_scores = [
    mobilenet_metrics[metric] for metric in metrics
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    x - width / 2,
    cnn_scores,
    width,
    label="CNN"
)

bars2 = plt.bar(
    x + width / 2,
    mobilenet_scores,
    width,
    label="MobileNetV2"
)

plt.bar_label(bars1, fmt="%.2f", padding=3)
plt.bar_label(bars2, fmt="%.2f", padding=3)

plt.title("CNN vs MobileNetV2 Performance Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")

plt.xticks(x, metrics)
plt.ylim(0, 105)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "reports/cnn_vs_mobilenetv2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nComparison graph saved successfully!")
print("Comparison table saved successfully!")
