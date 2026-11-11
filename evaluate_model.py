
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Settings
DATASET_PATH = Path("data/dataset-resized")
MODEL_PATH = "models/best_garbage_classifier.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# 2. Get class names
class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print("Classes:", class_names)

# 3. Collect image paths and labels
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

# 4. Recreate the same 70/15/15 split
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

print("\nTest images:", len(test_df))

# 5. Load and preprocess images
def load_image(image_path, label):
    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = image / 255.0

    return image, label


# 6. Create test dataset
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

# 7. Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

# 8. Evaluate test dataset
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 9. Predict test images
predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)
y_true = test_df["label"].to_numpy()

# 10. Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

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
plt.title("Garbage Classification - Confusion Matrix")

plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=300)
plt.show()

# 11. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)
