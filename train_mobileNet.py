import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report


DATASET_PATH = Path("data/dataset-resized")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

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

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = preprocess_input(image)

    return image, label

def create_dataset(dataframe, shuffle=False):

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

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
train_ds = create_dataset(train_df, shuffle=True)

val_ds = create_dataset(val_df)

test_ds = create_dataset(test_df)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([

    tf.keras.Input(shape=(128, 128, 3)),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/best_mobilenetv2_garbage.keras",
        monitor="val_loss",
        save_best_only=True
    )
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)
model.save("models/mobilenetv2_garbage_classifier.keras")

print("MobileNetV2 model saved successfully!")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MobileNetV2 Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig("reports/mobilenetv2_training_performance.png")

plt.show()