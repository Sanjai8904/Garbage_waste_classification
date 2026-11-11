
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# STEP 1: SETTINGS
# ==========================================

DATASET_PATH = Path("data/dataset-resized")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print("Classes:", class_names)


# ==========================================
# STEP 2: COLLECT IMAGE PATHS AND LABELS
# ==========================================

image_paths = []
image_labels = []

for label, category in enumerate(class_names):

    folder_path = DATASET_PATH / category

    for image_file in folder_path.iterdir():

        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            image_paths.append(str(image_file))
            image_labels.append(label)


df = pd.DataFrame({
    "image_path": image_paths,
    "label": image_labels
})


# ==========================================
# STEP 3: SPLIT DATASET
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

# print("\nTraining:", len(train_df))
# print("Validation:", len(val_df))
# print("Testing:", len(test_df))


# ==========================================
# STEP 4: LOAD AND PREPROCESS IMAGES
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

    image = image / 255.0

    return image, label


def create_dataset(dataframe, training=False):

    paths = dataframe["image_path"].values
    labels = dataframe["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(tf.data.AUTOTUNE)


train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


# ==========================================
# STEP 5: DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1)

])


# ==========================================
# STEP 6: BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(128, 128, 3)),

    data_augmentation,

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        len(class_names), activation="softmax"
    )

])


# ==========================================
# STEP 7: DISPLAY MODEL ARCHITECTURE
# ==========================================

model.summary()


# ==========================================
# STEP 8: COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# STEP 9: HANDLE CLASS IMBALANCE
# ==========================================

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_df["label"]),

    y=train_df["label"]

)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:", class_weights)


# ==========================================
# STEP 10: TRAIN MODEL
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        "models/best_garbage_classifier.keras",

        save_best_only=True,

        monitor="val_loss"

    )

]


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ==========================================
# STEP 11: SAVE FINAL MODEL
# ==========================================

model.save("models/garbage_classifier.keras")

print("\nModel Training Completed!")
print("Model Saved Successfully!")


# ==========================================
# STEP 12: VISUALIZE TRAINING PERFORMANCE
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()


plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()


plt.tight_layout()

plt.savefig("reports/training_performance.png", dpi=300)

plt.show()

best_val_loss = min(history.history["val_loss"])
best_epoch = history.history["val_loss"].index(best_val_loss) + 1

print("Best Validation Loss:", best_val_loss)
print("Best Epoch:", best_epoch)