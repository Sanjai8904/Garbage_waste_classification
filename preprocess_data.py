
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


# STEP 1: Define dataset path and image settings

DATASET_PATH = Path("data/dataset-resized")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# STEP 2: Collect image paths and their labels

class_names = sorted([
    folder.name
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

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

# print("Class Names:", class_names)
# print("Total Images:", len(df))


# STEP 3: Split into training, validation and testing

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

# print("\nDataset Split:")
# print("Training Images:", len(train_df))
# print("Validation Images:", len(val_df))
# print("Testing Images:", len(test_df))


# STEP 4: Create a function to load and preprocess images

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


# STEP 5: Create TensorFlow datasets

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


# STEP 6: Define data augmentation

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# print("\nData augmentation layers:")
# data_augmentation.summary()


# STEP 7: Visualize augmented images

images, labels = next(iter(train_ds))

augmented_images = data_augmentation(
    images,
    training=True
)

# plt.figure(figsize=(10, 8))

# for i in range(9):

#     plt.subplot(3, 3, i + 1)

#     plt.imshow(augmented_images[i].numpy())

#     plt.title(class_names[int(labels[i])])

#     plt.axis("off")

# plt.tight_layout()

# plt.savefig(
#     "reports/augmented_samples.png",
#     dpi=300
# )

# plt.show()


# STEP 8: Verify dataset shapes and pixel values

images, labels = next(iter(train_ds))

# print("\nBatch Verification:")
# print("Image Batch Shape:", images.shape)
# print("Label Batch Shape:", labels.shape)
# print("Minimum Pixel Value:", tf.reduce_min(images).numpy())
# print("Maximum Pixel Value:", tf.reduce_max(images).numpy())

# print("\nPreprocessing Completed Successfully!")
