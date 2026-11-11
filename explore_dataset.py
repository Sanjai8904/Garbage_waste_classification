
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 1. Define dataset path
DATASET_PATH = Path("data/dataset-resized")

# 2. Get all category names
categories = sorted(
    [folder.name for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

# print("Waste Categories:", categories)
# print("Number of Categories:", len(categories))


# 3. Count images in each category
image_counts = {}

for category in categories:
    folder_path = DATASET_PATH / category

    images = [
        file for file in folder_path.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    image_counts[category] = len(images)

# print("\nImage Count:")
# for category, count in image_counts.items():
#     print(f"{category}: {count}")

# print("\nTotal Images:", sum(image_counts.values()))


# 4. Create a DataFrame
df = pd.DataFrame(
    list(image_counts.items()),
    columns=["Category", "Image_Count"]
)

# print("\nDataset Summary:")
# print(df)


# 5. Check image dimensions
dimensions = []

for category in categories:
    folder_path = DATASET_PATH / category

    for file in folder_path.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            with Image.open(file) as img:
                dimensions.append(img.size)

dimension_df = pd.Series(dimensions)

# print("\nMost Common Image Dimensions:")
# print(dimension_df.value_counts().head(10))


# 6. Visualize class distribution
# plt.figure(figsize=(10, 5))

# sns.barplot(
#     data=df,
#     x="Category",
#     y="Image_Count"
# )

# plt.title("Number of Images in Each Waste Category")
# plt.xlabel("Waste Category")
# plt.ylabel("Number of Images")
# plt.tight_layout()

# plt.savefig("reports/class_distribution.png", dpi=300)
# plt.show()


# 7. Display sample images from all categories
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# for i, category in enumerate(categories):
#     folder_path = DATASET_PATH / category

#     image_files = sorted([
#         file for file in folder_path.iterdir()
#         if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
#     ])

#     sample_image = Image.open(image_files[0])

#     axes[i // 3, i % 3].imshow(sample_image)
#     axes[i // 3, i % 3].set_title(category.capitalize())
#     axes[i // 3, i % 3].axis("off")

# plt.tight_layout()
# plt.savefig("reports/sample_images.png", dpi=300)
# plt.show()
