
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

# 1. APPLICATION CONFIGURATION

st.set_page_config(
    page_title="Garbage Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

MODEL_PATH = Path("models/best_mobilenetv2_garbage.keras")
IMAGE_SIZE = (128, 128)

# Must match the sorted class-folder order used during training
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# 2. LOAD TRAINED MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)



# 3. IMAGE PREPROCESSING

def preprocess_image(image):
    # Convert image into RGB
    image = image.convert("RGB")

    # Convert PIL image to TensorFlow tensor
    image_array = np.array(image)

    image_tensor = tf.convert_to_tensor(image_array)

    # Resize to model input dimensions
    image_tensor = tf.image.resize(
        image_tensor,
        IMAGE_SIZE
    )

    # Apply MobileNetV2 preprocessing
    image_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_tensor
    )

    # Add batch dimension
    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    return image_tensor


# 4. APPLICATION INTERFACE

st.title("♻️ Garbage Waste Classification")

st.write(
    "Upload a garbage image and let our AI model "
    "identify its waste category."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose a garbage image",
    type=["jpg", "jpeg", "png"]
)


# 5. IMAGE DISPLAY AND PREDICTION

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Image selected for classification",
        width="stretch"
    )

    if st.button("Predict Waste Category", type="primary"):

        try:
            with st.spinner("Analyzing image..."):

                # Load model
                model = load_model()

                # Preprocess uploaded image
                processed_image = preprocess_image(image)

                # Generate prediction probabilities
                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                # Find predicted class
                predicted_index = int(np.argmax(predictions))

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    predictions[predicted_index] * 100
                )

            # Display result
            st.success("Prediction completed!")

            st.subheader("Prediction Result")

            st.markdown(
                f"### ♻️ {predicted_class.upper()}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # Display top 3 predictions
            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(predictions)[::-1][:3]

            for index in top_indices:

                category = CLASS_NAMES[index]
                probability = float(predictions[index] * 100)

                st.write(
                    f"**{category.capitalize()}** — "
                    f"{probability:.2f}%"
                )

                st.progress(probability / 100)

        except Exception as error:

            st.error(
                f"Something went wrong while predicting: {error}"
            )

else:

    st.info("Please upload a JPG, JPEG or PNG image to get started.")


# 6. FOOTER

st.divider()

st.caption(
    "Powered by TensorFlow and MobileNetV2 | "
    "Garbage Waste Classification Project"
)
