# ♻️ Garbage Waste Classification

A Deep Learning-based image classification project that identifies different types of garbage using CNN and MobileNetV2. The project includes model training, evaluation, and a Streamlit web application for image-based waste classification.

## 🎯 Project Overview

The system classifies garbage images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## 🧠 Models Implemented

1. **Custom CNN** – Built and trained from scratch.
2. **MobileNetV2** – Transfer learning using pretrained ImageNet weights.

### Model Performance

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 52.89% | **83.42%** |
| Macro F1-Score | 0.52 | **0.80** |

MobileNetV2 achieved better performance and was selected for the Streamlit application.

## 📊 Dataset

- **Dataset:** [TrashNet](https://github.com/garythung/trashnet)
- Image size: 128 × 128 pixels
- Classes: 6
- Training: 70%
- Validation: 15%
- Testing: 15%

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- CNN and MobileNetV2
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Pillow

## 🖥️ Run the Application

Clone the repository:

```bash
git clone https://github.com/Sanjai8904/garbage-waste-classification.git
cd garbage-waste-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```text
garbage-waste-classification/
│
├── app.py
├── train_model.py
├── train_mobileNet.py
├── evaluate_model.py
├── evaluate_mobileNet.py
├── preprocess_data.py
├── explore_dataset.py
│
├── models/
│   ├── best_garbage_classifier.keras
│   └── best_mobilenetv2_garbage.keras
│
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Future Improvements

- Improve classification accuracy.
- Experiment with additional pretrained models.
- Support real-time camera-based classification.
- Deploy the application online.

## 👨‍💻 Author

**Sanjai S.**

GitHub: [Sanjai8904](https://github.com/Sanjai8904)

