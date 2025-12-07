import numpy as np
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess,
    decode_predictions,
)

# -------------------------------
# 1. GLOBAL SETTINGS
# -------------------------------
FASHION_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

MODEL_PATH = "fashion_mnist_cnn_model.h5"  # keep in same folder as app.py


# -------------------------------
# 2. CACHED HELPERS (MODELS + DATA)
# -------------------------------
@st.cache_resource
def load_fashion_model():
    """Load your trained Fashion-MNIST CNN (.h5)."""
    model = load_model(MODEL_PATH)
    return model


@st.cache_data
def load_fashion_test_data():
    """Load Fashion-MNIST test set (for demo)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)  # (N, 28, 28, 1)
    return x_test, y_test


@st.cache_resource
def load_mobilenet_model():
    """
    Load pre-trained MobileNetV2 for real-world images.
    Trained on ImageNet (1000 classes: T-shirt, shoe, bag, etc.).
    """
    model = MobileNetV2(weights="imagenet")
    return model


# -------------------------------
# 3. PREPROCESSING FUNCTIONS
# -------------------------------
def preprocess_for_fashion_model(uploaded_file):
    """
    For your Fashion-MNIST CNN:
    - Convert to grayscale
    - Resize to 28x28
    - Binarize + optional invert
    """
    orig = Image.open(uploaded_file).convert("L")
    orig_resized = orig.resize((28, 28))

    img_array = np.array(orig_resized).astype("float32") / 255.0

    # Threshold to make silhouette like Fashion-MNIST
    thresh = 0.5
    binary = (img_array > thresh).astype("float32")

    # Invert if mostly white (so background is dark)
    if np.mean(binary) > 0.5:
        binary = 1.0 - binary

    model_input = np.expand_dims(binary, axis=-1)   # (28, 28, 1)
    model_input = np.expand_dims(model_input, axis=0)  # (1, 28, 28, 1)

    processed_pil = Image.fromarray((binary * 255).astype("uint8"))
    return model_input, orig_resized, processed_pil


def preprocess_for_mobilenet(uploaded_file):
    """
    For MobileNetV2:
    - Convert to RGB
    - Resize to 224x224
    - Use MobileNetV2 preprocessing
    """
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = mobilenet_preprocess(img_array)
    return img_array, img_resized


# -------------------------------
# 4. PREDICTION HELPERS
# -------------------------------
def predict_fashion(model, img_array):
    """
    Predict with Fashion-MNIST CNN.
    img_array shape: (1, 28, 28, 1)
    """
    preds = model.predict(img_array)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return idx, probs


def predict_mobilenet(model, img_array):
    """
    Predict with MobileNetV2 (ImageNet).
    img_array shape: (1, 224, 224, 3)
    Returns decoded top-3 predictions.
    """
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]  # list of (class_id, name, prob)
    return decoded


def show_fashion_prediction(predicted_index, probs):
    """Display result for Fashion-MNIST CNN."""
    confidence = float(probs[predicted_index] * 100)

    st.write("### Prediction Result (Fashion-MNIST CNN)")
    if confidence < 50:
        st.warning(
            f"Model is **not very confident**.\n\n"
            f"Most likely: **{FASHION_CLASS_NAMES[predicted_index]}** "
            f"({confidence:.2f}% confidence)."
        )
    else:
        st.success(
            f"Model predicts: **{FASHION_CLASS_NAMES[predicted_index]}** "
            f"({confidence:.2f}% confidence)."
        )

    top3_idx = np.argsort(probs)[::-1][:3]
    st.write("#### Top 3 guesses:")
    for i in top3_idx:
        st.write(f"- {FASHION_CLASS_NAMES[i]}: {float(probs[i] * 100):.2f}%")

    prob_dict = {
        FASHION_CLASS_NAMES[i]: float(probs[i]) for i in range(len(FASHION_CLASS_NAMES))
    }
    st.write("#### All class probabilities")
    st.bar_chart(prob_dict)


def show_mobilenet_prediction(decoded_preds):
    """Display result for MobileNetV2 (real-world photos)."""
    st.write("### Prediction Result (MobileNetV2 – Real Photos)")
    top1 = decoded_preds[0]
    top_name = top1[1].replace("_", " ")
    top_prob = top1[2] * 100.0

    if top_prob < 50:
        st.warning(
            f"Most likely: **{top_name}** "
            f"({top_prob:.2f}% confidence, not very sure)."
        )
    else:
        st.success(
            f"Model predicts: **{top_name}** "
            f"({top_prob:.2f}% confidence)."
        )

    st.write("#### Top 3 guesses:")
    for (class_id, name, prob) in decoded_preds:
        st.write(f"- {name.replace('_', ' ')}: {prob * 100:.2f}%")


# -------------------------------
# 5. STREAMLIT UI
# -------------------------------
def main():
    st.set_page_config(page_title="Fashion Image Classifier", layout="wide")
    st.title("🧥 Fashion Image Classifier – CNN + MobileNetV2")

    st.write(
        """
        This mini project uses:
        - ✅ **Fashion-MNIST CNN** (your trained model) – good for 28×28 gray images.
        - ✅ **MobileNetV2 (pretrained on ImageNet)** – better for **real clothing photos**.
        """
    )

    # Sidebar mode selection
    st.sidebar.header("Choose Mode")
    mode = st.sidebar.radio(
        "Select what you want to do:",
        (
            "1️⃣ Fashion-MNIST Test Image",
            "2️⃣ Upload Image → Use Fashion-MNIST CNN",
            "3️⃣ Upload Image → Use MobileNetV2 (Real Photo)"
        )
    )

    # Load models safely
    fashion_model = None
    mobilenet_model = None

    # Only load models when needed
    if mode in ("1️⃣ Fashion-MNIST Test Image", "2️⃣ Upload Image → Use Fashion-MNIST CNN"):
        try:
            fashion_model = load_fashion_model()
        except Exception as e:
            st.error(
                f"❌ Error loading Fashion-MNIST model from `{MODEL_PATH}`.\n"
                "Run your training script first so the `.h5` file is created."
            )
            st.exception(e)
            return

    if mode == "3️⃣ Upload Image → Use MobileNetV2 (Real Photo)":
        try:
            mobilenet_model = load_mobilenet_model()
        except Exception as e:
            st.error("❌ Error loading MobileNetV2 model.")
            st.exception(e)
            return

    # -----------------------------------
    # MODE 1: Fashion-MNIST test images
    # -----------------------------------
    if mode == "1️⃣ Fashion-MNIST Test Image":
        x_test, y_test = load_fashion_test_data()

        index = st.sidebar.slider(
            "Select test image index",
            min_value=0,
            max_value=len(x_test) - 1,
            value=0,
            step=1
        )

        st.subheader("Fashion-MNIST Test Image")

        img = x_test[index]
        true_label = int(y_test[index])

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img.squeeze(),
                width=200,
                caption=f"True label: {FASHION_CLASS_NAMES[true_label]}"
            )

        with col2:
            st.write(
                "This is from the **Fashion-MNIST test set**:\n"
                "- 28×28 grayscale\n"
                "- One of 10 clothing classes"
            )

        if st.button("🔍 Predict this test image"):
            img_array = np.expand_dims(img, axis=0)  # (1, 28, 28, 1)
            pred_idx, probs = predict_fashion(fashion_model, img_array)
            show_fashion_prediction(pred_idx, probs)

    # -----------------------------------
    # MODE 2: Upload image → use Fashion-MNIST CNN
    # -----------------------------------
    elif mode == "2️⃣ Upload Image → Use Fashion-MNIST CNN":
        st.subheader("Upload Image – Fashion-MNIST CNN")
        st.write(
            "Use this if you want to **demonstrate your own model**.\n"
            "Works better if the image is simple (plain background, single clothing item)."
        )

        uploaded_file = st.file_uploader(
            "Upload an image (jpg / jpeg / png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            model_input, orig_resized, processed_pil = preprocess_for_fashion_model(
                uploaded_file
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Original (resized to 28×28 grayscale)")
                st.image(orig_resized, width=200)

            with col2:
                st.write("Processed (binarized, like Fashion-MNIST)")
                st.image(processed_pil, width=200)

            if st.button("🔍 Predict with Fashion-MNIST CNN"):
                pred_idx, probs = predict_fashion(fashion_model, model_input)
                show_fashion_prediction(pred_idx, probs)
        else:
            st.info("⬆️ Please upload an image to classify with your CNN.")

    # -----------------------------------
    # MODE 3: Upload image → use MobileNetV2
    # -----------------------------------
    else:  # "3️⃣ Upload Image → Use MobileNetV2 (Real Photo)"
        st.subheader("Upload Real Clothing Image – MobileNetV2")
        st.write(
            "Use this for **real-life clothing photos**. "
            "MobileNetV2 is trained on millions of color images (ImageNet)."
        )

        uploaded_file = st.file_uploader(
            "Upload an image (jpg / jpeg / png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            model_input, img_resized = preprocess_for_mobilenet(uploaded_file)

            st.write("Resized image (224×224 RGB used for MobileNetV2):")
            st.image(img_resized, width=250)

            if st.button("🔍 Predict with MobileNetV2"):
                decoded = predict_mobilenet(mobilenet_model, model_input)
                show_mobilenet_prediction(decoded)
        else:
            st.info("⬆️ Please upload an image to classify with MobileNetV2.")


if __name__ == "__main__":
    main()
