import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


st.set_page_config(
    page_title="Food-101 Image Recognition & Smart Menu Classification",
    layout="wide"
)


MODEL_PATH = Path("models/food101_mobilenetv2_classifier.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
METRICS_PATH = Path("reports/model_metrics.txt")
CONFUSION_MATRIX_PATH = Path("reports/confusion_matrix.png")


MENU_MAPPING = {
    "fried_rice": ("Rice / Staple", ["rice dish", "staple", "hot dish", "buffet item"]),
    "dumplings": ("Appetiser / Side Dish", ["dumpling", "side dish", "sharing item"]),
    "gyoza": ("Appetiser / Side Dish", ["dumpling", "Japanese style", "side dish"]),
    "sushi": ("Cold Dish / Japanese", ["cold dish", "rice", "seafood", "Japanese cuisine"]),
    "sashimi": ("Cold Dish / Seafood", ["seafood", "cold dish", "premium item"]),
    "ramen": ("Noodle / Main Dish", ["noodle soup", "hot dish", "main dish"]),
    "pho": ("Noodle / Soup", ["noodle soup", "Vietnamese style", "hot dish"]),
    "pad_thai": ("Noodle / Main Dish", ["stir-fried noodle", "Thai style", "hot dish"]),
    "peking_duck": ("Meat / Main Dish", ["duck", "premium item", "Chinese cuisine"]),
    "samosa": ("Snack / Appetiser", ["fried snack", "appetiser", "street food"]),
    "spring_rolls": ("Snack / Appetiser", ["fried item", "appetiser", "sharing item"]),
    "pizza": ("Main Dish / Bakery", ["baked item", "sharing dish", "high demand item"]),
    "hamburger": ("Main Dish / Fast Food", ["meat", "fast food", "main dish"]),
    "steak": ("Meat / Main Dish", ["protein", "premium item", "hot dish"]),
    "fish_and_chips": ("Seafood / Main Dish", ["fried seafood", "main dish", "Western style"]),
    "grilled_salmon": ("Seafood / Main Dish", ["grilled item", "protein", "healthy option"]),
    "caesar_salad": ("Salad / Healthy Item", ["salad", "healthy option", "cold dish"]),
    "ice_cream": ("Dessert", ["cold dessert", "sweet item", "dairy"]),
    "cheesecake": ("Dessert", ["sweet item", "bakery", "dessert"]),
    "pancakes": ("Breakfast / Dessert", ["breakfast item", "sweet item", "bakery"]),
}


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


def prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    resized = img.resize((224, 224))
    arr = np.array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return img, arr


def get_top_predictions(predictions, class_names, top_k=3):
    probs = predictions[0]
    top_indices = probs.argsort()[-top_k:][::-1]

    rows = []
    for idx in top_indices:
        rows.append({
            "class_name": class_names[idx],
            "display_name": format_label(class_names[idx]),
            "confidence": float(probs[idx])
        })

    return pd.DataFrame(rows)


def get_menu_info(predicted_class):
    category, tags = MENU_MAPPING.get(
        predicted_class,
        ("Food Item", ["food item", "menu item", "needs review"])
    )
    return category, tags


def generate_operational_suggestion(predicted_class, category, tags, confidence):
    if confidence < 0.50:
        return (
            "The model confidence is moderate or low. A manual review is recommended before using this prediction for menu tagging or operational decisions."
        )

    if "dessert" in tags or category == "Dessert":
        return (
            "This item can be tagged as a dessert or sweet item. It may support dessert menu planning, customer preference analysis, and seasonal promotion design."
        )

    if "seafood" in tags:
        return (
            "This item can be tagged as a seafood menu item. It may support premium menu grouping, freshness control, and inventory planning."
        )

    if "healthy option" in tags or "salad" in tags:
        return (
            "This item can be tagged as a healthy or fresh menu option. It may support nutrition-oriented menu filtering and customer recommendation systems."
        )

    if "rice dish" in tags or "noodle" in " ".join(tags):
        return (
            "This item can support staple food demand forecasting, buffet replenishment planning, and menu category analysis."
        )

    if "premium item" in tags or "protein" in tags:
        return (
            "This item can be used for protein-based menu classification, cost control, and premium item demand analysis."
        )

    return (
        "This prediction can support digital menu tagging, menu search, food category filtering, and AI-assisted foodservice analytics."
    )


model = load_model()
class_names = load_class_names()


st.title("Food-101 Image Recognition & Smart Menu Classification System")

st.markdown(
    """
This dashboard uses a **MobileNetV2 transfer learning model** trained on a 20-class subset of the Food-101 dataset.
It recognises food images, shows top prediction probabilities, and converts the model result into smart menu tags and foodservice management suggestions.
"""
)

tab1, tab2, tab3 = st.tabs([
    "Image Recognition Demo",
    "Model Performance",
    "Project Notes"
])


with tab1:
    st.header("Upload a Food Image")

    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        original_img, processed_img = prepare_image(uploaded_file)

        predictions = model.predict(processed_img, verbose=0)
        top_df = get_top_predictions(predictions, class_names, top_k=3)

        top_class = top_df.iloc[0]["class_name"]
        top_display = top_df.iloc[0]["display_name"]
        top_confidence = top_df.iloc[0]["confidence"]

        menu_category, menu_tags = get_menu_info(top_class)
        suggestion = generate_operational_suggestion(
            predicted_class=top_class,
            category=menu_category,
            tags=menu_tags,
            confidence=top_confidence
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(original_img, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")
            st.metric("Top Prediction", top_display)
            st.metric("Confidence", f"{top_confidence:.2%}")
            st.info(f"Suggested Menu Category: {menu_category}")

            st.write("Suggested Menu Tags:")
            st.write(", ".join(menu_tags))

        st.subheader("Top 3 Predictions")
        display_df = top_df.copy()
        display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df[["display_name", "confidence"]], use_container_width=True)

        fig = px.bar(
            top_df,
            x="display_name",
            y="confidence",
            title="Top 3 Prediction Confidence",
            labels={"display_name": "Food Class", "confidence": "Confidence"}
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Operational Suggestion")
        st.write(suggestion)

    else:
        st.info("Please upload a food image to start the recognition demo.")


with tab2:
    st.header("Model Performance")

    st.markdown(
        """
The model was trained using MobileNetV2 transfer learning on a 20-class subset of Food-101.
The final validation accuracy is approximately **84%**.
"""
    )

    if METRICS_PATH.exists():
        st.subheader("Classification Report")
        st.text(METRICS_PATH.read_text())

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion Matrix")
        st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)


with tab3:
    st.header("Project Notes")

    st.markdown(
        """
### Project Purpose

This project demonstrates how deep learning-based image recognition can support digital menu systems and AI-assisted foodservice operations.

### Dataset

The model uses a selected 20-class subset of Food-101, including rice dishes, noodles, seafood, desserts, Western main dishes, appetisers, and salads.

### Model

- Base model: MobileNetV2
- Method: Transfer learning
- Image size: 224 × 224
- Number of classes: 20
- Validation accuracy: approximately 84%

### Operational Relevance

The system can support:

- Food image recognition
- Digital menu tagging
- Menu search and filtering
- Buffet menu classification
- Customer recommendation systems
- Foodservice data analytics

### Limitations

The model recognises only the selected 20 Food-101 classes. It is not yet trained on Malaysian-specific dishes such as nasi lemak, roti canai, or char kway teow.

### Future Improvements

- Add Malaysian cuisine image classes
- Fine-tune EfficientNet or Vision Transformer models
- Add Grad-CAM visual explanations
- Add nutrition and allergen tags
- Integrate with a digital menu management system
"""
    )
