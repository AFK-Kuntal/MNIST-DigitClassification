import streamlit as st
import numpy as np
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# --- Page config ---
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    return keras.models.load_model("digit_classifier.keras")

model = load_model()

# --- UI ---
st.title("🔢 MNIST Digit Classifier")
st.write("Draw a digit **(0–9)** in the box below, then click **Predict**.")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )

with col2:
    st.markdown("### Result")
    predict_btn = st.button("🔍 Predict", use_container_width=True)
    clear_note = st.caption("Use the ↺ button on the canvas to clear and redraw.")

    if predict_btn:
        if canvas_result.image_data is not None:
            # Convert canvas RGBA → grayscale → 28x28
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
            img = img.convert("L")                   # grayscale
            img = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            preds = model.predict(img_array, verbose=0)[0]
            digit = int(np.argmax(preds))
            confidence = float(preds[digit]) * 100

            st.metric(label="Predicted Digit", value=str(digit))
            st.metric(label="Confidence", value=f"{confidence:.1f}%")

            st.markdown("**All probabilities:**")
            for i, p in enumerate(preds):
                st.progress(float(p), text=f"  {i} → {p*100:.1f}%")
        else:
            st.warning("Please draw a digit first!")

st.divider()
st.caption("Model: CNN trained on MNIST · ~99% val accuracy")