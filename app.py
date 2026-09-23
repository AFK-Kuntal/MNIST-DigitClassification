import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

# ---------------------------------------------------
# Page config
# ---------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# ---------------------------------------------------
# Load model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return keras.models.load_model("digit_classifier.keras")


model = load_model()


# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

page_style = """
<style>

div.stButton > button:first-child {
    background-color: #00cc00;
    color: white;
    font-size: 20px;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    border: none;
}

div.stButton > button:first-child:hover {
    background-color: #00aa00;
    color: white;
}

.result-box {
    padding: 15px;
    margin: 10px 0px;
    background-color: #f0f2f6;
    border-radius: 10px;
    text-align: center;
}

.result-digit {
    font-size: 60px;
    font-weight: bold;
    margin: 0px;
}

.result-confidence {
    font-size: 22px;
    margin: 5px 0px;
}

</style>
"""

st.markdown(
    page_style,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# Header
# ---------------------------------------------------

html_temp = """
<div style="
    padding: 15px;
    margin: 15px 0px;
    background-color: #ffcc00;
    border-radius: 10px;
">
    <h1 style="
        color: black;
        text-align: center;
        margin: 0px;
    ">
        🔢 MNIST Digit Classifier
    </h1>
</div>

<h4 style="
    text-align: center;
    padding: 10px 0px;
">
    Upload an image of a handwritten digit and let our CNN classify it.
</h4>
"""

st.markdown(
    html_temp,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# Main UI
# ---------------------------------------------------

col1, col2 = st.columns([1, 1])


# ---------------------------------------------------
# Left column - Image upload
# ---------------------------------------------------

with col1:

    st.markdown("### 📤 Upload Your Digit")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded digit",
            width=280
        )

    else:

        st.info(
            "Upload a PNG or JPG image containing one handwritten digit."
        )


# ---------------------------------------------------
# Right column - Prediction
# ---------------------------------------------------

with col2:

    st.markdown("### 🔍 Prediction")

    predict_btn = st.button(
        "🔍 Predict",
        use_container_width=True
    )

    st.caption(
        "Upload an image first, then click Predict."
    )

    if predict_btn:

        if uploaded_file is not None:

            # ---------------------------------------
            # Open image
            # ---------------------------------------

            img = Image.open(uploaded_file)


            # ---------------------------------------
            # Convert image to grayscale
            # ---------------------------------------

            img = img.convert("L")


            # ---------------------------------------
            # Resize to MNIST size
            # ---------------------------------------

            img = img.resize(
                (28, 28),
                Image.LANCZOS
            )


            # ---------------------------------------
            # Convert to NumPy
            # ---------------------------------------

            img_array = np.array(
                img
            ).astype("float32") / 255.0


            # ---------------------------------------
            # CNN input shape
            # ---------------------------------------

            img_array = img_array.reshape(
                1,
                28,
                28,
                1
            )


            # ---------------------------------------
            # Prediction
            # ---------------------------------------

            preds = model.predict(
                img_array,
                verbose=0
            )[0]

            digit = int(
                np.argmax(preds)
            )

            confidence = float(
                preds[digit]
            ) * 100


            # ---------------------------------------
            # Display result
            # ---------------------------------------

            result_html = f"""
            <div class="result-box">
                <p style="margin-bottom:5px;">
                    Predicted Digit
                </p>
            
                <p class="result-digit">
                    {digit}
                </p>
            
                <p class="result-confidence">
                    Confidence: {confidence:.1f}%
                </p>
            </div>
            """
            
            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            # ---------------------------------------
            # Probability distribution
            # ---------------------------------------

            st.markdown(
                "### 📊 All Probabilities"
            )

            for i, p in enumerate(preds):

                st.progress(
                    float(p),
                    text=f"{i} → {p * 100:.1f}%"
                )

        else:

            st.warning(
                "Please upload an image first!"
            )


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "Model: CNN trained on MNIST · ~99% validation accuracy"
)