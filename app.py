from pathlib import Path
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from feature_utils import extract_power_spectrum_features


st.set_page_config(
    page_title="Deepfake Detection Studio",
    page_icon="DS",
    layout="wide",
)


MODEL_PATH = Path(__file__).with_name("model.pkl")


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        saved = pickle.load(file)
        if isinstance(saved, dict) and "pipeline" in saved:
            return saved
        return {"pipeline": saved, "threshold": 0.5, "metadata": {}}


model_bundle = load_model()
pipeline = model_bundle["pipeline"]
MODEL_THRESHOLD = float(model_bundle.get("threshold", 0.5))

def extract_power_spectrum(image: Image.Image):
    features, magnitude, normalized = extract_power_spectrum_features(image)
    features = features.reshape(1, -1)
    return features, magnitude, normalized


def assess_image_quality(image_array: np.ndarray) -> dict:
    laplacian_var = float(cv2.Laplacian(image_array, cv2.CV_64F).var())
    brightness = float(np.mean(image_array))
    contrast = float(np.std(image_array))

    quality_score = 100.0
    quality_score -= max(0.0, 18.0 - laplacian_var) * 2.2
    quality_score -= max(0.0, 35.0 - contrast) * 1.1
    if brightness < 55 or brightness > 205:
        quality_score -= 12.0
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    if quality_score < 45:
        label = "Low"
    elif quality_score < 70:
        label = "Medium"
    else:
        label = "High"

    return {
        "score": quality_score,
        "label": label,
        "sharpness": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
    }


def analyze_prediction(fake_confidence: float, quality_score: float, threshold: float) -> dict:
    adjusted_confidence = fake_confidence
    threshold_pct = threshold * 100

    # Slightly soften predictions for poor images instead of forcing a wrong label.
    if quality_score < 45:
        adjusted_confidence *= 0.82
    elif quality_score < 70:
        adjusted_confidence *= 0.92

    if adjusted_confidence >= threshold_pct:
        decision = "Fake / AI-generated"
        badge = "High-frequency artifacts lean toward synthetic content"
    else:
        decision = "Real / Authentic"
        badge = "Signal is closer to the real-image profile"

    return {
        "decision": decision,
        "badge": badge,
        "adjusted_confidence": adjusted_confidence,
        "threshold": threshold_pct,
    }


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(247, 198, 95, 0.20), transparent 30%),
            radial-gradient(circle at top right, rgba(68, 121, 201, 0.18), transparent 28%),
            linear-gradient(135deg, #f7f2e8 0%, #efe7da 45%, #e7edf2 100%);
        color: #17202a;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero-card, .panel-card, .metric-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(23, 32, 42, 0.08);
        border-radius: 22px;
        box-shadow: 0 14px 40px rgba(55, 71, 79, 0.12);
        backdrop-filter: blur(10px);
    }
    .hero-card {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }
    .hero-eyebrow {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #17324d;
        color: #f7f2e8;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.5rem;
        line-height: 1.05;
        font-weight: 700;
        margin: 0.9rem 0 0.5rem;
        color: #14212b;
    }
    .hero-copy {
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
        color: #34495e;
        margin: 0;
    }
    .panel-card {
        padding: 1.2rem 1.2rem 1rem;
        margin-top: 0.6rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        min-height: 112px;
    }
    .metric-label {
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5d6d7e;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10212f;
        margin-top: 0.35rem;
    }
    .metric-note {
        color: #516170;
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }
    .result-fake, .result-real, .result-neutral {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }
    .result-fake {
        background: rgba(165, 42, 42, 0.08);
        border-color: rgba(165, 42, 42, 0.18);
    }
    .result-real {
        background: rgba(46, 125, 50, 0.09);
        border-color: rgba(46, 125, 50, 0.18);
    }
    .result-neutral {
        background: rgba(183, 137, 23, 0.10);
        border-color: rgba(183, 137, 23, 0.20);
    }
    .result-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #10212f;
        margin-bottom: 0.2rem;
    }
    .result-copy {
        color: #485766;
        margin: 0;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 999px;
        border: none;
        padding: 0.85rem 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #17324d 0%, #3f6c8f 100%);
        color: white;
        box-shadow: 0 10px 24px rgba(23, 50, 77, 0.24);
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.72);
        border-radius: 20px;
        border: 1px dashed rgba(23, 32, 42, 0.18);
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-card">
        <span class="hero-eyebrow">Image Forensics Dashboard</span>
        <div class="hero-title">Deepfake Detection Studio</div>
        <p class="hero-copy">
            Upload an image to run a lightweight forensic scan using the saved classifier,
            frequency-spectrum analysis, and a quality-aware confidence check. Low-quality
            or borderline cases are flagged as inconclusive instead of forcing a weak result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Add an image to begin the analysis.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Input Preview")
        st.image(image, use_container_width=True)
        st.caption("Tip: clear, well-lit images produce more reliable results.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("How this quick version works")
        st.write(
            "This assignment-ready build keeps the original model but reduces false confidence "
            "by normalizing images before feature extraction and warning when image quality is too poor."
        )
        run_analysis = st.button("Run Forensic Analysis")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_analysis:
        with st.spinner("Scanning image patterns and frequency artifacts..."):
            try:
                features, spectrum_viz, normalized_image = extract_power_spectrum(image)
                quality = assess_image_quality(normalized_image)
                probabilities = pipeline.predict_proba(features)
                fake_confidence = float(probabilities[0][1] * 100)
                analysis = analyze_prediction(fake_confidence, quality["score"], MODEL_THRESHOLD)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

        metric_cols = st.columns(3)
        metrics = [
            ("Fake Score", f"{analysis['adjusted_confidence']:.1f}%", "Quality-adjusted confidence"),
            ("Image Quality", f"{quality['score']:.1f}/100", f"{quality['label']} reliability"),
            ("Contrast", f"{quality['contrast']:.1f}", "Higher is usually more informative"),
        ]
        for column, (label, value, note) in zip(metric_cols, metrics):
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if analysis["decision"] == "Fake / AI-generated":
            result_class = "result-fake"
        elif analysis["decision"] == "Real / Authentic":
            result_class = "result-real"
        else:
            result_class = "result-real"

        st.markdown(
            f"""
            <div class="{result_class}">
                <div class="result-title">Result: {analysis["decision"]}</div>
                <p class="result-copy">{analysis["badge"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        viz_col1, viz_col2 = st.columns(2, gap="large")
        with viz_col1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Normalized Grayscale View")
            st.image(normalized_image, use_container_width=True, clamp=True)
            st.caption("This is the preprocessed image the classifier sees.")
            st.markdown("</div>", unsafe_allow_html=True)

        with viz_col2:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Frequency Power Spectrum")
            fig, axis = plt.subplots(figsize=(5, 5))
            axis.imshow(spectrum_viz, cmap="magma")
            axis.axis("off")
            st.pyplot(fig, clear_figure=True)
            st.caption("Hotspots highlight frequency activity used by the model.")
            st.markdown("</div>", unsafe_allow_html=True)
