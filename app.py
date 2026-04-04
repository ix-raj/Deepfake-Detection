import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deepfake Detection AI", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure model.pkl is in your GitHub root folder
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# --- IMAGE PROCESSING WITH INTERPOLATION ---
def extract_power_spectrum(image):
    # 1. Convert to grayscale and resize
    img_array = np.array(image.convert('L'))
    img_resized = cv2.resize(img_array, (128, 128)) 
    
    # 2. Fourier Transform
    f = np.fft.fft2(img_resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    
    # 3. Flatten the 2D spectrum (16,384 points)
    flat_features = magnitude_spectrum.flatten()
    
    # 4. INTERPOLATION: Compress 16,384 points into exactly 722
    x_old = np.linspace(0, 1, len(flat_features))
    x_new = np.linspace(0, 1, 722)
    f_interp = interp1d(x_old, flat_features, kind='linear')
    
    features = f_interp(x_new).reshape(1, -1)
    return features, magnitude_spectrum

# --- MAIN UI ---
st.title("🛡️ Deepfake Detection Dashboard")
st.write("Analyzing images for frequency artifacts using a 60% confidence threshold.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Run Forensic Analysis"):
        try:
            with st.spinner('Processing Pixels...'):
                features, spectrum_viz = extract_power_spectrum(image)
                
                # Get the probability for each class
                # prob[0][1] is the confidence for 'Fake'
                prob = pipeline.predict_proba(features)
                fake_confidence = prob[0][1] * 100 

            with col2:
                st.subheader("Analysis Results")
                
                # --- NEW THRESHOLD LOGIC ---
                # Default ML uses 50%. We use 60% to reduce false 'Fake' flags
                if fake_confidence > 60:
                    st.error(f"### Result: FAKE / AI-GENERATED")
                    st.write(f"Confidence: {fake_confidence:.2f}%")
                else:
                    st.success(f"### Result: REAL / AUTHENTIC")
                    st.write(f"Confidence: {(100 - fake_confidence):.2f}%")
                
                # Visualization
                st.write("**Frequency Power Spectrum:**")
                fig, ax = plt.subplots()
                ax.imshow(spectrum_viz, cmap='magma')
                ax.axis('off')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Analysis failed. Error: {e}")