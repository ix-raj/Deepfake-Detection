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
    # Loading the pipeline saved from model_train.py
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# --- IMPROVED IMAGE PROCESSING ---
def extract_power_spectrum(image):
    # 1. Convert to grayscale and resize for consistent FFT calculation
    img_array = np.array(image.convert('L'))
    img_resized = cv2.resize(img_array, (128, 128)) 
    
    # 2. Fourier Transform to get Frequency Domain
    f = np.fft.fft2(img_resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    
    # 3. Flatten the 2D spectrum into 1D
    flat_features = magnitude_spectrum.flatten()
    
    # 4. INTERPOLATION (The Fix)
    # Instead of cutting off data, we compress 16384 points into 722 points
    x_old = np.linspace(0, 1, len(flat_features))
    x_new = np.linspace(0, 1, 722)
    f_interp = interp1d(x_old, flat_features, kind='linear')
    
    # This results in exactly 722 features for the StandardScaler
    features = f_interp(x_new).reshape(1, -1)
    
    return features, magnitude_spectrum

# --- MAIN UI ---
st.title("🛡️ Deepfake Detection Dashboard")
st.write("Upload an image to analyze frequency artifacts common in AI generation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Run Forensic Analysis"):
        try:
            with st.spinner('Calculating Power Spectrum...'):
                features, spectrum_viz = extract_power_spectrum(image)
                
                # Predict using the Random Forest pipeline
                prediction = pipeline.predict(features)
                probability = pipeline.predict_proba(features)
            
            with col2:
                st.subheader("Analysis Results")
                
                if prediction[0] == 1:
                    st.error("### Result: FAKE / AI-GENERATED")
                else:
                    st.success("### Result: REAL / AUTHENTIC")
                
                st.metric("Confidence", f"{np.max(probability)*100:.2f}%")
                
                # Visualization of the Frequency Domain
                st.write("**Power Spectrum Visualization:**")
                fig, ax = plt.subplots()
                ax.imshow(spectrum_viz, cmap='magma')
                ax.axis('off')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Analysis Error: {e}")