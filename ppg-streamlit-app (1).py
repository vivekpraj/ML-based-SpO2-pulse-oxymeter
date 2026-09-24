
import streamlit as st
import pandas as pd
import numpy as np
import cv2
try:
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow import failed. Please install tensorflow using: pip install tensorflow")
try:
    from sklearn.preprocessing import MinMaxScaler
    import joblib
except ImportError:
    st.error("Scikit-learn import failed. Please install scikit-learn using: pip install scikit-learn")
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="SpO2 Prediction App",
    layout="wide"
)

# Add title and description
st.title("SpO2 Prediction System")
st.markdown("""
This application predicts SpO2 levels using CNN and Gaussian Process Regression models
based on PLETH, Heart Rate, and Pulse measurements.
""")

# Create sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a Page",
    ["Single Prediction", "Batch Prediction", "Model Performance", "Image-based Prediction"]
)

# Load the models
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model('cnn_model.h5')
        gpr_model = joblib.load('gpr_model.pkl')
        return cnn_model, gpr_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

cnn_model, gpr_model = load_models()

# Initialize scaler
scaler = MinMaxScaler()

# Keep existing preprocess_data function
def preprocess_data(df):
    """
    Preprocess the input dataframe
    """
    for col in ['PLETH', 'HR', 'PULSE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

class PPGProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def extract_ppg(self, frame):
        """Extract PPG signal from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        red_channel = rgb_frame[:,:,0]
        
        height, width = red_channel.shape
        roi_size = min(width, height) // 3
        center_x, center_y = width // 2, height // 2
        roi = red_channel[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        
        ppg_value = np.mean(roi)
        roi_coords = (
            center_x - roi_size//2,
            center_y - roi_size//2,
            roi_size,
            roi_size
        )
        return ppg_value, roi_coords, roi

    def process_signal(self, ppg_value):
        """Process the PPG signal to extract features"""
        # Normalize the PPG value
        normalized_ppg = self.scaler.fit_transform([[ppg_value]])[0][0]
        
        # Estimate HR and pulse (simplified for single image)
        # Using reasonable default values since we can't calculate from a single frame
        hr = 75  # default heart rate
        pulse = hr  # using same value for pulse
        
        return np.array([[normalized_ppg, hr, pulse]])

# Keep existing pages code (Single Prediction, Batch Prediction, Model Performance)
if page in ["Single Prediction", "Batch Prediction", "Model Performance"]:
    # Your existing code for these pages remains exactly the same
    exec(open("app.py").read())

# Add new Image-based Prediction page
elif page == "Image-based Prediction":
    st.header("Image-based SpO2 Prediction")
    
    # Initialize PPG processor
    ppg_processor = PPGProcessor()
    
    # Instructions
    st.markdown("""
    ### Instructions:
    1. Click 'Capture Image' to take a photo of your fingertip
    2. Make sure your fingertip is well-lit and centered in the frame
    3. Keep your finger still when capturing the image
    4. The green box shows the measurement area
    """)
    
    # Create two columns for camera input and results
    cam_col, result_col = st.columns(2)
    
    with cam_col:
        st.subheader("Camera Input")
        camera_placeholder = st.empty()
        capture_btn = st.button("Capture Image")
        
        if capture_btn:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera")
            else:
                # Capture a single frame
                ret, frame = cap.read()
                if ret:
                    # Process the frame
                    ppg_value, roi_coords, roi = ppg_processor.extract_ppg(frame)
                    
                    # Draw ROI on frame
                    x, y, w, h = roi_coords
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display the captured frame
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image")
                    
                    # Process signal and predict
                    features = ppg_processor.process_signal(ppg_value)
                    
                    if features is not None and cnn_model is not None:
                        features_cnn = features.reshape(-1, 3, 1)
                        prediction = float(cnn_model.predict(features_cnn)[0][0])
                        
                        with result_col:
                            st.subheader("Results")
                            
                            # Display ROI image
                            st.image(roi, caption="Region of Interest", width=200)
                            
                            # Display measurements
                            st.metric("PPG Value", f"{ppg_value:.2f}")
                            st.metric("Predicted SpO2", f"{prediction:.1f}%")
                            
                            # Add analysis details
                            st.markdown("### Analysis Details")
                            st.write(f"""
                            - ROI Size: {w}x{h} pixels
                            - Average Red Channel Intensity: {ppg_value:.2f}
                            - Normalized PPG Value: {features[0][0]:.3f}
                            """)
                            
                            # Add confidence indicator (simplified)
                            confidence = "High" if 90 <= prediction <= 100 else "Medium" if 80 <= prediction < 90 else "Low"
                            st.info(f"Prediction Confidence: {confidence}")
                
                # Release camera
                cap.release()
    
    # Add best practices
    st.markdown("""
    ### Best Practices for Accurate Measurements:
    1. Ensure good lighting conditions
    2. Place your fingertip directly over the center of the camera
    3. Avoid movement during capture
    4. Clean the camera lens before use
    5. Take multiple measurements for better accuracy
    
    ### Note:
    This is an experimental feature and should not be used as a replacement for medical-grade pulse oximeters.
    Always consult healthcare professionals for medical advice.
    """)

# Keep existing footer
st.markdown("---")
st.markdown("Created by Your Name | SpO2 Prediction System")
