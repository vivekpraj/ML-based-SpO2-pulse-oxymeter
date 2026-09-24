import streamlit as st
import pandas as pd
import numpy as np
import cv2
from collections import deque
import plotly.graph_objects as go
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

# Create sidebar for navigation with added Real-time option
page = st.sidebar.selectbox(
    "Choose a Page",
    ["Single Prediction", "Batch Prediction", "Model Performance", "Real-time Prediction"]
)

# Load the models (keeping existing code)
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

# Add PPG Processing class
class PPGProcessor:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.signal_buffer = deque(maxlen=buffer_size)
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
        return ppg_value, (
            center_x - roi_size//2,
            center_y - roi_size//2,
            roi_size,
            roi_size
        )

    def process_signal(self):
        """Process the buffered signal to extract features"""
        if len(self.signal_buffer) < self.buffer_size:
            return None
            
        signal = np.array(self.signal_buffer).reshape(-1, 1)
        normalized_signal = self.scaler.fit_transform(signal)
        
        pleth = np.mean(normalized_signal)
        peaks = np.array(normalized_signal).reshape(-1)
        peak_count = len([i for i in range(1, len(peaks)-1) if peaks[i-1] < peaks[i] > peaks[i+1]])
        hr = (peak_count * 60) / (self.buffer_size / 30)
        
        pulse = hr
        
        return np.array([[pleth, hr, pulse]])

# Keep existing pages code (Single Prediction, Batch Prediction, Model Performance)
if page in ["Single Prediction", "Batch Prediction", "Model Performance"]:
    # Your existing code for these pages remains exactly the same
    exec(open("app.py").read())

# Add new Real-time Prediction page
elif page == "Real-time Prediction":
    st.header("Real-time Camera-based SpO2 Prediction")
    
    # Initialize session state for PPG processor if not exists
    if 'ppg_processor' not in st.session_state:
        st.session_state.ppg_processor = PPGProcessor()
        st.session_state.signal_values = []
        st.session_state.predictions = []
    
    # Camera input
    camera_col, plot_col = st.columns(2)
    
    with camera_col:
        camera_placeholder = st.empty()
        start_button = st.button("Start/Stop Camera")
    
    with plot_col:
        signal_plot = st.empty()
        prediction_text = st.empty()
    
    if start_button:
        if 'camera' not in st.session_state:
            st.session_state.camera = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Extract and process PPG signal
                ppg_value, roi = st.session_state.ppg_processor.extract_ppg(frame)
                st.session_state.ppg_processor.signal_buffer.append(ppg_value)
                st.session_state.signal_values.append(ppg_value)
                
                # Draw ROI on frame
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display frame
                camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Process signal and predict SpO2
                features = st.session_state.ppg_processor.process_signal()
                if features is not None and cnn_model is not None:
                    features_cnn = features.reshape(-1, 3, 1)
                    prediction = float(cnn_model.predict(features_cnn)[0][0])
                    st.session_state.predictions.append(prediction)
                    
                    # Update plots
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=list(st.session_state.ppg_processor.signal_buffer),
                                           mode='lines',
                                           name='PPG Signal'))
                    fig.update_layout(title='PPG Signal',
                                    xaxis_title='Sample',
                                    yaxis_title='Intensity')
                    signal_plot.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction
                    prediction_text.metric("Predicted SpO2", f"{prediction:.1f}%")
                
                # Check for stop button
                if not start_button:
                    break
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if 'camera' in st.session_state:
                st.session_state.camera.release()
                del st.session_state.camera

    # Add instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'Start/Stop Camera' button to begin capture
    2. Position your fingertip over the camera lens
    3. Keep your finger still for accurate measurements
    4. The green box shows the measurement area
    5. Click the button again to stop
    
    ### Notes:
    - Ensure good lighting for accurate measurements
    - The prediction may take a few seconds to stabilize
    - If you see errors, try adjusting your finger position
    """)

# Keep existing footer
st.markdown("---")
st.markdown("Created by Your Name | SpO2 Prediction System")
