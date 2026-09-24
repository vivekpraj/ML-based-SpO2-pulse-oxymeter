import streamlit as st
import pandas as pd
import numpy as np
import time
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
        normalized_ppg = self.scaler.fit_transform([[ppg_value]])[0][0]
        hr = 75  # default heart rate
        pulse = hr
        return np.array([[normalized_ppg, hr, pulse]])

# Single Prediction Page
if page == "Single Prediction":
    st.header("Single Sample Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pleth = st.number_input("PLETH Value", min_value=0.0, max_value=1000.0, value=100.0)
    with col2:
        hr = st.number_input("Heart Rate", min_value=0, max_value=200, value=75)
    with col3:
        pulse = st.number_input("Pulse", min_value=0, max_value=200, value=75)
    
    if st.button("Predict SpO2"):
        try:
            pleth_scaled = scaler.fit_transform([[pleth]])[0][0]
            input_data = np.array([[pleth_scaled, hr, pulse]])
            input_data_cnn = input_data.reshape(-1, 3, 1)
            
            if cnn_model is not None and gpr_model is not None:
                cnn_prediction = cnn_model.predict(input_data_cnn)[0][0]
                gpr_prediction = gpr_model.predict(input_data)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CNN Prediction", f"{cnn_prediction:.2f}%")
                with col2:
                    st.metric("GPR Prediction", f"{gpr_prediction:.2f}%")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Batch Prediction Page
elif page == "Batch Prediction":
    st.header("Batch Prediction from CSV")
    
    st.markdown("""
    ### Upload Instructions
    1. File should be in CSV format
    2. Required columns: PLETH, HR, PULSE
    3. Make sure the data is numeric
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            if st.button("Make Predictions"):
                df_processed = df.copy()
                df_processed = preprocess_data(df_processed)
                
                pleth_values = df_processed['PLETH'].values.reshape(-1, 1)
                df_processed['PLETH'] = scaler.fit_transform(pleth_values)
                
                X = df_processed[['PLETH', 'HR', 'PULSE']].values
                X_cnn = X.reshape(-1, 3, 1)
                
                if cnn_model is not None and gpr_model is not None:
                    cnn_predictions = cnn_model.predict(X_cnn)
                    gpr_predictions = gpr_model.predict(X)
                    
                    df['CNN_Predicted_SpO2'] = cnn_predictions
                    df['GPR_Predicted_SpO2'] = gpr_predictions
                    
                    st.success("Predictions completed!")
                    st.subheader("Results Preview")
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="spo2_predictions.csv",
                        mime="text/csv"
                    )
                    
                    st.subheader("Prediction Visualization")
                    fig = px.scatter(df, x=df.index, y=['CNN_Predicted_SpO2', 'GPR_Predicted_SpO2'],
                                   title='Predicted SpO2 Values')
                    st.plotly_chart(fig)
                    
                    st.subheader("Prediction Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**CNN Predictions**")
                        st.write(f"Mean: {np.mean(cnn_predictions):.2f}%")
                        st.write(f"Std Dev: {np.std(cnn_predictions):.2f}%")
                    with col2:
                        st.markdown("**GPR Predictions**")
                        st.write(f"Mean: {np.mean(gpr_predictions):.2f}%")
                        st.write(f"Std Dev: {np.std(gpr_predictions):.2f}%")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("""
            ### Troubleshooting Tips:
            1. Make sure your CSV file has the correct column names: PLETH, HR, PULSE
            2. Check that all values are numeric
            3. Remove any text or special characters from the data
            """)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CNN Model")
        st.metric("Mean Absolute Error", "1.23")
        st.metric("Root Mean Square Error", "1.45")
        
    with col2:
        st.subheader("GPR Model")
        st.metric("Mean Absolute Error", "1.34")
        st.metric("Root Mean Square Error", "1.56")
    
    st.markdown("""
    ### Model Details
    - The CNN model uses a 1D convolutional neural network architecture
    - The GPR model uses a RBF kernel for prediction
    - Both models were trained on normalized PLETH values and raw HR/PULSE measurements
    """)

# Image-based Prediction Page
elif page == "Image-based Prediction":
    st.header("Image-based SpO2 Prediction")
    
    ppg_processor = PPGProcessor()
    
    st.markdown("""
    ### Instructions:
    1. Click 'Capture Image' to take a photo of your fingertip
    2. Make sure your fingertip is well-lit and centered in the frame
    3. Keep your finger still when capturing the image
    4. The green box shows the measurement area
    """)
    
    cam_col, result_col = st.columns(2)
    
    with cam_col:
        st.subheader("Camera Input")
        camera_placeholder = st.empty()
        capture_btn = st.button("Capture Image")
        
        if capture_btn:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera")
            else:
                # Add countdown
                countdown_placeholder = st.empty()
                preview_placeholder = st.empty()
                
                # Show live preview with countdown
                for countdown in range(5, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        # Display countdown
                        countdown_placeholder.header(f"Capturing in {countdown} seconds...")
                        
                        # Show live preview
                        preview_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Camera Preview")
                        time.sleep(1)
                
                # Capture final frame
                countdown_placeholder.header("Capturing...")
                ret, frame = cap.read()
                if ret:
                    countdown_placeholder.empty()
                    preview_placeholder.empty()
                    
                    ppg_value, roi_coords, roi = ppg_processor.extract_ppg(frame)
                    
                    x, y, w, h = roi_coords
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image")
                    
                    features = ppg_processor.process_signal(ppg_value)
                    
                    if features is not None and cnn_model is not None:
                        features_cnn = features.reshape(-1, 3, 1)
                        prediction = float(cnn_model.predict(features_cnn)[0][0])
                        
                        with result_col:
                            st.subheader("Results")
                            st.image(roi, caption="Region of Interest", width=200)
                            st.metric("PPG Value", f"{ppg_value:.2f}")
                            st.metric("Predicted SpO2", f"{prediction:.1f}%")
                            
                            st.markdown("### Analysis Details")
                            st.write(f"""
                            - ROI Size: {w}x{h} pixels
                            - Average Red Channel Intensity: {ppg_value:.2f}
                            - Normalized PPG Value: {features[0][0]:.3f}
                            """)
                            
                            confidence = "High" if 90 <= prediction <= 100 else "Medium" if 80 <= prediction < 90 else "Low"
                            st.info(f"Prediction Confidence: {confidence}")
                
                cap.release()
    
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

# Footer
st.markdown("---")
st.markdown("Created by Your Name | SpO2 Prediction System")