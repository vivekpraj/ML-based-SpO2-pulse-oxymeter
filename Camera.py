import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Step 1: Accessing the camera and capturing an image
cap = cv2.VideoCapture(0)  # Change to 1 if you want to use the secondary camera
ret, frame = cap.read()
cap.release()

if ret:
    # Save or process the frame
    cv2.imwrite('captured_image.jpg', frame)

    # Step 2: Extract PPG signal from the captured image
    def extract_ppg_from_image(image_path):
        # Load image and process it to extract PPG signal (dummy example)
        # Replace with actual PPG extraction logic
        # For demonstration, we simulate a PPG signal with exactly 64 points
        ppg_signal = np.random.rand(64)  # Simulated PPG signal with 64 points
        return ppg_signal

    ppg_signal = extract_ppg_from_image('captured_image.jpg')

    # Step 3: Preprocess the PPG signal for prediction
    ppg_signal = ppg_signal.reshape(-1, 1)  # Reshape for scaling

    # Load scaler (ensure it's fitted on training data)
    scaler = MinMaxScaler()
    # Fit scaler on known training data range (replace with actual min/max values)
    scaler.fit(np.array([[0], [1]]))  # This should be based on your training dataset

    # Scale the PPG signal
    ppg_signal_scaled = scaler.transform(ppg_signal)

    # Prepare input for model prediction
    X_input = ppg_signal_scaled.flatten().reshape(1, -1)  # Reshape to (1, number_of_features)

    # Ensure that we have exactly 64 features for prediction
    if X_input.shape[1] != 64:
        print("Error: Input does not have the expected number of features (64).")
        exit()

    # Load the trained model (ensure you have saved it after training)
    try:
        model_cnn = load_model('cnn_model.h5')  # Update with your actual model path
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Make prediction
    try:
        predicted_spo2 = model_cnn.predict(X_input)
        print(f'Predicted SpO2: {predicted_spo2[0][0]}')
    except Exception as e:
        print(f"Error during prediction: {e}")
else:
    print("Failed to capture image.")
