import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

class PPGProcessor:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.signal_buffer = deque(maxlen=buffer_size)
        self.scaler = MinMaxScaler()
        
    def extract_ppg(self, frame):
        """Extract PPG signal from a single frame"""
        # Convert to RGB and extract red channel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        red_channel = rgb_frame[:,:,0]
        
        # Define ROI (center of the frame)
        height, width = red_channel.shape
        roi_size = min(width, height) // 3
        center_x, center_y = width // 2, height // 2
        roi = red_channel[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        
        # Calculate mean intensity in ROI
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
        
        # Calculate features (PLETH, HR, PULSE)
        pleth = np.mean(normalized_signal)
        
        # Simple HR estimation from peak detection
        peaks, _ = np.array(normalized_signal).reshape(-1), None
        peak_count = len([i for i in range(1, len(peaks)-1) if peaks[i-1] < peaks[i] > peaks[i+1]])
        hr = (peak_count * 60) / (self.buffer_size / 30)  # Assuming 30 fps
        
        # Use HR as pulse for simplicity
        pulse = hr
        
        return np.array([[pleth, hr, pulse]])

class SpO2Predictor:
    def __init__(self, model_path='cnn_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        
    def predict(self, features):
        """Predict SpO2 from PPG features"""
        features_reshaped = features.reshape(-1, 3, 1)  # Reshape for CNN input
        prediction = self.model.predict(features_reshaped)
        return float(prediction[0][0])

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    # Initialize processors
    ppg_processor = PPGProcessor()
    spo2_predictor = SpO2Predictor()
    
    # Initialize plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    line1, = ax1.plot([], [])
    ax1.set_title('PPG Signal')
    text_spo2 = ax2.text(0.5, 0.5, '', horizontalalignment='center')
    ax2.set_title('SpO2 Prediction')
    ax2.axis('off')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Extract PPG signal
            ppg_value, roi = ppg_processor.extract_ppg(frame)
            ppg_processor.signal_buffer.append(ppg_value)
            
            # Draw ROI on frame
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Process signal and predict SpO2
            features = ppg_processor.process_signal()
            if features is not None:
                spo2 = spo2_predictor.predict(features)
                
                # Update plots
                line1.set_data(range(len(ppg_processor.signal_buffer)), 
                             list(ppg_processor.signal_buffer))
                ax1.relim()
                ax1.autoscale_view()
                text_spo2.set_text(f'Predicted SpO2: {spo2:.1f}%')
                plt.pause(0.01)
                
            # Display frame
            cv2.imshow('Camera Feed', frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    main()
