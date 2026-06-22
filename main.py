import time
import torch
import numpy as np
from collections import deque

from acquisition import LSLAcquisition
from preprocessing import Preprocessor
from model import EEGNet
from control import HandController

def main():
    print("=== Real-time EEG MI Robotic Hand Control System ===")
    
    # 1. Configuration
    # Updated based on scan: "CGX Quick-20r Q20r-0162"
    stream_name = "CGX Quick-20r Q20r-0162" 
    window_duration = 1.0 
    update_interval = 0.1 
    srate = 500 # Updated from 250
    n_channels = 26 # Updated from 20
    
    
    # 2. Initialization
    # Acquisition
    acq = LSLAcquisition(stream_name=stream_name)
    try:
        acq.connect()
    except Exception as e:
        print(f"Error connecting to LSL: {e}")
        print("Make sure 'mock_stream.py' is running or headset is connected.")
        return

    # Preprocessing
    prep = Preprocessor(sfreq=srate, n_channels=n_channels)
    
    # Model
    # Note: In a real scenario, load weights here: model.load_state_dict(torch.load('model.pth'))
    print("Initializing EEGNet model (Random Weights)...")
    model = EEGNet(nb_classes=2, Chans=n_channels, Samples=int(window_duration*srate))
    model.eval()
    
    # Control
    ctrl = HandController()
    
    # Smoothing Buffer
    prediction_history = deque(maxlen=3)
    
    print("Starting Main Control Loop...")
    try:
        while True:
            cycle_start = time.time()
            
            # A. Update Buffer
            acq.update()
            
            # Check if we have enough data for a full window
            if acq.samples_in_buffer < int(window_duration * srate):
                # Initial buffering
                time.sleep(0.1)
                continue
                
            # B. Get Data
            try:
                raw_data = acq.get_latest_window(window_duration)
            except ValueError:
                # Buffer might not be full yet if using rigid pointer math
                continue
                
            # C. Preprocess
            input_tensor = prep.prepare_input(raw_data)
            input_tensor = torch.from_numpy(input_tensor)
            
            # D. Inference
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
                
            # E. Logic & Control (Class 0: Rest/Open, Class 1: Close)
            prediction_history.append(pred_class)
            
            # Simple voting voting
            if sum(prediction_history) == 3: # All 1s
                ctrl.send_command("CLOSE")
            elif sum(prediction_history) == 0 and len(prediction_history) == 3: # All 0s
                ctrl.send_command("OPEN")
            
            # Logging
            # print(f"Pred: {pred_class} ({confidence:.2f}) | History: {list(prediction_history)}")
            
            # F. Timing
            cycle_duration = time.time() - cycle_start
            sleep_time = max(0, update_interval - cycle_duration)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
