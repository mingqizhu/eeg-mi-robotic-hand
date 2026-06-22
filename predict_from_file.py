import mne
import numpy as np
import torch
import time
import os
import glob
from collections import deque
from statistics import mode
from model import EEGNet
from preprocessing import Preprocessor

# Configuration defaults
MODEL_PATH = 'best_model.pth'
UPDATE_INTERVAL = 0.4  # Matches realtime_predict.py
BUFFER_DURATION = 4.0
VOTE_WINDOW_SIZE = 5

def predict_existed_data(fif_path, scaler_path, model_path=MODEL_PATH):
    """
    Replays a .fif file through the prediction pipeline.
    """
    print(f"\n=== Offline Prediction Replay ===")
    print(f"File: {fif_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Model: {model_path}")
    
    # 1. Load Data
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True)
    except Exception as e:
        print(f"Error loading FIF file: {e}")
        return

    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    print(f"Loaded Data: {n_channels} channels @ {sfreq} Hz, Duration: {raw.times[-1]:.2f}s")
    
    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples_per_epoch = int(BUFFER_DURATION * sfreq)
    
    model = EEGNet(nb_classes=3, Chans=n_channels, Samples=samples_per_epoch)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load Scaler
    try:
        scaler = np.load(scaler_path)
        scaler_mean = scaler['mean']
        scaler_std = scaler['std']
        print(f"Scaler loaded: Mean={scaler_mean:.3f}, Std={scaler_std:.3f}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return

    # 4. Initialize Preprocessor & ASR
    preprocessor = Preprocessor(sfreq=sfreq, n_channels=n_channels)
    
    # Auto-calibrate ASR on the first 5 seconds of the file (Simulating 'Rest' phase)
    # In a real scenario, we might want to use specific Rest markers, but for general replay, 
    # we assume the start is resting or we just take the first chunk.
    print("[Calibration] Using first 5 seconds for ASR calibration...")
    calib_duration = 5.0
    calib_samples = int(calib_duration * sfreq)
    if raw.n_times > calib_samples:
        calib_data = raw.get_data(start=0, stop=calib_samples)
        preprocessor.calibrate_asr(calib_data)
    else:
        print("Warning: File too short for calibration. Skipping ASR.")

    # 5. Replay Loop
    total_samples = raw.n_times
    step_samples = int(UPDATE_INTERVAL * sfreq)
    buffer_samples = int(BUFFER_DURATION * sfreq)
    
    # We start modifying the buffer from the beginning.
    # To simulate real-time, we step through the file.
    
    vote_window = deque(maxlen=VOTE_WINDOW_SIZE)
    full_data = raw.get_data() # (n_channels, n_times)
    
    print("\n--- Starting Replay ---")
    current_idx = 0
    
    # We need enough data for one buffer
    current_idx = buffer_samples 
    
    while current_idx < total_samples:
        # Extract buffer: [current_idx - buffer_samples : current_idx]
        start_idx = current_idx - buffer_samples
        end_idx = current_idx
        
        data_chunk = full_data[:, start_idx:end_idx]
        
        # --- Processing Pipeline (Matches realtime_predict.py) ---
        # 1. ASR
        data_proc = preprocessor.apply_asr(data_chunk)
        
        # 2. Filter
        data_proc = preprocessor.filter_data(data_proc)
        
        # 3. Laplacian
        data_proc = preprocessor.apply_laplacian(data_proc, raw.ch_names)
        
        # 4. Scale
        data_proc = (data_proc - scaler_mean) / (scaler_std + 1e-8)
        
        # 5. Inference
        input_tensor = torch.tensor(data_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            raw_label = predicted.item()
            
        # 6. Voting
        vote_window.append(raw_label)
        if len(vote_window) > 0:
            try:
                final_decision = mode(vote_window)
            except:
                final_decision = raw_label
        else:
            final_decision = raw_label
            
        # 7. Output
        timestamp = current_idx / sfreq
        vote_str = f"Win:{list(vote_window)}"
        
        action_str = ""
        if final_decision == 1:
            action_str = "GRASP (Right Hand)"
        elif final_decision == 0:
            action_str = "REST (Open)"
        else:
            action_str = "NULL (Ignore)"
            
        print(f"[{timestamp:6.1f}s] Raw:{raw_label} -> Vote:{final_decision} | {action_str} | {vote_str}")
        
        # Step forward
        current_idx += step_samples
        
        # Optional: Simulate real-time speed?
        # time.sleep(UPDATE_INTERVAL) 
        
    print("\nReplay Finished.")

def main():
    # Helper to find files
    fif_files = glob.glob("*.fif")
    fif_files.sort()
    
    print("Available FIF files:")
    for i, f in enumerate(fif_files):
        print(f"{i+1}: {f}")
        
    choice = input("\nSelect file number to replay: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(fif_files):
        print("Invalid selection.")
        return
        
    target_fif = fif_files[int(choice)-1]
    
    # Check for scaler
    if os.path.exists("scaler_params.npz"):
        target_scaler = "scaler_params.npz"
    else:
        target_scaler = input("Enter path to scaler_params.npz: ").strip()
        
    predict_existed_data(target_fif, target_scaler)

if __name__ == "__main__":
    main()
