import mne
import numpy as np
import torch
import time
import os
import glob
import sys
import serial
from collections import deque, Counter
from model import EEGNet
from preprocessing import Preprocessor
import logging

# --- Configuration ---
MODEL_PATH = "best_model_70acc.pth"
SCALER_PATH = "scaler_params.npz" # Will try to auto-find based on file name or use default
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Timing & Buffer
BUFFER_DURATION = 4.0   # 4 seconds memory for model input
PREDICTION_INTERVAL = 0.2 # 200 ms between predictions
ASR_CALIB_DURATION = 10.0 # Standard calibration duration

# Voting Filter Settings (Tuned)
VOTE_WINDOW = 8
VOTE_THRESHOLD = 5

class VotingFilter:
    def __init__(self, window_size=VOTE_WINDOW, threshold=VOTE_THRESHOLD):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
        self.REST = 0
        self.GRASP = 1
        self.NULL = 2
        
    def update(self, prediction):
        self.window.append(prediction)
        counts = Counter(self.window)
        
        # Decision Logic: Return Class if count >= Threshold, else None
        if counts[self.GRASP] >= self.threshold:
            return self.GRASP
        elif counts[self.REST] >= self.threshold:
            return self.REST
        else:
            return None # Unstable / Null

class SerialCommander:
    def __init__(self, port, baudrate):
        self.ser = None
        self.port = port
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            print(f"[Serial] Connected to {port}")
        except Exception as e:
            print(f"[Serial] Warning: Could not connect to {port}. ({e})")
            
    def send(self, command_char):
        if self.ser:
            try:
                self.ser.write(command_char.encode())
            except Exception as e:
                print(f"[Serial] Send Error: {e}")
                
    def close(self):
        if self.ser:
            self.ser.close()

def control_from_file(fif_path):
    print(f"\n=== Control Simulation from File: {fif_path} ===")
    
    # 1. Load Data
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error loading {fif_path}: {e}")
        return

    # Extract Events for ASR Calibration
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Standardize event IDs
    # Our data usually has: Rest:1, Grasp:2, Null:3 (from collect_data.py)
    # But sometimes event dict might differ.
    # Let's try to map 'Rest' from annotations if available.
    rest_id = None
    if event_id:
        for k, v in event_id.items():
            if 'Rest' in k or 'rest' in k:
                rest_id = v
                break
    
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    print(f"[System] Data loaded: {n_channels} ch @ {sfreq}Hz, {raw.times[-1]:.1f}s")
    
    # 2. Setup Resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples_model = int(BUFFER_DURATION * sfreq)
    
    model = EEGNet(nb_classes=3, Chans=n_channels, Samples=samples_model)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"[System] Model {MODEL_PATH} loaded.")
    except Exception as e:
        print(f"[System] Error loading model: {e}")
        return

    # Preprocessor
    preprocessor = Preprocessor(sfreq=sfreq, n_channels=n_channels)
    
    # Scaler
    scaler_mean = 0
    scaler_std = 1
    if os.path.exists(SCALER_PATH):
        s = np.load(SCALER_PATH)
        scaler_mean = s['mean']
        scaler_std = s['std']
        print("[System] Scaler loaded.")
    else:
        print("[System] Warning: No scaler found. Using 0/1.")

    # Commander
    commander = SerialCommander(SERIAL_PORT, BAUD_RATE)
    
    # Voter
    voting = VotingFilter()

    # 3. ASR Calibration Logic (Robust)
    print("[ASR] searching for calibration data...")
    # Strategy: Find first 'Rest' event window >= 10s? Or just accumulate Rest periods?
    # User asked: "auto search first 10s Rest data to force fit".
    # We will look for 10s worth of data marked as Rest.
    
    calib_data = None
    
    if rest_id is not None:
        # Find all Rest events
        rest_events = events[events[:, 2] == rest_id]
        
        # Accumulate up to 10s
        accumulated_data = []
        accumulated_samples = 0
        target_samples = int(ASR_CALIB_DURATION * sfreq)
        
        data_full = raw.get_data()
        
        for ev in rest_events:
            start = ev[0]
            # How long is this event? Annotations usually have duration, but events_from_annot gives onsets.
            # We need the raw annotations to know duration.
            # Let's approximate or check raw.annotations directly.
            pass
        
        # Using annotations directly is better for duration
        annot = raw.annotations
        current_samples = 0
        
        for i, desc in enumerate(annot.description):
            if 'Rest' in desc or 'rest' in desc:
                onset = annot.onset[i]
                duration = annot.duration[i]
                
                # Extract
                start_idx = raw.time_as_index(onset)[0]
                n_samp = raw.time_as_index(duration)[0] # duration is in seconds
                
                # Careful with bounds
                end_idx = min(start_idx + n_samp, data_full.shape[1])
                
                chunk = data_full[:, start_idx:end_idx]
                accumulated_data.append(chunk)
                current_samples += chunk.shape[1]
                
                if current_samples >= target_samples:
                    break
        
        if current_samples > 0:
            calib_data = np.concatenate(accumulated_data, axis=1)
            # Trim to target if needed, but more is fine
            if calib_data.shape[1] > target_samples:
                calib_data = calib_data[:, :target_samples]
            print(f"[ASR] Found {current_samples/sfreq:.1f}s of Rest data.")
        else:
             # Fallback: check if the first 10s of file is generally usable? 
             # No, if we can't find explicitly marked Rest, we shouldn't guess for ASR.
             pass
             
    else:
        # No Rest ID found.
        # Fallback: Try to use the very first 10s of the file, assuming it starts with Rest (common protocol)
        # But only if user didn't specify strictness. User said "search for".
        # Let's try to assume first 10s is rest if no annotations found.
        if len(annot) == 0: 
             print("[ASR] No annotations found. Assuming first 10s is Rest.")
             calib_data = raw.get_data(start=0, stop=int(10*sfreq))
        else:
             print("[ASR] Annotations exist but no 'Rest' found.")

    # Perform Calibration
    if calib_data is not None and calib_data.shape[1] >= int(5 * sfreq): # At least 5s
        try:
            preprocessor.calibrate_asr(calib_data)
            print("[ASR] Calibration Complete.")
        except Exception as e:
             print(f"[ASR] Calibration Failed: {e}")
             # Disable verbose logging from mne/asr if possible?
    else:
        print("[ASR] WARNING: Could not find 10s of Rest data. ASR Disabled.")
        # Suppress warnings
        logging.getLogger('mne').setLevel(logging.ERROR)
        preprocessor.asr_enabled = False # We need to ensure logic handles this
        # Actually our Preprocessor class needs to know not to apply ASR.
        # Check Preprocessor: it has `apply_asr`. If state mapping is not set, it might error or do nothing.
        # We'll just define a flag.
        DISABLE_ASR = True

    # 4. Main Simulation Loop
    print("\n[System] STARTING SIMULATION")
    print("Printing '.' every second. Large status change will be printed below.")
    
    total_samples = raw.n_times
    step_samples = int(PREDICTION_INTERVAL * sfreq) # 200ms step
    
    # Needs 4s buffer to start
    current_ptr = samples_model
    
    full_data = raw.get_data()
    
    # State tracking
    last_action_str = "None"
    last_print_time = 0
    start_time = time.time()
    
    grasp_count = 0
    
    # Progress dot timer
    dot_timer = 0
    
    try:
        while current_ptr < total_samples:
            # 1. Get Chunk (simulate buffer)
            # Ideally we extract exactly SAMPLES_MODEL ending at current_ptr
            chunk = full_data[:, current_ptr-samples_model : current_ptr]
            
            # Copy
            proc_data = chunk.copy()
            
            # 2. Process
            # Filter
            proc_data = preprocessor.filter_data(proc_data)
            
            # Lap
            proc_data = preprocessor.apply_laplacian(proc_data, raw.ch_names)
            
            # ASR
            try:
                # If calibration failed, this might raise error if we didn't disable it inside preprocessor.
                # Let's wrap it.
                proc_data = preprocessor.apply_asr(proc_data)
            except:
                pass # Skip ASR if it fails
                
            # Norm
            proc_data = (proc_data - scaler_mean) / (scaler_std + 1e-6)
            
            # 3. Model
            t1 = time.time()
            x = torch.tensor(proc_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                
            # 4. Vote
            final_decision = voting.update(pred_idx)
            
            # 5. Action
            new_action_str = last_action_str
            
            if final_decision == voting.GRASP:
                if last_action_str != "GRASP":
                    print(f"\n[! CHANGE !] >>> GRASP DETECTED <<< (Conf: {probs[1]:.2f})")
                    commander.send('G')
                    new_action_str = "GRASP"
                    grasp_count += 1
            elif final_decision == voting.REST:
                if last_action_str != "REST":
                    print(f"\n[Status] -> REST (Conf: {probs[0]:.2f})")
                    commander.send('R')
                    new_action_str = "REST"
            
            last_action_str = new_action_str
            
            # 6. Progress Dot
            current_sim_time = current_ptr / sfreq
            if current_sim_time - dot_timer > 1.0:
                sys.stdout.write(".")
                sys.stdout.flush()
                dot_timer = current_sim_time
                
            # 7. Step
            current_ptr += step_samples
            
            # Optional: Simulate Real Time Delay?
            # time.sleep(0.01) # Small delay to not blast through 1hr file in 1sec, but maybe User wants fast result?
            # User said "benchmark realtime control", usually implies checking speed or just logic correctness.
            # "use old file as real-time signal to test servo" -> Implies we should drive servo at real speed?
            # Or just check if servo moves.
            # If we run too fast using a file, the servo might get flooded with commands (though we filter repetition).
            # Let's run SLIGHTLY faster than real time but not instant.
            time.sleep(0.05) # 50ms per loop (process takes 200ms real time). 4x speed.
            
    except KeyboardInterrupt:
        print("\nStopped.")
        
    commander.close()
    print(f"\n\n=== Report ===")
    print(f"Total Grasp Actions Triggered: {grasp_count}")
    print("Done.")

def main():
    # Select File
    fif_files = glob.glob("*.fif")
    if not fif_files:
        print("No .fif files found.")
        return
        
    fif_files.sort()
    print("Select a file to replay:")
    for i, f in enumerate(fif_files):
        print(f"{i+1}: {f}")
        
    try:
        idx = int(input("File #: ")) - 1
        if 0 <= idx < len(fif_files):
            target_file = fif_files[idx]
            control_from_file(target_file)
        else:
            print("Invalid.")
    except Exception:
        print("Invalid input.")

if __name__ == "__main__":
    main()
