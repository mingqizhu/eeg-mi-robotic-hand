import time
import numpy as np
import torch
import mne
import serial
import multiprocessing
import queue  # For Empty exception
import random
from collections import deque
from statistics import mode
from pylsl import StreamInlet, resolve_streams
from model import EEGNet
from control import HandController
from preprocessing import Preprocessor

# Configuration
STREAM_NAME = "CGX Quick-20r Q20r-0162"
BUFFER_DURATION = 4.0   # seconds
UPDATE_INTERVAL = 0.4   # seconds (2.5Hz prediction)
MODEL_PATH = 'best_model.pth'
SERIAL_PORT = 'COM3'    # Default port
VOTE_WINDOW_SIZE = 5    # Recent predictions to check
VOTE_THRESHOLD = 4      # Votes needed to trigger 'Grasp'

# --- Child Process: Data Acquisition ---
def data_acquisition_worker(stream_name, data_queue, stop_event):
    """
    Continuously pulls data from LSL and pushes to queue.
    """
    print(f"[Data Process] Looking for stream: {stream_name}...")
    streams = resolve_streams()
    target = next((s for s in streams if s.name() == stream_name), None)
    
    if not target:
        print(f"[Data Process] Error: Stream '{stream_name}' not found.")
        return

    inlet = StreamInlet(target)
    print(f"[Data Process] Connected. Pushing data...")
    
    while not stop_event.is_set():
        # Pull chunk
        chunk, ts = inlet.pull_chunk(timeout=1.0)
        if chunk:
            # Send (data, timestamps) tuple
            data_queue.put((chunk, ts))
        else:
            # Sleep briefly to prevent busy waiting if no data
            time.sleep(0.001)

def simulation_data_worker(fs, n_channels, data_queue, stop_event):
    """
    Generates random noise to simulate LSL stream.
    """
    print(f"[Data Process] Starting Simulation Mode...")
    chunk_size = 10 # Simulate 10 samples per chunk
    while not stop_event.is_set():
        # Generate random noise: (chunk_size, n_channels) - LSL format
        chunk = np.random.randn(chunk_size, n_channels).tolist() 
        ts = [time.time()] * chunk_size
        
        data_queue.put((chunk, ts))
        time.sleep(chunk_size / fs)

# --- Main Process ---
def main():
    # 0. Setup Serial
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.1)
        print(f"Serial connected on {SERIAL_PORT}")
    except Exception as e:
        print(f"Warning: Could not connect to serial port {SERIAL_PORT}: {e}")
        print("Running in simulation mode (no hardware output).")

    # 1. Initial LSL Connection (for Info & Calibration)
    print(f"[Main] Looking for stream: {STREAM_NAME}...")
    streams = resolve_streams()
    target = next((s for s in streams if s.name() == STREAM_NAME), None)
    
    use_simulation = False
    
    if not target:
        print(f"Error: Stream '{STREAM_NAME}' not found.")
        user_input = input("Stream not found. Do you want to use simulated signals? (y/n): ").strip().lower()
        if user_input == 'y':
            use_simulation = True
            print("=== RUNNING IN SIMULATION MODE ===")
            # Mock Parameters
            fs = 500
            n_channels = 20
            ch_names = [f"Ch{i+1}" for i in range(n_channels)]
        else:
            print("Exiting.")
            return
    else:
        inlet = StreamInlet(target)
        info_lsl = inlet.info()
        fs = int(info_lsl.nominal_srate())
        n_channels = info_lsl.channel_count()
        
        # Get Channels
        ch = info_lsl.desc().child("channels").child("channel")
        ch_names = []
        for _ in range(n_channels):
            name = ch.child_value("label")
            if name:
                ch_names.append(name)
            ch = ch.next_sibling()
            
        if not ch_names or len(ch_names) != n_channels:
            print("Warning: Channel names not found. Using defaults.")
            ch_names = [f"Ch{i+1}" for i in range(n_channels)]
        
        print(f"Connected to {info_lsl.name()} ({n_channels} ch @ {fs} Hz)")
        print(f"Channels: {ch_names}")
    
    # 2. Load Model & Components (Only if not in simulation?? actually we still need model for pipeline integrity, 
    # but we will override prediction in sim mode)
    samples_per_epoch = int(BUFFER_DURATION * fs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3-Class System
    model = EEGNet(nb_classes=3, Chans=n_channels, Samples=samples_per_epoch)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        if not use_simulation: # If real stream, we need model
            return
        print("Continuing without model in simulation mode.")

    try:
        scaler = np.load("scaler_params.npz")
        scaler_mean = scaler['mean']
        scaler_std = scaler['std']
        print(f"Loaded Scaler: Mean={scaler_mean:.3g}, Std={scaler_std:.3g}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scaler_mean = 0
        scaler_std = 1e-6 

    controller = HandController()
    preprocessor = Preprocessor(sfreq=fs, n_channels=n_channels)
    
    # 3. ASR Calibration (Main Thread)
    if not use_simulation:
        print("\n[CALIBRATION] Recording 5 seconds of baseline data for ASR...")
        print("Please sit RELAXED and STILL.")
        time.sleep(2)
        
        calib_data = []
        end_time = time.time() + 5.0
        while time.time() < end_time:
            chunk, _ = inlet.pull_chunk(timeout=0.0)
            if chunk:
                calib_data.extend(chunk)
            time.sleep(0.01)
            
        if calib_data:
            calib_arr = np.array(calib_data).T 
            print(f"Collected {calib_arr.shape[1]} samples. Calibrating...")
            preprocessor.calibrate_asr(calib_arr)
        else:
            print("Warning: No calibration data!")
            
        # Close local inlet
        del inlet 
    else:
        print("[CALIBRATION] Skipping ASR calibration in simulation mode.")

    # 4. Start Data Process
    data_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    
    if use_simulation:
         p = multiprocessing.Process(target=simulation_data_worker, 
                                    args=(fs, n_channels, data_queue, stop_event))
    else:
        p = multiprocessing.Process(target=data_acquisition_worker, 
                                    args=(STREAM_NAME, data_queue, stop_event))
    p.start()
    print(f"Data process started (PID: {p.pid})")
    
    # 5. Real-time Loop
    buffer = np.zeros((n_channels, samples_per_epoch))
    vote_window = deque(maxlen=VOTE_WINDOW_SIZE)
    
    print(f"\nStarting Prediction Loop (Update every {UPDATE_INTERVAL}s)...")
    print(f"Voting Logic: Majority Vote of {VOTE_WINDOW_SIZE} frames.")
    print("Classes: 0=Rest, 1=Grasp, 2=Null")
    print("Press Ctrl+C to stop.\n")
    
    last_update_time = time.time()
    
    # Simulation State
    sim_state = 0
    sim_state_start = time.time()
    sim_state_duration = 3.0 # seconds per state
    
    try:
        while True:
            # 1. Consume ALL available data from Queue
            try:
                while True:
                    # Get data without blocking
                    chunk, ts = data_queue.get_nowait()
                    chunk = np.array(chunk).T # (Channels, Time)
                    if chunk.shape[1] > 0:
                        n_new = chunk.shape[1]
                        # Update Ring Buffer
                        buffer = np.roll(buffer, -n_new, axis=1)
                        if n_new >= samples_per_epoch:
                            buffer = chunk[:, -samples_per_epoch:]
                        else:
                            buffer[:, -n_new:] = chunk
            except queue.Empty:
                pass 
            
            # 2. Prediction Schedule
            now = time.time()
            if now - last_update_time >= UPDATE_INTERVAL:
                last_update_time = now
                
                # --- Pipeline ---
                # NOTE: Even in sim mode, we run pipeline to ensure no crashes, 
                # but we ignore result.
                data_proc = buffer.copy()
                if not use_simulation:
                     data_proc = preprocessor.apply_asr(data_proc)
                
                # Filter & Laplacian (Skip if simulation to save CPU, or keep to test pipeline?)
                # Let's keep minimal processing or mock it.
                if not use_simulation:
                    data_proc = preprocessor.filter_data(data_proc)
                    data_proc = preprocessor.apply_laplacian(data_proc, ch_names)
                    data_proc = (data_proc - scaler_mean) / (scaler_std + 1e-8) 
                    
                    # Inference
                    input_tensor = torch.tensor(data_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output.data, 1)
                        raw_label = predicted.item() # 0, 1, or 2
                else:
                    # Simulation Logic: Randomly switch state every few seconds
                    if now - sim_state_start > sim_state_duration:
                        sim_state = random.choice([0, 1, 2])
                        sim_state_start = now
                        sim_state_duration = random.uniform(2.0, 5.0)
                        
                    raw_label = sim_state
                
                # Voting
                vote_window.append(raw_label)
                
                # Majority Decision
                if len(vote_window) > 0:
                    try:
                        final_decision = mode(vote_window)
                    except:
                        final_decision = raw_label # tie-break
                else:
                    final_decision = raw_label
                
                # Logic:
                # 0 (Rest) -> Open
                # 1 (Grasp) -> Close
                # 2 (Null) -> Ignore (Do nothing)
                
                vote_str = f"Win:{list(vote_window)}"
                
                if final_decision == 1:
                    print(f"[动作]：右手抓握 (Grasp) | {vote_str} -> 'H'")
                    controller.send_command("CLOSE") 
                    if ser: ser.write(b'H')
                elif final_decision == 0:
                    print(f"[休息]：松开 (Rest)    | {vote_str} -> 'L'")
                    controller.send_command("OPEN") 
                    if ser: ser.write(b'L')
                else:
                    # Class 2 is Null/Noise
                    print(f"[干扰]：忽略 (Null)    | {vote_str} -> No Action")
                    # Do not send command, let previous command persist

            # Sleep to yield CPU
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        p.join()
        if ser: ser.close()
        print("Exited cleanly.")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
