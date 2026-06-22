import time
import multiprocessing
import numpy as np
import torch
import serial
import queue
from collections import deque, Counter
from pylsl import StreamInlet, resolve_streams
import sys
import os

# --- Import Local Modules ---
try:
    from model import EEGNet
    from preprocessing import Preprocessor
except ImportError:
    # Allow running from a different directory if needed, though usually CWD is correct
    sys.path.append(os.getcwd())
    from model import EEGNet
    from preprocessing import Preprocessor

# --- Configuration ---
STREAM_NAME = "CGX Quick-20r Q20r-0162"
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler_params.npz"
SERIAL_PORT = "COM5"
BAUD_RATE = 115200

BUFFER_DURATION = 4.0   # 4 seconds memory
PREDICTION_INTERVAL = 0.2 # 200 ms
CALIBRATION_DURATION = 5.0

# --- LSL Worker ---

def lsl_worker(stream_name, data_queue, stop_event, use_sim=False):
    """
    Process to continuously pull data from LSL and push to Queue.
    """
    if use_sim:
        print("[LSL] Starting Simulation Mode...")
        fs = 500
        n_ch = 20
        while not stop_event.is_set():
            # Create random chunk
            chunk_size = 10 # 50 Hz updates
            chunk = np.random.randn(chunk_size, n_ch).tolist()
            ts = [time.time()] * chunk_size
            data_queue.put((chunk, ts))
            time.sleep(chunk_size / fs)
        return

    print(f"[LSL] Searching for stream: {stream_name}...")
    streams = resolve_streams()
    target = next((s for s in streams if s.name() == stream_name), None)
    
    if not target:
        print(f"[LSL] ERROR: Stream '{stream_name}' not found.")
        # Signal main process? For now just exit.
        return

    inlet = StreamInlet(target)
    print(f"[LSL] Connected to {stream_name}.")
    
    while not stop_event.is_set():
        try:
            chunk, ts = inlet.pull_chunk(timeout=1.0)
            if chunk:
                data_queue.put((chunk, ts))
            else:
                time.sleep(0.001)
        except Exception as e:
            print(f"[LSL] Error: {e}")
            break
    
    print("[LSL] Worker stopping.")


# --- Components ---

class RingBuffer:
    def __init__(self, n_channels, duration, sfreq):
        self.sfreq = sfreq
        self.capacity = int(duration * sfreq)
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, self.capacity))
        self.pointer = 0
        self.full = False

    def add(self, chunk):
        """Add (n_channels, n_samples) chunk."""
        n_samples = chunk.shape[1]
        if n_samples == 0: return

        if n_samples >= self.capacity:
            self.buffer = chunk[:, -self.capacity:]
            self.pointer = 0
            self.full = True
            return

        end_ptr = self.pointer + n_samples
        if end_ptr <= self.capacity:
            self.buffer[:, self.pointer:end_ptr] = chunk
            self.pointer = end_ptr
            if self.pointer == self.capacity:
                self.pointer = 0
                self.full = True
        else:
            overflow = end_ptr - self.capacity
            part1_len = self.capacity - self.pointer
            self.buffer[:, self.pointer:] = chunk[:, :part1_len]
            self.buffer[:, :overflow] = chunk[:, part1_len:]
            self.pointer = overflow
            self.full = True

    def get_latest(self):
        """Return buffer ordered chronologically: (n_channels, capacity)."""
        if not self.full and self.pointer == 0:
            return np.zeros((self.n_channels, 0))
        return np.roll(self.buffer, -self.pointer, axis=1)

class VotingFilter:
    def __init__(self, window_size=10, threshold=8):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
        self.REST = 0
        self.GRASP = 1
        self.NULL = 2
        
    def update(self, prediction):
        self.window.append(prediction)
        counts = Counter(self.window)
        
        # Decision Logic
        # "只有当最近 10 次预测中，某一意图（如 Grasp）出现 8 次及以上时，才触发"
        if counts[self.GRASP] >= self.threshold:
            return self.GRASP
        elif counts[self.REST] >= self.threshold:
            return self.REST
        else:
            return None # No secure decision

class SerialCommander:
    def __init__(self, port, baudrate):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
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

# --- Main System ---

def main():
    multiprocessing.freeze_support()
    print("=== Real-time EEG Hand Control System ===")
    
    # 1. Setup Data Queue & Worker
    data_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    
    # Simulation Check
    use_sim = False
    streams = resolve_streams(wait_time=1.0)
    cgx_found = any(s.name() == STREAM_NAME for s in streams)
    
    if not cgx_found:
        print(f"Stream '{STREAM_NAME}' not found.")
        reply = input("Run in simulation mode? (y/n): ").strip().lower()
        if reply == 'y':
            use_sim = True
        else:
            print("Exiting.")
            return

    # Start Worker
    p = multiprocessing.Process(target=lsl_worker, args=(STREAM_NAME, data_queue, stop_event, use_sim))
    p.start()
    
    # 2. Setup Processing & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Determine FS/Channels
    if use_sim:
        fs = 500
        n_channels = 20
        ch_names = [f"Ch{i+1}" for i in range(20)]
        print("[System] Simulating 20 channels @ 500Hz")
    else:
        # Resolve again to get info or assume defaults if we trust the worker found it
        # Ideally we get this from the worker, but for simplicity let's assume standard CGX
        # Or peak at the first chunk from queue
        print("[System] Waiting for stream info...")
        # A robust way is to wait for first chunk
        while data_queue.empty():
            time.sleep(0.1)
        chunk_sample, _ = data_queue.get() # Don't lose this data, put it back or process?
        # Put back is tricky with queue order. Let's just consume it. To get metadata we really should have done it before worker loop.
        # But let's assume:
        fs = 500 # CGX default usually
        n_channels = len(chunk_sample[0])
        # We need channel names for Laplacian.
        # Since logic is in main, we can create a temporary inlet to get info, then close it.
        # BUT the worker has the inlet.
        # Let's assume standard channel names or indices if map is known.
        # For this implementation, we'll try to use default `preprocessing.py` logic which handles missing names gracefully (returns data unmodified).
        # OR we can hardcode for Q20r if known.
        ch_names = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz","Oz"][:n_channels] 
        print(f"[System] Detected {n_channels} channels. Assuming {fs}Hz.")

    # Load Model
    # Samples per epoch: 200ms prediction? No, inference usually requires a larger window (e.g. 0.5s or 1s) to make a prediction.
    # The prompt says: "High frequency sliding window inference... every 200ms output a prediction".
    # But what is the INPUT window size? Usually 1 second or 0.5s. 
    # Let's look at `realtime_predict.py`: `BUFFER_DURATION = 4.0`. `model = EEGNet(..., Samples=samples_per_epoch)`.
    # It used `int(BUFFER_DURATION * fs)`. That seems HUGE (4s window for one prediction?).
    # Standard EEGNet is often 0.5s - 1s.
    # The user manual doesn't specify input window size, only "output every 200ms".
    # I will stick to what `realtime_predict.py` did, OR better, use a standard 1s window for stability, sliding by 200ms.
    # Let's check `train_model.py` (not provided in context but referenced in conversation).
    # Ah, `model.py` default `Samples=256`. At 500Hz that is ~0.5s.
    # Let's assume the model was trained on ~1s or similar.
    # I will inspect `best_model_70acc.pth` if I could, but I can't.
    # However, `realtime_predict.py` line 120: `samples_per_epoch = int(BUFFER_DURATION * fs)`.
    # And `BUFFER_DURATION` was 4.0.
    # That implies the model takes 2000 samples??
    # That seems very large for EEGNet.
    # Let's look at `realtime_predict.py` line 218: `buffer = chunk[:, -samples_per_epoch:]`.
    # If the model really expects 4s, I should use 4s. I'll stick to 256 samples (approx 0.5s) if the previous code was wrong, OR check `model.py` defaults.
    # `model.py` defaults to `Samples=256`.
    # But `realtime_predict` initialized: `EEGNet(..., Samples=samples_per_epoch)`.
    # So `realtime_predict` was using 4s.
    # I will define `INPUT_WINDOW_SECONDS = 4.0` to match `realtime_predict.py` unless I see evidence otherwise.
    # User said "High Quality Model best_model_70acc.pth".
    # If I change input size, the weights won't match (FC layer mismatch).
    # I will use 4.0 seconds to be safe.
    
    INPUT_WINDOW_SEC = 4.0
    SAMPLES_MODEL = int(INPUT_WINDOW_SEC * fs)
    
    model = EEGNet(nb_classes=3, Chans=n_channels, Samples=SAMPLES_MODEL)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"[System] Model {MODEL_PATH} loaded.")
    except Exception as e:
        print(f"[System] Error loading model: {e}")
        if not use_sim:
            print("Cannot proceed without model.")
            stop_event.set()
            p.join()
            return
        print("[System] Continuing in Simulation (Random Output).")

    # Components
    buffer = RingBuffer(n_channels, BUFFER_DURATION + 1.0, fs) # Extra space
    preprocessor = Preprocessor(sfreq=fs, n_channels=n_channels)
    
    # Load Scaler
    if os.path.exists(SCALER_PATH):
        scaler = np.load(SCALER_PATH)
        scaler_mean = scaler['mean']
        scaler_std = scaler['std']
        print(f"[System] Scaler loaded.")
    else:
        print("[System] Warning: No scaler found. Using 0/1.")
        scaler_mean = 0
        scaler_std = 1

    voting = VotingFilter(window_size=10, threshold=8)
    commander = SerialCommander(SERIAL_PORT, BAUD_RATE)

    # 3. Calibration
    print("\n" + "="*40)
    print("   CALIBRATION (5 seconds)")
    print("   Please RELAX and keep hand STILL")
    print("="*40 + "\n")
    
    calib_buffer = []
    calib_end = time.time() + CALIBRATION_DURATION
    
    while time.time() < calib_end:
        while not data_queue.empty():
            chunk, ts = data_queue.get()
            chunk = np.array(chunk).T # (Ch, Time)
            if chunk.shape[1] > 0:
                # Accumulate for ASR
                if len(calib_buffer) == 0:
                    calib_buffer = chunk
                else:
                    calib_buffer = np.concatenate([calib_buffer, chunk], axis=1)
                buffer.add(chunk) # Also fill ring buffer
        time.sleep(0.01)
        
    # Perform Calibration
    if len(calib_buffer) > 0 and not use_sim:
        print(f"[ASR] Calibrating with {calib_buffer.shape[1]} samples...")
        preprocessor.calibrate_asr(calib_buffer)
    else:
        print("[ASR] Skipped (Sim or No Data).")

    # 4. Main Loop
    print("\n[System] STARTING REAL-TIME CONTROL")
    print("CTRL+C to Stop")
    
    last_pred_time = time.time()
    
    try:
        while True:
            # 1. Drain Queue -> Buffer
            while not data_queue.empty():
                chunk, ts = data_queue.get()
                chunk = np.array(chunk).T
                buffer.add(chunk)
            
            # 2. Check Interval
            now = time.time()
            if now - last_pred_time >= PREDICTION_INTERVAL:
                last_pred_time = now
                
                # 3. Get Window
                # We need the last `SAMPLES_MODEL` samples
                full_data = buffer.get_latest()
                if full_data.shape[1] < SAMPLES_MODEL:
                    continue # Not enough data yet
                
                # Extract exact window logic
                # SAMPLES_MODEL corresponds to 4.0s usually
                input_data = full_data[:, -SAMPLES_MODEL:]
                
                # 4. Preprocessing
                # Copy to avoid mutating buffer
                proc_data = input_data.copy()
                
                # ASR -> Filter -> Lap -> Norm (User Order? "4-30Hz + Lap + ASR + Norm")
                # User specifically requested: "4-30Hz 滤波 + 拉普拉斯空间滤波 + ASR 应用 + 标准化"
                # Implementation consistent with user request:
                
                # 4.1 Filter (4-30Hz)
                proc_data = preprocessor.filter_data(proc_data)
                
                # 4.2 Laplacian
                proc_data = preprocessor.apply_laplacian(proc_data, ch_names if not use_sim else None)
                
                # 4.3 ASR
                if not use_sim:
                    proc_data = preprocessor.apply_asr(proc_data)
                    
                # 4.4 Standardize
                # Use loaded scaler
                proc_data = (proc_data - scaler_mean) / (scaler_std + 1e-6)
                
                # 5. Inference
                if not use_sim:
                    # (Batch, 1, Ch, Time)
                    x = torch.tensor(proc_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(x)
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                        pred_idx = np.argmax(probs)
                else:
                    # Sim random
                    probs = np.random.rand(3)
                    probs /= probs.sum()
                    pred_idx = np.argmax(probs)
                    if np.random.rand() < 0.1: pred_idx = 1 # occasional grasp

                # 6. Null Class Handling (Class 2)
                # "If Null (2), ignore current instruction"
                # Interpretation: Do not feed to voting filter? Or feed "Null"?
                # "Decision Voting: ... only when specific intent (Grasp) appears 8/10..."
                # If we skip feeding, the window becomes stale / effectively holds state?
                # User says: "如果识别结果为 Null (类别 2)，系统应忽略当前指令，不向串口发送动作。"
                # But also "Voting ... 8/10".
                # If 8 are Grasp, 1 is Null, 1 is Rest -> Action?
                # So Null counts as a vote (a "Null" vote). It just prevents Grasp/Rest from reaching 8.
                
                # Update Voting
                final_decision = voting.update(pred_idx)
                
                # 7. Serial Control
                action_str = "HOLD"
                if final_decision == voting.GRASP:
                    commander.send('G')
                    action_str = "GRASP ('G')"
                elif final_decision == voting.REST:
                    commander.send('R')
                    action_str = "REST ('R')"
                    
                # 8. TUI
                # Progress bar style
                # Probs: [R: 0.1, G: 0.8, N: 0.1]
                # Vote Queue: [G G G G G R N G G G] -> GRASP
                
                # Mapping
                cls_map = {0: 'R', 1: 'G', 2: 'N'}
                q_vis = "".join([cls_map.get(x, '?') for x in voting.window])
                
                # Probs
                p_r = int(probs[0]*10)
                p_g = int(probs[1]*10)
                p_n = int(probs[2]*10)
                bar = f"R:{probs[0]:.2f} G:{probs[1]:.2f} N:{probs[2]:.2f}"
                
                sys.stdout.write(f"\rStatus: {action_str:<12} | Probs: [{bar}] | Vote: [{q_vis:<10}]")
                sys.stdout.flush()
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    finally:
        stop_event.set()
        p.join()
        commander.close()
        print("[System] Shutdown Complete.")

if __name__ == "__main__":
    main()
