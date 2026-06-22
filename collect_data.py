import time
import random
import numpy as np
import mne
from pylsl import StreamInlet, resolve_streams

def collect_data():
    # Configuration
    STREAM_NAME = "CGX Quick-20r Q20r-0162"
    TRIALS_PER_CLASS = 20
    TRIAL_DURATION = 4.0 # seconds
    ITI_DURATION = 2.0   # Inter-trial interval
    
    # 3-Class System
    # 1: Rest
    # 2: Grasp
    # 3: Null (Random movement / Distraction)
    CLASSES = {'Rest': 1, 'Right Hand Grasp': 2, 'Null': 3}
    EXPECTED_CHANNELS = 26
    
    # 1. Connect to LSL
    print(f"Looking for stream: {STREAM_NAME}...")
    streams = resolve_streams()
    target = next((s for s in streams if s.name() == STREAM_NAME), None)
    
    if not target:
        print(f"Error: Stream '{STREAM_NAME}' not found.")
        return
        
    inlet = StreamInlet(target)
    info_lsl = inlet.info()
    fs = info_lsl.nominal_srate()
    n_channels = info_lsl.channel_count()
    print(f"Connected to {info_lsl.name()} ({n_channels} ch @ {fs} Hz)")
    
    if n_channels != EXPECTED_CHANNELS:
        print(f"WARNING: Expected {EXPECTED_CHANNELS} channels but got {n_channels}.")
    
    # Prompt for Subject Name
    subject_name = input("Enter Subject Name (e.g. 'Finn'): ").strip()
    if not subject_name: subject_name = "Subject"
    
    # buffers to hold all data
    all_data = []
    all_times = []
    events = [] # (onset_sample, duration, event_id)
    
    # 2. Prepare Protocol
    # Create list of labels
    labels = []
    for cls_name in CLASSES.keys():
        labels.extend([cls_name] * TRIALS_PER_CLASS)
    
    random.shuffle(labels)
    
    print("\n" + "="*40)
    print(f"Starting Data Collection Protocol")
    print(f"Classes: {list(CLASSES.keys())}")
    print(f"Total Trials: {len(labels)} ({TRIALS_PER_CLASS} per class)")
    print(f"Trial Length: {TRIAL_DURATION}s + ITI {ITI_DURATION}s")
    print("Please relax and follow the on-screen prompts.")
    print("For 'Null', perform random small movements or distractions.")
    print("="*40 + "\n")
    
    input("Press ENTER to start recording...")
    
    start_time = time.time()
    
    # Helper to pull data continuously
    def pull_data_for(duration):
        end_time = time.time() + duration
        pts = 0
        while time.time() < end_time:
            chunk, ts = inlet.pull_chunk(timeout=0.0)
            if chunk:
                all_data.extend(chunk)
                all_times.extend(ts) # TS from LSL (absolute)
                pts += len(chunk)
            time.sleep(0.005) # Yield
        return pts

    try:
        for i, label in enumerate(labels):
            iter_start = time.time()
            print(f"\nTrial {i+1}/{len(labels)}")
            
            # A. Inter-Trial Interval (Rest/Blank)
            print(">>> RELAX <<<")
            pull_data_for(ITI_DURATION)
            
            # B. Action Trial
            print(f">>> IMAGINE/DO: [{label}] <<<")
            event_code = CLASSES[label]
            
            onset_rel = time.time() - start_time
            events.append([onset_rel, TRIAL_DURATION, event_code])
            
            # Record data
            pts = pull_data_for(TRIAL_DURATION)
            print(f"  -> Recorded ~{pts} samples.")
            
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user. Saving partial data...")
    except Exception as e:
        print(f"\n\nError occurred: {e}. Saving partial data...")
        
    print("\nRecording finished. Processing data...")
    
    if len(all_data) == 0:
        print("No data recorded!")
        return
        
    # 3. Save to MNE format
    data_array = np.array(all_data).T # (Channels, Samples)
    print(f"Data Shape: {data_array.shape}")
    
    data_times = np.array(all_times)
    
    # Create MNE Info
    # Try to use standard channel names if possible, but for now generic.
    ch_names = [f"Ch{i+1}" for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data_array, info)
    
    # Add Annotations
    if events:
        # Convert list to arrays
        ev_arr = np.array(events)
        onsets = ev_arr[:, 0]
        durations = ev_arr[:, 1]
        codes = ev_arr[:, 2]
        
        # Map codes back to strings
        # Inverse mapping
        inv_classes = {v: k for k, v in CLASSES.items()}
        descriptions = [inv_classes[int(c)] for c in codes]
        
        print(f"Saving {len(events)} annotations...")
        annot = mne.Annotations(onsets, durations, descriptions)
        raw.set_annotations(annot)
    else:
        print("WARNING: No events recorded!")
    
    # Find next available filename
    # Format: [Name]_[Number].fif
    import glob
    existing = glob.glob(f"{subject_name}_*.fif")
    next_num = 1
    if existing:
        # very simple increment
        next_num = len(existing) + 1
        
    filename = f"{subject_name}_{next_num}.fif"
    raw.save(filename)
    print(f"Data saved to: {filename}")
    print("You can verify it using: python inspect_data.py")

if __name__ == "__main__":
    collect_data()
