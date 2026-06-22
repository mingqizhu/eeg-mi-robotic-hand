import mne
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from model import EEGNet
from preprocessing import Preprocessor

def train():
    # 1. Load Data
    # 1. Load Data
    print("\n=== Training Data Selection ===")
    dataset_name = input("Enter dataset name (e.g. 'Finn'): ").strip()
    if not dataset_name:
        print("Error: Name cannot be empty.")
        return

    pattern = f"{dataset_name}_*.fif"
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"Error: No files found matching '{pattern}'")
        return
        
    all_files.sort() # Ensure logical order
    print(f"Found {len(all_files)} files matching '{dataset_name}':")
    for f in all_files:
        print(f" - {f}")
    
    # Selection prompt
    selection_input = input(f"\nEnter file numbers to USE (e.g. '1 3'), or press Enter to consume ALL: ").strip()
    
    data_files = []
    if not selection_input:
        data_files = all_files
        print(f"Using ALL {len(data_files)} files.")
    else:
        target_nums = selection_input.split()
        print(f"Filtering for numbers: {target_nums}...")
        
        for f in all_files:
            try:
                # Expecting Name_Number.fif
                base = os.path.splitext(f)[0]
                num_part = base.split('_')[-1]
                if num_part in target_nums:
                    data_files.append(f)
            except:
                pass
                
        if not data_files:
            print("Error: No files matched your selection!")
            return
            
        print(f"Selected {len(data_files)} files: {data_files}")
    
    raws = []
    for f in data_files:
        print(f"Loading {f}...")
        try:
            raw = mne.io.read_raw_fif(f, preload=True)
            raws.append(raw)
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
            
    if not raws:
        return

    # ... (Loading files code above remains similar, assuming 'data_files' is populated) ...
    # This block replaces the main processing and training loop in train().
    
    # Concatenate
    raw = mne.concatenate_raws(raws)
    print(f"Combined data: {raw.n_times} samples, {len(raw.annotations)} annotations.")
    
    # --- PIPELINE ALIGNMENT: 1. ASR Calibration ---
    # We must calibrate ASR on 'Rest' data, similar to real-time.
    # 1. Initialize Preprocessor
    n_chans = raw.info['nchan']
    sfreq = raw.info['sfreq']
    preprocessor = Preprocessor(sfreq=sfreq, n_channels=n_chans)
    
    # 2. Extract Rest Data for Calibration
    # Events: Rest should be mapped.
    # We need to parse annotations to find 'Rest' segments.
    # Or just use the first 'Rest' event found.
    # Let's be thorough: use all Rest segments available to calibrate? 
    # Or just the first 10-20 seconds of Rest.
    # Note: mne.events_from_annotations returns event_id mapping.
    
    events, event_id = mne.events_from_annotations(raw)
    print(f"Found events: {event_id}")
    
    rest_id = event_id.get('Rest')
    
    should_calibrate = False
    if rest_id is None:
        print("Warning: No 'Rest' events found. Skipping ASR.")
    else:
        # Get all Rest event indices
        rest_onsets = events[events[:, 2] == rest_id, 0]
        
        if len(rest_onsets) == 0:
            print("Warning: 'Rest' event ID exists but no events found. Skipping ASR.")
        else:
            calib_data = []
            all_raw_data = raw.get_data()
            
            for onset in rest_onsets:
                # onset is sample index
                # Duration is 4.0s per trial
                start_samp = onset
                end_samp = int(onset + 4.0 * sfreq)
                
                # Bounds check
                if start_samp < 0 or end_samp > all_raw_data.shape[1]:
                    continue
                    
                chunk = all_raw_data[:, start_samp:end_samp]
                calib_data.append(chunk)

            total_samples = sum([c.shape[1] for c in calib_data]) if calib_data else 0
            total_seconds = total_samples / sfreq
            
            if total_seconds < 10.0:
                print(f"Warning: Only found {total_seconds:.1f}s of Rest data (required 10s). Skipping ASR.")
            else:
                should_calibrate = True
                calib_arr = np.concatenate(calib_data, axis=1) # (Ch, Time)
                print(f"Calibrating ASR on {total_seconds:.1f}s of Rest data...")
                preprocessor.calibrate_asr(calib_arr)

    # --- PIPELINE ALIGNMENT: 2. Apply ASR to Full Data ---
    print("Applying ASR to full dataset artifact removal...")
    # This might take a moment
    full_data = raw.get_data() # (Ch, Time)
    full_data_clean = preprocessor.apply_asr(full_data)
    
    # Re-wrap in RawArray for easy filtering/epoching
    raw_clean = mne.io.RawArray(full_data_clean, raw.info)
    # Restore annotations
    raw_clean.set_annotations(raw.annotations)
    
    # --- PIPELINE ALIGNMENT: 3. Filter ---
    # 4-30Hz
    print("Filtering (4-30Hz)...")
    raw_clean.filter(4., 30., fir_design='firwin')
    
    # --- PIPELINE ALIGNMENT: 4. Laplacian ---
    # C3/C4
    print("Applying Laplacian Filter...")
    ch_names = raw.ch_names
    # Need to manually apply this, MNE Raw doesn't have a simple method for our custom Laplacian
    # We get data again
    data_filtered = raw_clean.get_data()
    data_lap = preprocessor.apply_laplacian(data_filtered, ch_names)
    
    # Re-wrap again (inefficient but safe for MNE Steps)
    raw_final = mne.io.RawArray(data_lap, raw.info)
    raw_final.set_annotations(raw.annotations)
    
    # 3. Extract Epochs
    # Events: Rest, Right Hand Grasp, Null
    # Update event extraction on new raw
    events, event_id = mne.events_from_annotations(raw_final)
    print(f"Events after preprocessing: {event_id}")
    
    # Target Classes: Rest, Grasp, Null
    target_classes = ['Rest', 'Right Hand Grasp', 'Null']
    used_event_id = {k: v for k, v in event_id.items() if k in target_classes}
    print(f"Using Event IDs: {used_event_id}")
    
    tmin, tmax = 0, 4.0 - 1.0/raw.info['sfreq'] 
    epochs = mne.Epochs(raw_final, events, event_id=used_event_id, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True)
    
    # Get data
    X = epochs.get_data() 
    y = epochs.events[:, -1]
    
    if len(X) < 10:
        print(f"\nWARNING: Only {len(X)} epochs found!")
    
    # --- Scaling (StandardScaler) ---
    print("Scaling data (StandardScaler)...")
    scaler_mean = np.mean(X)
    scaler_std = np.std(X)
    X = (X - scaler_mean) / (scaler_std + 1e-8)
    print(f"Data scaled. Mean: {np.mean(X):.3f}, Std: {np.std(X):.3f}")
    
    # Save Scaler
    np.savez("scaler_params.npz", mean=scaler_mean, std=scaler_std)
    print("Saved scaler_params.npz")

    # Remap Labels for 3-Class
    # 0: Rest, 1: Grasp, 2: Null
    rest_id = event_id.get('Rest')
    grasp_id = event_id.get('Right Hand Grasp')
    null_id = event_id.get('Null')
    
    new_y = []
    for label in y:
        if label == rest_id:
            new_y.append(0) 
        elif label == grasp_id:
            new_y.append(1)
        elif null_id is not None and label == null_id:
            new_y.append(2)
            
    y = np.array(new_y)
    
    print(f"Data Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}") # Should see 0, 1, 2
    
    # 4. Prepare Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 5. Model Setup
    n_classes = 3 # 3-Class System
    n_samples = X.shape[2]
    
    model = EEGNet(nb_classes=n_classes, Chans=n_chans, Samples=n_samples)
    
    # --- Transfer Learning Logic ---

    pretrained_path = "pretrained_eegnet.pth"
    if os.path.exists(pretrained_path):
        print(f"\n[Transfer Learning] Found '{pretrained_path}'. Loading weights...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            saved_state = torch.load(pretrained_path, map_location=device)
            
            # Smart Loading: Filter out keys with shape mismatches
            model_state = model.state_dict()
            clean_state_dict = {}
            
            for k, v in saved_state.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        clean_state_dict[k] = v
                    else:
                        print(f"   Mismatch: Skipping '{k}' ({v.shape} != {model_state[k].shape})")
                
            # Load compatible weights
            model.load_state_dict(clean_state_dict, strict=False)
            print("   Weights loaded successfully (compatible layers only).")
            
            # Freeze First Two Conv Layers (Conv1 & Conv2)
            # This corresponds to the temporal and spatial filters.
            print("   Freezing first two convolution layers (Conv1 & Conv2)...")
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
                
            print("   Migration complete. Training will update subsequent layers.")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Falling back to standard training.")
    else:
        print(f"\n[Standard Training] '{pretrained_path}' not found. logic: straight training.")
    
    criterion = nn.CrossEntropyLoss()
    # Optimizer: Only optimize parameters that require grad? 
    # Standard Adam will handle requires_grad=False correctly (no update).
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler: Reduce LR when Test Accuracy plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 6. Training Loop
    n_epochs = 40
    best_acc = 0.0
    
    print("Starting training (3-Class)...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_acc = 100 * test_correct / test_total
        
        # Scheduler Step
        scheduler.step(test_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Early Stopping
        if current_lr < 1e-6:
            print("Learning rate too low. Early stopping triggered.")
            break
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    print(f"Training Finished. Best Test Accuracy: {best_acc:.2f}%")
    print("Best model saved to 'best_model.pth'")

if __name__ == "__main__":
    train()
