import mne
import os
import glob
import numpy as np

def inspect_data():
    # Find all matching files
    data_files = glob.glob("Finn_*.fif")
    if not data_files:
        print("Error: No Finn_*.fif files found!")
        return

    print(f"Found {len(data_files)} data files: {data_files}")
    
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

    # Concatenate
    raw = mne.concatenate_raws(raws)
    print("\n" + "="*40)
    print("Combined Raw Info:")
    print(raw.info)
    print("="*40 + "\n")

    # Extract Events
    try:
        events, event_id = mne.events_from_annotations(raw)
        print(f"\nTotal Extracted Events: {len(events)}")
        print(f"Event IDs: {event_id}")
        
        # Count per type
        counts = {}
        for ev in events:
            code = ev[2]
            # Reverse lookup name
            name = [k for k, v in event_id.items() if v == code][0]
            counts[name] = counts.get(name, 0) + 1
            
        print("\nEvent Counts from ALL files:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
            
    except Exception as e:
        print(f"Error extracting events: {e}")

if __name__ == "__main__":
    inspect_data()
