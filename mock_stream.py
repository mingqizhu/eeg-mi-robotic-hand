import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

def run_mock_stream(stream_name="CGX Quick-20r Q20r-0162", n_channels=26, srate=500):
    print(f"Starting Mock LSL Stream: {stream_name}")
    
    # create stream info
    info = StreamInfo(stream_name, 'EEG', n_channels, srate, 'float32', 'myuidw43536')
    
    # append some meta-data
    channels = info.desc().append_child("channels")
    for c in range(n_channels):
        channels.append_child("channel").append_child_value("label", f"Ch{c+1}")
        
    outlet = StreamOutlet(info)
    
    print("Now sending data...")
    start_time = time.time()
    
    while True:
        # make a new random 20-channel sample; 
        # normally we would send this sample by sample or in chunks
        # Here we send chunks of 10 samples (arbitrary) to simulate packet arrival
        chunk_size = 10
        mysample = np.random.randn(chunk_size, n_channels)
        
        # Simulate "Motor Imagery" (Sine wave in channels C3/C4 approx - lets say Ch 5 and 15)
        # Every 5 seconds, inject signal for 2 seconds
        current_time = time.time()
        cycle = current_time % 10
        if 2 < cycle < 4:
            # Active "Event"
            t = np.linspace(current_time, current_time + chunk_size/srate, chunk_size)
            mysample[:, 4] += 5 * np.sin(2 * np.pi * 10 * t) # Ch5 10Hz
            mysample[:, 14] += 5 * np.sin(2 * np.pi * 10 * t) # Ch15 10Hz
            
        outlet.push_chunk(mysample)
        
        # separate by correct time interval
        time.sleep(chunk_size / srate)

if __name__ == "__main__":
    run_mock_stream()
