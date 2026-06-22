import numpy as np
from pylsl import StreamInlet, resolve_streams

class LSLAcquisition:
    """
    Handles real-time data acquisition from an LSL stream.
    Maintains a ring buffer to store incoming samples.
    """
    def __init__(self, stream_name="CGX Quick-20r Q20r-0162", buffer_duration=5.0):
        self.stream_name = stream_name
        self.inlet = None
        self.buffer = None
        self.timestamps = None
        self.sfreq = None
        self.n_channels = None
        self.buffer_duration = buffer_duration
        self.samples_in_buffer = 0
        self.pointer = 0
        
    def connect(self):
        """Resolves the stream and initializes the buffer."""
        print(f"Looking for LSL stream: {self.stream_name}...")
        streams = resolve_streams()
        target_stream = None
        for stream in streams:
            if stream.name() == self.stream_name:
                target_stream = stream
                break
        
        if not target_stream:
            raise RuntimeError(f"Could not find stream {self.stream_name}")
        
        self.inlet = StreamInlet(target_stream)
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_channels = info.channel_count()
        
        if self.sfreq == 0:
            print("Warning: Stream has variable sampling rate or 0 specified. Assuming 250Hz.")
            self.sfreq = 250.0
            
        print(f"Connected to {info.name()} - {self.n_channels} channels @ {self.sfreq}Hz")
        
        # Initialize Ring Buffer
        buffer_samples = int(self.sfreq * self.buffer_duration)
        self.buffer = np.zeros((buffer_samples, self.n_channels))
        self.timestamps = np.zeros(buffer_samples)
        self.samples_in_buffer = buffer_samples # Use size for wrapping logic, but count actual valid samples separately if needed.
        
    def update(self):
        """Pull available chunks from LSL and update the ring buffer."""
        if self.inlet is None:
            return

        chunk, ts = self.inlet.pull_chunk(timeout=0.0)
        if chunk:
            chunk = np.array(chunk)
            ts = np.array(ts)
            n_new_samples = len(ts)
            
            # Simple ring buffer implementation
            buffer_len = self.buffer.shape[0]
            indices = np.arange(self.pointer, self.pointer + n_new_samples) % buffer_len
            
            self.buffer[indices] = chunk
            self.timestamps[indices] = ts
            self.pointer = (self.pointer + n_new_samples) % buffer_len
            
    def get_latest_window(self, window_duration=1.0):
        """
        Retrieve the most recent window of data.
        Returns: (n_channels, n_samples)
        """
        n_samples = int(window_duration * self.sfreq)
        buffer_len = self.buffer.shape[0]
        
        # Unwrap the buffer for simplified logical access of "latest" data
        # Note: This is a slightly expensive operation but fine for <1s windows in Python loop
        
        if n_samples > buffer_len:
            raise ValueError("Requested window is larger than buffer size.")

        # Calculate indices of the last n_samples
        # The 'pointer' points to the *next* insertion index, so 'pointer - 1' is the newest.
        indices = np.arange(self.pointer - n_samples, self.pointer) % buffer_len
        
        data_window = self.buffer[indices]
        
        # Transpose to (Channels, Time) as expected by MNE/PyTorch
        return data_window.T

if __name__ == "__main__":
    # Test connection (requires a valid stream)
    try:
        acq = LSLAcquisition(stream_name="CGX_Stream")
        acq.connect()
    except Exception as e:
        print(e)
