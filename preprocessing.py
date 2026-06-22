import numpy as np
import mne
try:
    from meegkit.asr import ASR
    HAS_MEEGKIT = True
except ImportError:
    HAS_MEEGKIT = False

class Preprocessor:
    """
    Handles signal processing using MNE and advanced filtering.
    """
    def __init__(self, sfreq=500.0, l_freq=4.0, h_freq=30.0, n_channels=26):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.n_channels = n_channels
        
        # ASR State
        self.asr_model = None
        if HAS_MEEGKIT:
            # method="euclid" is faster for real-time
            self.asr_model = ASR(method="euclid") 
        else:
            print("Warning: 'meegkit' not installed. ASR will be disabled.")
            print("To enable, run: pip install meegkit")

    def filter_data(self, data):
        """
        Apply bandpass filter. 
        Input: (Channels, Time)
        """
        # Note: For real-time statefulness, consider sosfilt. 
        # Here using MNE filter for consistency.
        filtered = mne.filter.filter_data(data, self.sfreq, self.l_freq, self.h_freq, 
                                          verbose=False)
        return filtered

    def apply_laplacian(self, data, ch_names):
        """
        Apply Small Laplacian spatial filter to C3 and C4.
        Formula: C_new = C - mean(neighbors)
        """
        if not ch_names:
            return data
            
        # Define neighbors for CGX Q20r (approximate 10-20 locations)
        # C3 neighbors: F3, T7, P3, Cz
        # C4 neighbors: F4, T8, P4, Cz
        neighbors_map = {
            'C3': ['F3', 'T7', 'P3', 'Cz'],
            'C4': ['F4', 'T8', 'P4', 'Cz']
        }
        
        # Create a copy to avoid modifying original in-place during calculation
        filtered_data = data.copy()
        
        for center_ch, neighbors in neighbors_map.items():
            if center_ch in ch_names:
                center_idx = ch_names.index(center_ch)
                
                # Find valid neighbor indices
                neighbor_indices = []
                for n in neighbors:
                    if n in ch_names:
                        neighbor_indices.append(ch_names.index(n))
                
                if neighbor_indices:
                    # Calculate average of neighbors
                    neighbor_avg = np.mean(data[neighbor_indices, :], axis=0)
                    # Subtract from center
                    filtered_data[center_idx, :] = data[center_idx, :] - neighbor_avg
                    
        return filtered_data

    def calibrate_asr(self, data):
        """
        Calibrate ASR using clean baseline data (e.g., Rest period).
        Input: (Channels, Time)
        """
        if not self.asr_model:
            return
            
        print("Calibrating ASR...")
        # ASR expects (Time, Channels) usually? meegkit ASR fits on (n_times, n_chans)
        # Our data is (Channels, Time/Samples)
        data_T = data.T 
        try:
            self.asr_model.fit(data_T)
            print("ASR Calibrated.")
        except Exception as e:
            print(f"ASR Calibration failed: {e}")

    def apply_asr(self, data):
        """
        Apply ASR to remove artifacts.
        Input: (Channels, Time)
        Output: (Channels, Time)
        """
        if not self.asr_model:
            return data
            
        try:
            # Transform expects (n_times, n_chans)
            data_T = data.T
            clean_T = self.asr_model.transform(data_T)
            return clean_T.T # Convert back to (Channels, Time)
        except Exception as e:
            # Fallback if ASR fails (e.g. not calibrated)
            return data

    def standardize(self, data):
        """
        Standard scalar normalization (Zero mean, Unit variance).
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-6)

    def prepare_input(self, data, ch_names=None):
        """
        Process raw data window into model input tensor format.
        Input: (Channels, Time)
        Output: (1, 1, Channels, Time) -> PyTorch Tensor
        """
        # 0. ASR (Artifact Removal) - ideally before filtering or after?
        # ASR works best on broadband data usually.
        # However, for simplicity/speed, we can try it here.
        # If strict real-time, ASR might be heavy.
        data = self.apply_asr(data)
        
        # 1. Bandpass Filter
        filtered = self.filter_data(data)
        
        # 2. Spatial Filter (Laplacian)
        if ch_names:
            filtered = self.apply_laplacian(filtered, ch_names)
        
        # 3. Standardize
        normalized = self.standardize(filtered)
        
        # 4. Reshape for EEGNet (Batch, 1, Channels, Time)
        input_tensor = normalized[np.newaxis, np.newaxis, :, :]
        
        return input_tensor.astype(np.float32)
