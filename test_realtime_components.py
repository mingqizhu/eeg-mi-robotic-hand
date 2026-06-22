import unittest
import numpy as np
import collections
from collections import deque, Counter

# --- Component Implementations (to be moved to main file later) ---

class RingBuffer:
    def __init__(self, n_channels, duration, sfreq):
        self.sfreq = sfreq
        self.capacity = int(duration * sfreq)
        self.n_channels = n_channels
        # Buffer shape: (n_channels, capacity)
        self.buffer = np.zeros((n_channels, self.capacity))
        self.pointer = 0
        self.full = False

    def add(self, chunk):
        """
        Add chunk of data (n_channels, n_samples).
        """
        n_samples = chunk.shape[1]
        
        # If chunk is larger than buffer, just take the last part
        if n_samples >= self.capacity:
            self.buffer = chunk[:, -self.capacity:]
            self.pointer = 0
            self.full = True
            return

        # Indices for circular buffer
        # We need to handle wrapping
        end_ptr = self.pointer + n_samples
        
        if end_ptr <= self.capacity:
            self.buffer[:, self.pointer:end_ptr] = chunk
            self.pointer = end_ptr
            if self.pointer == self.capacity:
                self.pointer = 0
                self.full = True
        else:
            # Wrap around
            overflow = end_ptr - self.capacity
            part1_len = self.capacity - self.pointer
            
            self.buffer[:, self.pointer:] = chunk[:, :part1_len]
            self.buffer[:, :overflow] = chunk[:, part1_len:]
            
            self.pointer = overflow
            self.full = True

    def get_latest(self):
        """
        Get the buffer ordered chronologically.
        """
        if not self.full and self.pointer == 0:
            return np.zeros((self.n_channels, 0))

        # Roll the buffer so that the oldest data is at index 0
        # The 'pointer' points to the oldest data (insertion point for next data)
        # So we roll by -pointer
        return np.roll(self.buffer, -self.pointer, axis=1)

class VotingFilter:
    def __init__(self, window_size=10, threshold=8):
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.threshold = threshold
        
        # Class mapping
        self.REST = 0
        self.GRASP = 1
        self.NULL = 2
        
    def update(self, prediction):
        self.window.append(prediction)
        
        counts = Counter(self.window)
        
        # Default decision is None (hold previous state)
        decision = None
        
        # Check strict threshold
        if counts[self.GRASP] >= self.threshold:
            decision = self.GRASP
        elif counts[self.REST] >= self.threshold:
            decision = self.REST
            
        return decision

# --- Tests ---

class TestRingBuffer(unittest.TestCase):
    def setUp(self):
        self.n_ch = 2
        self.fs = 10
        self.dur = 2 # 20 samples total
        self.rb = RingBuffer(self.n_ch, self.dur, self.fs)

    def test_add_small_chunk(self):
        chunk = np.ones((self.n_ch, 5))
        self.rb.add(chunk)
        # Check if added at start
        np.testing.assert_array_equal(self.rb.buffer[:, 0:5], chunk)
        self.assertEqual(self.rb.pointer, 5)

    def test_wrap_around(self):
        # Fill first
        chunk1 = np.ones((self.n_ch, 15)) * 1
        self.rb.add(chunk1)
        
        # Add chunk that wraps: 5 remaining, add 10 -> wraps by 5
        chunk2 = np.ones((self.n_ch, 10)) * 2
        self.rb.add(chunk2)
        
        # Expected:
        # Buffer was [1...1 (15), 0...0 (5)]
        # After chunk2:
        # Should fill last 5 with 2s, and first 5 with 2s.
        # Middle 10 should still be 1s.
        
        # Check last 5 indices (15-19)
        np.testing.assert_array_equal(self.rb.buffer[:, 15:], chunk2[:, :5])
        # Check first 5 indices (0-4)
        np.testing.assert_array_equal(self.rb.buffer[:, :5], chunk2[:, 5:])
        
        self.assertEqual(self.rb.pointer, 5)
        self.assertTrue(self.rb.full)
        
        # Check get_latest() returns ordered data
        ordered = self.rb.get_latest()
        print(f"Ordered shape: {ordered.shape}")
        
        # Expected ordered: 111...111 (10) then 222...222 (10)
        # Because we overwrote the oldest 5 ones with 2s.
        # Wait. 
        # Old buffer: 111 (15) 0000 (5) -> Pointer at 15
        # Add 10 2s.
        # 5 2s go to 15-19. Buffer: 111 (15) 222 (5). Pointer 0.
        # 5 2s go to 0-4.   Buffer: 222 (5) 111 (10) 222 (5). Pointer 5.
        # Oldest data is now at Pointer (5).
        # get_latest rolls by -5.
        # [5:] + [:5]
        # [111 (10) 222 (5)] + [222 (5)] = 111 (10), 222 (10)
        
        expected = np.concatenate([np.ones((self.n_ch, 10))*1, np.ones((self.n_ch, 10))*2], axis=1)
        np.testing.assert_array_equal(ordered, expected)

class TestVotingFilter(unittest.TestCase):
    def setUp(self):
        self.vf = VotingFilter(window_size=10, threshold=8)
        
    def test_mixed_input(self):
        # 5 Grasp, 5 Rest - specific order?? No, sliding window.
        # let's feed 7 grasps.
        for _ in range(7):
            res = self.vf.update(1)
            self.assertIsNone(res) # < 8
            
        # 8th grasp
        res = self.vf.update(1)
        self.assertEqual(res, 1) # Hit threshold
        
        # Feed a Null (2)
        # Window: G G G G G G G G N
        # Count Grasp: 8 (Wait, window size is 10. we fed 8 Gs initially? No we fed 8 total.)
        # Items: G G G G G G G G (8 items).
        # Now add N. Items: G G G G G G G G N (9 items). Grasp=8.
        res = self.vf.update(2)
        # Should still be Grasp because 8/9 are Grasp.
        self.assertEqual(res, 1)
        
        # Feed 2 more Nulls.
        # Items: G G G G G G G G N N N (11 items -> pop oldest G)
        # Window: G G G G G G G N N N (7 Gs, 3 Ns)
        self.vf.update(2) 
        res = self.vf.update(2)
        self.assertIsNone(res) # Grasp count is 7.

if __name__ == '__main__':
    unittest.main()
