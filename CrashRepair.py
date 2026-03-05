import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import traceback

def restructure_arr(arr):
    diff = np.diff(arr)
    differences = [np.median(diff) if val == 0 else val for val in diff]
    return np.concatenate(([arr[0]], arr[0] + np.cumsum(differences)))

class CrashRepair:
    def __init__(self, data_df, sampling_rate=30, target_max_position=0.4, window_size=3.0):
        """
        Initialize the CrashRepair class with data and parameters.
        
        :param data_df: DataFrame containing the tracking data.
        :param sampling_rate: Sampling rate of the data.
        :param target_max_position: Maximum allowed position value.
        :param window_size: Size of the window for crash detection.
        """
        self.data = data_df.copy()
        self.sampling_rate = sampling_rate
        self.frame_duration = 1 / sampling_rate
        self.target_max_position = target_max_position
        self.window_frame_count = int(window_size * sampling_rate)
        self._segments = None
        
    def set_target_max_position(self):
        """
        Set the target maximum position based on the 99th percentile of
        the absolute stimulus positions. This overwrites the target_max_position
        set in the constructor.
        """
        self.target_max_position = np.percentile(np.abs(self.data['stim_pos'].values), 99)

    def smooth_dampen(self, values, target_max, scale_factor=1.5):
        """
        Apply a damping function to keep values within the target range.
        
        :param values: Array of values to dampen.
        :param target_max: Maximum target value.
        :param scale_factor: Factor to control the damping.
        :return: Dampened values.
        """
        if target_max == 0:
            return values
        normalized = np.abs(values) / target_max
        dampened = np.tanh(normalized / scale_factor) * target_max
        return np.sign(values) * dampened
    
    def _fit_pchip(self, times, values):
        try:
            return PchipInterpolator(times, values)
        except Exception:
            try:
                return PchipInterpolator(restructure_arr(times), values)
            except Exception:
                traceback.print_exc()
                return None

    def find_crash_segments(self):
        """
        Identify segments in the data where crashes occur.
        
        :return: List of dictionaries containing crash segment information.
        """
        if self._segments is not None:
            return self._segments

        crash_transitions = self.data[self.data['crash_count'].diff() != 0].index.tolist()
        segments = []

        for crash_idx in crash_transitions:
            # Adjust window sizes based on available data
            pre_window = min(self.window_frame_count, crash_idx)
            post_window = min(self.window_frame_count, len(self.data) - crash_idx)
            
            # Only skip if we have no data before or after crash
            if pre_window == 0 or post_window == 0:
                continue
                
            pre_crash = self.data.iloc[crash_idx - pre_window:crash_idx]
            post_crash = self.data.iloc[crash_idx:crash_idx + post_window]
            gap_duration = post_crash['flip_time'].iloc[0] - pre_crash['flip_time'].iloc[-1]
            n_missing_frames = int(np.round(gap_duration * self.sampling_rate))
            
            segments.append({
                'crash_idx': crash_idx,
                'pre_crash': pre_crash,
                'post_crash': post_crash,
                'gap_duration': gap_duration,
                'n_missing_frames': n_missing_frames,
                'is_partial': pre_window < self.window_frame_count or post_window < self.window_frame_count
            })
        
        self._segments = segments
        return self._segments

    def compute_transition(self, pre_data, post_data):
        if len(pre_data) < 2 or len(post_data) < 2:
            return None

        n_frames = len(pre_data) + len(post_data)
        times = np.linspace(
            pre_data['flip_time'].iloc[0],
            post_data['flip_time'].iloc[-1],
            n_frames
        )

        all_times = np.concatenate([pre_data['flip_time'].values, post_data['flip_time'].values])
        stim_vals = np.concatenate([pre_data['stim_pos'].values, post_data['stim_pos'].values])
        user_vals = np.concatenate([pre_data['user_pos'].values, post_data['user_pos'].values])

        stim_pchip = self._fit_pchip(all_times, stim_vals)
        if stim_pchip is None:
            return None
        user_pchip = self._fit_pchip(all_times, user_vals)
        if user_pchip is None:
            return None

        interp_stim = stim_pchip(times)
        interp_user = user_pchip(times)

        interp_stim = self.smooth_dampen(interp_stim, self.target_max_position)
        interp_user = self.smooth_dampen(interp_user, self.target_max_position)

        window_length = min(15, len(interp_stim) // 2 * 2 - 1)
        if window_length >= 5:
            interp_stim = savgol_filter(interp_stim, window_length, 3)
            interp_user = savgol_filter(interp_user, window_length, 3)

        return pd.DataFrame({
            'stim_pos': interp_stim,
            'user_pos': interp_user,
            'flip_time': times
        })

    def resample_data(self, data, target_frequency=30):
        # Calculate the new time index based on the target frequency
        start_time = data['flip_time'].iloc[0]
        end_time = data['flip_time'].iloc[-1]
        new_time_index = np.arange(start_time, end_time, 1.0 / target_frequency)

        # Interpolate the data to the new time index
        stim_interp = np.interp(new_time_index, data['flip_time'], data['stim_pos'])
        user_interp = np.interp(new_time_index, data['flip_time'], data['user_pos'])
        # Create a new DataFrame with the resampled data
        resampled_data = pd.DataFrame({
            'flip_time': new_time_index,
            'stim_pos': stim_interp,
            'user_pos': user_interp
        })

        return resampled_data

    def repair_tracking(self):
        segments = self.find_crash_segments()
        repaired_data = self.data.copy()

        for segment in segments:
            transition_df = self.compute_transition(
                segment['pre_crash'],
                segment['post_crash']
            )

            if transition_df is None:
                continue

            start_idx = segment['crash_idx'] - len(segment['pre_crash'])
            end_idx = segment['crash_idx'] + len(segment['post_crash'])
            # Ensure we have the correct number of points
            if len(transition_df) != end_idx - start_idx:
                print(f"Warning: Mismatch in transition length {len(transition_df)} vs window length {end_idx - start_idx}")
                continue
            # Create the index range and verify lengths match
            idx_range = pd.RangeIndex(start=start_idx, stop=end_idx)
            # Update the data
            repaired_data.loc[idx_range, 'stim_pos'] = transition_df['stim_pos'].values
            repaired_data.loc[idx_range, 'user_pos'] = transition_df['user_pos'].values
            repaired_data.loc[idx_range, 'flip_time'] = transition_df['flip_time'].values
            repaired_data.loc[idx_range, 'did_crash'] = False
            last_crash_count = repaired_data.loc[start_idx-1, 'crash_count'] if start_idx > 0 else 0
            repaired_data.loc[idx_range, 'crash_count'] = last_crash_count

        repaired_data = self.resample_data(repaired_data, target_frequency=30)
        return repaired_data

    def plot_repair(self, repaired_data, segment_index=0):
        """
        Plot the original and repaired data for a specific crash segment.

        :param repaired_data: DataFrame containing the repaired data.
        :param segment_index: Index of the crash segment to plot.
        :return: Matplotlib figure object.
        """

        segments = self.find_crash_segments()
        if segment_index >= len(segments):
            print(f"Warning: Segment index {segment_index} is out of range")
            return None

        segment = segments[segment_index]
        crash_idx = segment['crash_idx']
        window_size = int(2 * self.window_frame_count)
        start_idx = crash_idx - window_size
        end_idx = crash_idx + window_size

        if start_idx < 0:
            print(f"Warning: Calculated start index {start_idx} is out of bounds.")
            return None
        if end_idx >= len(self.data):
            print(f"Warning: Calculated end index {end_idx} is out of bounds.")
            return None

        half_window = window_size // 2
        span_start = self.data.loc[crash_idx - half_window, 'flip_time']
        span_end = self.data.loc[crash_idx, 'flip_time'] + segment['gap_duration']

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        ax1.plot(self.data.loc[start_idx:end_idx, 'flip_time'],
                self.data.loc[start_idx:end_idx, 'stim_pos'],
                'r--', label='Original Stimulus')
        ax1.plot(repaired_data.loc[start_idx:end_idx, 'flip_time'],
                repaired_data.loc[start_idx:end_idx, 'stim_pos'],
                'b-', label='Repaired Stimulus')
        ax1.axvspan(span_start, span_end, color='gray', alpha=0.2, label='Reset Period')
        ax1.axhline(y=self.target_max_position, color='g', linestyle=':')
        ax1.axhline(y=-self.target_max_position, color='g', linestyle=':')
        ax1.set_ylabel('Stimulus Position')
        ax1.legend()

        ax2.plot(self.data.loc[start_idx:end_idx, 'flip_time'],
                self.data.loc[start_idx:end_idx, 'user_pos'],
                'r--', label='Original User')
        ax2.plot(repaired_data.loc[start_idx:end_idx, 'flip_time'],
                repaired_data.loc[start_idx:end_idx, 'user_pos'],
                'b-', label='Repaired User')
        ax2.axvspan(span_start, span_end, color='gray', alpha=0.2)
        ax2.axhline(y=self.target_max_position, color='g', linestyle=':')
        ax2.axhline(y=-self.target_max_position, color='g', linestyle=':')
        ax2.set_ylabel('User Position')
        ax2.legend()

        # Wider context view around the crash region
        wide_start = max(0, crash_idx - 2 * window_size)
        wide_end = min(len(self.data) - 1, crash_idx + 2 * window_size)
        ax3.plot(self.data.loc[wide_start:wide_end, 'flip_time'],
                self.data.loc[wide_start:wide_end, 'stim_pos'],
                'r--', label='Original Stimulus')
        ax3.plot(self.data.loc[wide_start:wide_end, 'flip_time'],
                self.data.loc[wide_start:wide_end, 'user_pos'],
                'b--', label='Original User')
        ax3.plot(repaired_data.loc[wide_start:wide_end, 'flip_time'],
                repaired_data.loc[wide_start:wide_end, 'stim_pos'],
                'm-', label='Repaired Stimulus')
        ax3.plot(repaired_data.loc[wide_start:wide_end, 'flip_time'],
                repaired_data.loc[wide_start:wide_end, 'user_pos'],
                'g-', label='Repaired User')
        ax3.axvspan(
            self.data.loc[crash_idx - window_size, 'flip_time'],
            self.data.loc[crash_idx, 'flip_time'] + segment['gap_duration'],
            color='gray', alpha=0.2
        )
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Time (s)')
        ax3.legend()

        plt.tight_layout()
        return fig
