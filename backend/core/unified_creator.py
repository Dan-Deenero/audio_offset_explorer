#!/usr/bin/env python3
"""
Core Unified Creator Module
Handles track alignment and unification into single output
"""

import numpy as np
import librosa
import soundfile as sf
import time
from pathlib import Path
from typing import Dict, Tuple

from core.track_analyser import TrackInfo
from core.global_solver import GlobalAlignment

class UnifiedCreator:
    """Creates unified audio track from globally aligned tracks"""
    
    def __init__(self, target_sr=None, verbose=False):
        self.target_sr = target_sr
        self.verbose = verbose
    
    def create_unified_timeline(self, tracks: Dict[str, TrackInfo], 
                               global_alignment: GlobalAlignment,
                               output_path: str = "unified.wav", 
                               merge_method: str = 'intelligent') -> Tuple[np.ndarray, int]:
        """
        Create unified track using the global alignment solution
        
        Args:
            tracks: Dictionary of track info
            global_alignment: Global timeline solution
            output_path: Where to save unified track
            merge_method: How to merge overlapping audio ('intelligent', 'average', 'quality_weighted')
            
        Returns:
            (unified_audio, sample_rate)
        """
        if self.verbose:
            print(f"\n🎼 Creating Unified Track: {output_path}")
            print(f"   Merge method: {merge_method}")
        
        start_time = time.time()
        
        # Load all audio tracks
        audio_tracks = {}
        target_sr = None
        
        for track_path in tracks.keys():
            try:
                audio, sr = librosa.load(track_path, sr=self.target_sr, mono=True)
                audio_tracks[track_path] = audio
                
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    # Resample if necessary
                    audio_tracks[track_path] = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Warning: Failed to load {track_path}: {str(e)}")
                continue
        
        if not audio_tracks:
            raise ValueError("No audio tracks could be loaded")
        
        # Calculate global timeline bounds
        track_offsets = global_alignment.track_offsets
        
        earliest_start = min(track_offsets.values())
        latest_end = max(
            track_offsets[track] + len(audio_tracks[track]) / target_sr 
            for track in track_offsets.keys() if track in audio_tracks
        )
        
        total_duration = latest_end - earliest_start
        total_samples = int(total_duration * target_sr)
        
        if self.verbose:
            print(f"📏 Timeline: {earliest_start:.3f}s to {latest_end:.3f}s ({total_duration:.3f}s)")
            print(f"   Total samples: {total_samples}")
        
        # Create aligned audio arrays
        aligned_tracks = {}
        
        for track_path, offset in track_offsets.items():
            if track_path not in audio_tracks:
                continue
                
            audio = audio_tracks[track_path]
            
            # Calculate position in global timeline
            start_sample = int((offset - earliest_start) * target_sr)
            end_sample = start_sample + len(audio)
            
            # Create aligned array
            aligned_audio = np.zeros(total_samples, dtype=np.float32)
            
            if start_sample >= 0 and start_sample < total_samples:
                copy_end = min(end_sample, total_samples)
                copy_length = copy_end - start_sample
                if copy_length > 0:
                    aligned_audio[start_sample:copy_end] = audio[:copy_length]
            
            aligned_tracks[track_path] = aligned_audio
        
        # Apply merge method
        if merge_method == 'intelligent':
            unified_audio = self._intelligent_merge(aligned_tracks, tracks, target_sr)
        elif merge_method == 'average':
            unified_audio = self._average_merge(aligned_tracks)
        elif merge_method == 'quality_weighted':
            unified_audio = self._quality_weighted_merge(aligned_tracks, tracks)
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
        
        # Save unified track
        sf.write(output_path, unified_audio, target_sr)
        
        processing_time = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Unified track created: {output_path}")
            print(f"   Duration: {len(unified_audio)/target_sr:.2f}s at {target_sr} Hz")
            print(f"   Processing time: {processing_time:.2f}s")
        
        return unified_audio, target_sr
    
    def _intelligent_merge(self, aligned_tracks: Dict[str, np.ndarray], 
                          tracks: Dict[str, TrackInfo], sr: int) -> np.ndarray:
        """
        Intelligent merge using RMS activity detection
        Only averages where multiple tracks are active
        """
        if self.verbose:
            print("🧠 Using intelligent merge method...")
        
        track_arrays = list(aligned_tracks.values())
        n_samples = len(track_arrays[0])
        
        # Calculate activity masks for each track
        window_size = min(1024, max(256, n_samples // 1000))
        
        def calculate_activity_mask(audio, threshold_factor=0.001):
            """Calculate where audio has significant content using RMS energy"""
            rms_energy = np.array([
                np.sqrt(np.mean(audio[max(0, i-window_size//2):i+window_size//2]**2))
                for i in range(len(audio))
            ])
            threshold = np.max(rms_energy) * threshold_factor
            return rms_energy > threshold
        
        masks = [calculate_activity_mask(audio) for audio in track_arrays]
        
        # Create unified audio
        unified_audio = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            active_tracks = [j for j, mask in enumerate(masks) if mask[i]]
            
            if active_tracks:
                # Average only the active tracks
                active_samples = [track_arrays[j][i] for j in active_tracks]
                unified_audio[i] = np.mean(active_samples)
            # else: leave as silence (0)
        
        if self.verbose:
            total_active_time = sum(np.sum(mask) for mask in masks) / sr
            print(f"   Total active audio time: {total_active_time:.1f}s")
        
        return unified_audio
    
    def _average_merge(self, aligned_tracks: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple averaging of all tracks"""
        if self.verbose:
            print("📊 Using average merge method...")
        
        track_arrays = list(aligned_tracks.values())
        unified_audio = np.mean(track_arrays, axis=0).astype(np.float32)
        
        return unified_audio
    
    def _quality_weighted_merge(self, aligned_tracks: Dict[str, np.ndarray], 
                               tracks: Dict[str, TrackInfo]) -> np.ndarray:
        """
        Quality-weighted merge based on track quality scores
        """
        if self.verbose:
            print("⚖️  Using quality-weighted merge method...")
        
        # Calculate normalized weights
        track_weights = {}
        total_weight = 0.0
        
        for track_path in aligned_tracks.keys():
            if track_path in tracks:
                weight = tracks[track_path].quality_score
                track_weights[track_path] = weight
                total_weight += weight
        
        # Normalize weights
        for track_path in track_weights:
            track_weights[track_path] /= total_weight
        
        if self.verbose:
            print("   Quality weights:")
            for track_path, weight in track_weights.items():
                name = Path(track_path).name
                print(f"     {name}: {weight:.3f}")
        
        # Weighted sum
        unified_audio = np.zeros_like(list(aligned_tracks.values())[0])
        
        for track_path, audio in aligned_tracks.items():
            if track_path in track_weights:
                weight = track_weights[track_path]
                unified_audio += weight * audio
        
        return unified_audio.astype(np.float32)
    
    def create_from_single_reference(self, reference_track: str, 
                                   new_track: str, 
                                   offset_seconds: float,
                                   output_path: str = "aligned.wav") -> Tuple[np.ndarray, int]:
        """
        Create aligned track from single reference + new track
        Useful for incremental additions
        
        Args:
            reference_track: Path to reference audio
            new_track: Path to new track to align
            offset_seconds: Calculated offset of new_track relative to reference
            output_path: Where to save result
            
        Returns:
            (unified_audio, sample_rate)
        """
        if self.verbose:
            print(f"\n🔗 Creating aligned track from reference")
            print(f"   Reference: {Path(reference_track).name}")
            print(f"   New track: {Path(new_track).name}")
            print(f"   Offset: {offset_seconds:.3f}s")
        
        # Load audio files
        ref_audio, ref_sr = librosa.load(reference_track, sr=self.target_sr, mono=True)
        new_audio, new_sr = librosa.load(new_track, sr=self.target_sr, mono=True)
        
        # Use consistent sample rate
        sr = ref_sr if self.target_sr is None else self.target_sr
        if ref_sr != sr:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sr)
        if new_sr != sr:
            new_audio = librosa.resample(new_audio, orig_sr=new_sr, target_sr=sr)
        
        # Apply offset alignment
        offset_samples = int(offset_seconds * sr)
        
        if offset_samples >= 0:
            # New track starts after reference
            padded_ref = ref_audio
            padded_new = np.pad(new_audio, (offset_samples, 0), 'constant')
        else:
            # New track starts before reference
            pad_amount = abs(offset_samples)
            padded_ref = np.pad(ref_audio, (pad_amount, 0), 'constant')
            padded_new = new_audio
        
        # Align to same length
        len1, len2 = len(padded_ref), len(padded_new)
        final_length = max(len1, len2)
        
        aligned_ref = np.pad(padded_ref, (0, final_length - len1), 'constant')
        aligned_new = np.pad(padded_new, (0, final_length - len2), 'constant')
        
        # Simple intelligent merge
        unified_audio = self._simple_intelligent_merge(aligned_ref, aligned_new, sr)
        
        # Save result
        sf.write(output_path, unified_audio, sr)
        
        if self.verbose:
            print(f"✅ Aligned track saved: {output_path}")
            print(f"   Duration: {len(unified_audio)/sr:.2f}s")
        
        return unified_audio, sr
    
    def _simple_intelligent_merge(self, audio1: np.ndarray, audio2: np.ndarray, sr: int) -> np.ndarray:
        """Simple intelligent merge for two tracks"""
        window_size = min(1024, max(256, len(audio1) // 1000))
        
        def calculate_activity_mask(audio, threshold_factor=0.001):
            rms_energy = np.array([
                np.sqrt(np.mean(audio[max(0, i-window_size//2):i+window_size//2]**2))
                for i in range(len(audio))
            ])
            threshold = np.max(rms_energy) * threshold_factor
            return rms_energy > threshold
        
        mask1 = calculate_activity_mask(audio1)
        mask2 = calculate_activity_mask(audio2)
        
        unified_audio = np.zeros_like(audio1)
        
        # Use track 1 where only it's active
        only_1_active = mask1 & ~mask2
        unified_audio[only_1_active] = audio1[only_1_active]
        
        # Use track 2 where only it's active
        only_2_active = ~mask1 & mask2
        unified_audio[only_2_active] = audio2[only_2_active]
        
        # Average where both are active
        both_active = mask1 & mask2
        unified_audio[both_active] = (audio1[both_active] + audio2[both_active]) / 2.0
        
        return unified_audio.astype(np.float32)