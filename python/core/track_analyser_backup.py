#!/usr/bin/env python3
"""
Core Track Analyzer Module
Handles quality assessment and pairwise offset calculations
"""

import numpy as np
import librosa
from collections import Counter, defaultdict
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class TrackInfo:
    """Container for track metadata and quality metrics"""
    filepath: str
    duration: float
    sample_rate: int
    rms_energy: float
    dynamic_range: float
    snr_estimate: float
    spectral_richness: float
    consistency: float
    quality_score: float

@dataclass
class PairwiseResult:
    """Container for pairwise offset calculation results"""
    track1: str
    track2: str
    offset_seconds: float
    confidence: float
    total_matches: int
    best_count: int
    tolerance_frames: int
    processing_time: float
    error: Optional[str] = None

class TrackAnalyser:
    """Core audio track analysis and comparison"""
    
    def __init__(self, hop_length=512, target_sr=None, peaks_per_second=20, verbose=False):
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.peaks_per_second = peaks_per_second
        self.verbose = verbose
    
    def calculate_quality_metrics(self, audio_path: str) -> TrackInfo:
        """
        Calculate comprehensive quality metrics for a track
        Primary metric: RMS energy (as per original plan)
        Secondary metrics: dynamic range, SNR, spectral richness, consistency
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.verbose:
            print(f"  📊 Quality assessment: {Path(audio_path).name}")
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")
        
        duration = len(audio) / sr
        
        # PRIMARY: RMS energy assessment (as per original plan)
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # SECONDARY: Supporting quality metrics
        peak_amplitude = np.max(np.abs(audio))
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # SNR estimate: compare signal power to noise floor (quietest 10%)
        sorted_magnitudes = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_magnitudes[:int(0.1 * len(sorted_magnitudes))])
        snr_estimate = 10 * np.log10((rms_energy**2) / (noise_floor**2 + 1e-10))
        
        # Spectral richness: variation in spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_richness = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)
        
        # Temporal consistency: frame-to-frame RMS variation
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        consistency = 1.0 / (1.0 + np.std(frame_rms) / (np.mean(frame_rms) + 1e-10))
        
        # Composite quality score (RMS-weighted as per plan)
        quality_score = (
            0.50 * min(rms_energy * 10, 1.0) +  # RMS is primary (50%)
            0.20 * min(max(dynamic_range, 0) / 60.0, 1.0) +  # Dynamic range
            0.15 * min(max(snr_estimate, 0) / 60.0, 1.0) +   # SNR
            0.10 * min(spectral_richness, 1.0) +             # Spectral richness  
            0.05 * consistency                               # Consistency
        )
        
        if self.verbose:
            print(f"    • Duration: {duration:.1f}s, RMS: {rms_energy:.4f}")
            print(f"    • Quality score: {quality_score:.3f}")
        
        return TrackInfo(
            filepath=audio_path,
            duration=duration,
            sample_rate=sr,
            rms_energy=rms_energy,
            dynamic_range=dynamic_range,
            snr_estimate=snr_estimate,
            spectral_richness=spectral_richness,
            consistency=consistency,
            quality_score=quality_score
        )
    
    def calculate_pairwise_offset(self, file1: str, file2: str, 
                                 peaks_per_second: Optional[int] = None) -> PairwiseResult:
        """
        Calculate offset between two tracks with configurable peak density
        """
        start_time = time.time()
        pps = peaks_per_second or self.peaks_per_second
        
        try:
            if self.verbose:
                print(f"  🔍 Pairwise: {Path(file1).name} vs {Path(file2).name} (peaks/sec: {pps})")
            
            # Load audio files
            audio1, sr1 = librosa.load(file1, sr=self.target_sr, mono=True)
            audio2, sr2 = librosa.load(file2, sr=self.target_sr, mono=True)
            
            # Handle sample rate consistency
            sr = sr1 if self.target_sr is None else self.target_sr
            if sr1 != sr2 and self.target_sr is None:
                sr = max(sr1, sr2)
                if sr1 != sr:
                    audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr)
                if sr2 != sr:
                    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr)
            
            # Extract fingerprints with configurable peak density
            hashes1 = self._extract_fingerprints(audio1, sr, pps)
            hashes2 = self._extract_fingerprints(audio2, sr, pps)
            
            if not hashes1 or not hashes2:
                return PairwiseResult(
                    track1=file1, track2=file2, offset_seconds=0.0, confidence=0.0,
                    total_matches=0, best_count=0, tolerance_frames=0,
                    processing_time=time.time() - start_time,
                    error="Failed to extract fingerprints"
                )
            
            # Find matches
            index = defaultdict(list)
            for h, t in hashes1:
                index[h].append(t)
            
            match_pairs = []
            for h, q in hashes2:
                if h in index:
                    for r in index[h]:
                        match_pairs.append((r, q))
            
            if not match_pairs:
                return PairwiseResult(
                    track1=file1, track2=file2, offset_seconds=0.0, confidence=0.0,
                    total_matches=0, best_count=0, tolerance_frames=0,
                    processing_time=time.time() - start_time,
                    error="No fingerprint matches found"
                )
            
            # Find best offset
            time_diffs = [r - q for r, q in match_pairs]
            time_diff_counter = Counter(time_diffs)
            best_offset_candidate, best_count = time_diff_counter.most_common(1)[0]
            
            # Adaptive tolerance (default = 2, as established)
            tolerance_frames = self._calculate_adaptive_tolerance(time_diffs, best_offset_candidate)
            
            # Confidence calculation
            confidence = self._calculate_confidence(time_diffs, best_offset_candidate, best_count, tolerance_frames)
            
            # Convert to time units
            offset_samples = int(best_offset_candidate * self.hop_length)
            offset_seconds = offset_samples / sr
            
            result = PairwiseResult(
                track1=file1, track2=file2, offset_seconds=offset_seconds, confidence=confidence,
                total_matches=len(match_pairs), best_count=best_count, tolerance_frames=tolerance_frames,
                processing_time=time.time() - start_time
            )
            
            if self.verbose:
                print(f"    • Offset: {offset_seconds:.3f}s, Confidence: {confidence:.1f}%")
            
            return result
            
        except Exception as e:
            return PairwiseResult(
                track1=file1, track2=file2, offset_seconds=0.0, confidence=0.0,
                total_matches=0, best_count=0, tolerance_frames=0,
                processing_time=time.time() - start_time, error=str(e)
            )
    
    def _extract_fingerprints(self, audio: np.ndarray, sr: int, peaks_per_second: int) -> List[Tuple[int, int]]:
        """Extract fingerprints using configurable peak density"""
        duration_seconds = len(audio) / sr
        total_target_peaks = int(peaks_per_second * duration_seconds)
        
        # Mel-spectrogram peak detection
        db_mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=self.hop_length), ref=np.max
        )
        
        potential_peaks = []
        for mel_idx in range(1, db_mel_spec.shape[0] - 1):
            for time_idx in range(1, db_mel_spec.shape[1] - 1):
                if db_mel_spec[mel_idx, time_idx] >= np.max(
                    db_mel_spec[mel_idx-1:mel_idx+2, time_idx-1:time_idx+2]
                ):
                    potential_peaks.append((db_mel_spec[mel_idx, time_idx], time_idx, mel_idx))
        
        potential_peaks.sort(key=lambda x: x[0], reverse=True)
        peaks = [(t, f) for db, t, f in potential_peaks[:total_target_peaks]]
        
        # Generate hashes
        return self._generate_hashes(peaks)
    
    def _generate_hashes(self, peaks: List[Tuple[int, int]], fan_value=15, max_time_delta=60):
        """Generate fingerprint hashes"""
        fingerprints = []
        peaks.sort(key=lambda x: x[0])
        
        for i in range(len(peaks)):
            for j in range(1, fan_value + 1):
                if i + j < len(peaks):
                    t1, f1 = peaks[i][0], peaks[i][1]
                    t2, f2 = peaks[i+j][0], peaks[i+j][1]
                    delta_t = t2 - t1
                    if 0 < delta_t <= max_time_delta:
                        h = hash((f1, f2, delta_t))
                        fingerprints.append((h, t1))
        
        return fingerprints
    
    def _calculate_adaptive_tolerance(self, time_diffs: List[int], best_offset: int) -> int:
        """Calculate adaptive tolerance (default=2, adjust based on distribution)"""
        if not time_diffs:
            return 2
        
        deviations = [abs(diff - best_offset) for diff in time_diffs if diff != best_offset]
        if not deviations:
            return 1
            
        std_dev = np.std(deviations)
        
        if std_dev < 0.8:
            return 1    # Very tight - be strict
        elif std_dev < 2.0:
            return 2    # Good distribution - default
        elif std_dev < 4.0:
            return 3    # Loose - need flexibility
        else:
            return min(4, int(std_dev))  # Very loose - cap at 4
    
    def _calculate_confidence(self, time_diffs: List[int], best_offset: int, 
                            best_count: int, tolerance_frames: int) -> float:
        """Enhanced confidence calculation with adaptive tolerance"""
        total_matches = len(time_diffs)
        if total_matches == 0:
            return 0.0
        
        # Multiple confidence metrics
        inlier_ratio = best_count / total_matches
        
        # Concentration with adaptive tolerance
        inliers_with_tolerance = sum(1 for diff in time_diffs if abs(diff - best_offset) <= tolerance_frames)
        concentration_ratio = inliers_with_tolerance / total_matches
        
        # Strict ratio (tolerance = 1)
        strict_inliers = sum(1 for diff in time_diffs if abs(diff - best_offset) <= 1)
        strict_ratio = strict_inliers / total_matches
        
        # Signal strength
        counter = Counter(time_diffs)
        sorted_counts = sorted(counter.values(), reverse=True)
        signal_strength = min(sorted_counts[0] / (sorted_counts[1] + 1e-6) / 10.0, 1.0) if len(sorted_counts) > 1 else 0.5
        
        # Composite confidence
        confidence_raw = (
            0.30 * concentration_ratio +
            0.25 * strict_ratio +
            0.25 * inlier_ratio +
            0.20 * signal_strength
        )
        
        # Calibrate to percentage
        if confidence_raw < 0.02:
            confidence_percentage = confidence_raw * 500
        elif confidence_raw < 0.10:
            confidence_percentage = 10 + (confidence_raw - 0.02) * 625
        else:
            confidence_percentage = 60 + (confidence_raw - 0.10) * 444
        
        return min(confidence_percentage, 100.0)