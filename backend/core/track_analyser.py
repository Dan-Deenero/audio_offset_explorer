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
    track1: str
    track2: str
    offset_seconds: float
    confidence: float
    total_matches: int
    best_count: int
    tolerance_frames: int
    processing_time: float
    error: Optional[str] = None
    # NEW: optional list of candidate offsets
    # each: {'offset_seconds': float, 'confidence': float, 'count': int, 'dominance_vs_best': float}
    candidates: Optional[List[Dict]] = None


class TrackAnalyser:
    """Core audio track analysis and comparison"""
    
    def __init__(self, hop_length=512, target_sr=None, peaks_per_second=20, verbose=False):
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.peaks_per_second = peaks_per_second
        self.verbose = verbose

    def _onset_confidence(self, y1: np.ndarray, y2: np.ndarray, sr: int, offset_s: float) -> float:
        """Rhythmic agreement at the given lag, 0..100."""
        min_ov = int(0.5 * sr)
        if len(y1) < min_ov or len(y2) < min_ov:
            return 0.0
        if offset_s >= 0:
            s1, s2 = int(offset_s * sr), 0
        else:
            s1, s2 = 0, int(-offset_s * sr)
        n = min(len(y1) - s1, len(y2) - s2)
        if n <= min_ov:
            return 0.0
        a = y1[s1:s1 + n]; b = y2[s2:s2 + n]
        o1 = librosa.onset.onset_strength(y=a, sr=sr, hop_length=self.hop_length)
        o2 = librosa.onset.onset_strength(y=b, sr=sr, hop_length=self.hop_length)
        if len(o1) < 4 or len(o2) < 4:
            return 0.0
        o1 = (o1 - o1.mean()) / (o1.std() + 1e-9)
        o2 = (o2 - o2.mean()) / (o2.std() + 1e-9)
        r = float(np.clip(np.dot(o1, o2) / ((np.linalg.norm(o1)+1e-9)*(np.linalg.norm(o2)+1e-9)), -1.0, 1.0))
        r = max(0.0, r)
        return (r ** 0.5) * 100.0

    @staticmethod
    def _preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        if x is None or x.size == 0:
            return x
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - coeff * x[:-1]
        return y

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def _chroma_bonus_transposed(self, y1: np.ndarray, y2: np.ndarray, sr: int, offset_s: float,
                                max_bonus: float = 20.0) -> float:
        """
        Transposition-invariant chroma similarity bonus at the given lag.
        Returns [0..max_bonus]. If overlap too short or drums-only, returns 0.
        """
        min_ov = int(0.8 * sr)  # need ≥0.8 s
        if len(y1) < min_ov or len(y2) < min_ov:
            return 0.0
        if offset_s >= 0:
            s1, s2 = int(offset_s * sr), 0
        else:
            s1, s2 = 0, int(-offset_s * sr)
        n = min(len(y1) - s1, len(y2) - s2)
        if n <= min_ov:
            return 0.0
        a = y1[s1:s1 + n]; b = y2[s2:s2 + n]

        # Chroma for both overlaps
        C1 = librosa.feature.chroma_cqt(y=a, sr=sr, hop_length=self.hop_length, n_chroma=12)
        C2 = librosa.feature.chroma_cqt(y=b, sr=sr, hop_length=self.hop_length, n_chroma=12)
        if float(np.mean(C1)) < 1e-4 or float(np.mean(C2)) < 1e-4:
            return 0.0  # too percussive / low energy

        # Normalize frames to unit length
        C1 = C1 / (np.linalg.norm(C1, axis=0, keepdims=True) + 1e-9)
        C2 = C2 / (np.linalg.norm(C2, axis=0, keepdims=True) + 1e-9)

        T = min(C1.shape[1], C2.shape[1])
        if T < 4:
            return 0.0
        C1 = C1[:, :T]; C2 = C2[:, :T]

        # For each time frame, take best cyclic shift (key transposition invariance)
        # cos_k(t) = max over shifts k of cos( C1(:,t), roll(C2(:,t), k) )
        cos_t = np.zeros(T, dtype=np.float32)
        for t in range(T):
            v1 = C1[:, t]
            best = 0.0
            v2 = C2[:, t]
            for k in range(12):
                best = max(best, self._cosine_sim(v1, np.roll(v2, k)))
            cos_t[t] = best

        sim = float(np.median(np.clip(cos_t, 0.0, 1.0)))  # robust to a few bad frames
        return max_bonus * sim  # linear mapping; tune if needed

    def _refine_offset_local(self, y1: np.ndarray, y2: np.ndarray, sr: int,
                            start_offset_s: float, window_s: float = 0.4,
                            step_s: float = 0.01) -> Tuple[float, float]:
        """
        Optional: refine offset by locally maximizing onset+chroma score within ±window_s.
        Returns (refined_offset_s, refined_conf_addition) where the addition is 0..10 pts.
        """
        lo = start_offset_s - window_s
        hi = start_offset_s + window_s
        best_off = start_offset_s
        best_score = -1.0
        y1p = self._preemphasis(y1)
        y2p = self._preemphasis(y2)
        w_onset, w_chroma = 0.5, 0.5  # weights for refinement objective

        s = lo
        while s <= hi + 1e-9:
            onset = self._onset_confidence(y1p, y2p, sr, s) / 100.0
            chroma = self._chroma_bonus_transposed(y1p, y2p, sr, s, max_bonus=20.0) / 20.0
            combo = w_onset * onset + w_chroma * chroma  # in [0..1]
            if combo > best_score:
                best_score = combo
                best_off = s
            s += step_s

        # Map improvement to a small confidence bump (0..10 pts)
        bump = float(max(0.0, min(10.0, 10.0 * best_score)))
        return best_off, bump

    
    @staticmethod
    def _bar_window_seconds_from_bpm(bpm: float, beats_per_bar: int = 4) -> float:
        """Approximate seconds per bar; fall back handled by caller."""
        if bpm <= 0:
            return 0.0
        seconds_per_beat = 60.0 / bpm
        return beats_per_bar * seconds_per_beat
    
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
        Fingerprints on RAW audio (preserves legacy offsets).
        Score top-N histogram bins (baseline + onset), then conservatively decide:
        only switch away from the top-count bin if the challenger is clearly better
        and close in time (≈ one bar).
        """
        start_time = time.time()
        pps = peaks_per_second or self.peaks_per_second

        try:
            if self.verbose:
                print(f"  🔍 Pairwise: {Path(file1).name} vs {Path(file2).name} (peaks/sec: {pps})")

            # ---- 1) Load mono audio & align SR ----
            y1, sr1 = librosa.load(file1, sr=self.target_sr, mono=True)
            y2, sr2 = librosa.load(file2, sr=self.target_sr, mono=True)

            sr = sr1 if self.target_sr is not None else max(sr1, sr2)
            if sr1 != sr:
                y1 = librosa.resample(y1, orig_sr=sr1, target_sr=sr)
            if sr2 != sr:
                y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr)

            # ---- 2) Fingerprints on RAW audio → time-diff histogram (frames) ----
            hashes1 = self._extract_fingerprints(y1, sr, pps)
            hashes2 = self._extract_fingerprints(y2, sr, pps)
            if not hashes1 or not hashes2:
                return PairwiseResult(file1, file2, 0.0, 0.0, 0, 0, 0, time.time()-start_time,
                                    error="Failed to extract fingerprints")

            from collections import defaultdict, Counter
            index = defaultdict(list)
            for h, t in hashes1:
                index[h].append(t)

            match_pairs = []
            for h, q in hashes2:
                if h in index:
                    for r in index[h]:
                        match_pairs.append((r, q))

            if not match_pairs:
                return PairwiseResult(file1, file2, 0.0, 0.0, 0, 0, 0, time.time()-start_time,
                                    error="No fingerprint matches found")

            diffs = [r - q for r, q in match_pairs]
            counter = Counter(diffs)

            TOP_K = min(5, len(counter)) or 1
            top_bins = counter.most_common(TOP_K)      # [(bin_frame, count), ...]
            top_count_bin, top_count = top_bins[0]

            tol_frames = self._calculate_adaptive_tolerance(diffs, top_count_bin)

            # ---- 3) Prep for confidence signals (do NOT use in fingerprinting) ----
            y1p = self._preemphasis(y1)
            y2p = self._preemphasis(y2)

            clip_dur = min(len(y1), len(y2)) / sr
            SHORT_SEC = 8.0
            short_scale = max(0.25, min(1.0, clip_dur / SHORT_SEC))  # keep tiny clips honest

            # Ambiguity & selection thresholds
            DOM_THR = 1.25
            DELTA_MARGIN = 7.5
            MIN_COUNT_SHORT, MIN_COUNT_LONG = 5, 10

            # One-bar proximity window (sec); estimate from tempo, with safe fallback
            try:
                onset_env = librosa.onset.onset_strength(y=y1, sr=sr, hop_length=self.hop_length)
                tempo = float(librosa.feature.rhythm.tempo(
                    onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, aggregate=None
                ).mean())
                bar_window = 4 * (60.0 / tempo) if tempo > 0 else 2.5
            except Exception:
                bar_window = 2.5

            # ---- 4) Score each candidate (baseline + onset + penalties + short scaling) ----
            candidates = []
            second_count = top_bins[1][1] if len(top_bins) > 1 else top_count

            for bin_frame, count in top_bins:
                base_conf = self._calculate_confidence(diffs, bin_frame, count, tol_frames)  # 0..100
                off_s = int(bin_frame * self.hop_length) / sr

                # onset agreement at this lag (MUST pass offset_s)
                onset_conf = self._onset_confidence(y1p, y2p, sr, off_s)                     # 0..100

                # dominance penalty vs the top bin
                if bin_frame == top_count_bin:
                    dom_vs_top = top_count / max(1.0, float(second_count))
                else:
                    dom_vs_top = top_count / max(1.0, float(count))
                penalty = 0.6 if (bin_frame != top_count_bin and dom_vs_top > DOM_THR) else 1.0

                adj = (0.5 * base_conf + 0.5 * onset_conf)
                adj = min(100.0, adj) * penalty
                adj = adj * short_scale

                candidates.append({
                    'bin_frame': int(bin_frame),
                    'offset_seconds': float(off_s),
                    'count': int(count),
                    'baseline_conf': float(base_conf),
                    'onset_conf': float(onset_conf),
                    'dominance_vs_top': float(dom_vs_top),
                    'adjusted_conf': float(adj),
                })

            # ---- 5) Conservative selection ----
            top_off_s = int(top_count_bin * self.hop_length) / sr
            top_cand = next(c for c in candidates if c['bin_frame'] == int(top_count_bin))
            C_top = top_cand['adjusted_conf']

            best_by_conf = max(candidates, key=lambda c: c['adjusted_conf'])
            C_best = best_by_conf['adjusted_conf']
            count_best = best_by_conf['count']
            off_best = best_by_conf['offset_seconds']

            min_count_req = MIN_COUNT_SHORT if clip_dur < SHORT_SEC else MIN_COUNT_LONG
            close_in_time = abs(off_best - top_off_s) <= bar_window

            if (C_best >= C_top + DELTA_MARGIN) and (count_best >= min_count_req) and close_in_time:
                selected = best_by_conf
            else:
                selected = top_cand

            # damp final confidence if histogram top is weak vs #2 → ambiguous rhythm
            if top_count / max(1.0, float(second_count)) < DOM_THR:
                selected['adjusted_conf'] *= 0.85

            offset_seconds = selected['offset_seconds']
            adjusted_conf = float(max(0.0, min(100.0, selected['adjusted_conf'])))

            return PairwiseResult(
                track1=file1,
                track2=file2,
                offset_seconds=float(offset_seconds),
                confidence=adjusted_conf,
                total_matches=len(match_pairs),
                best_count=max(c['count'] for c in candidates),
                tolerance_frames=tol_frames,
                processing_time=time.time() - start_time,
                error=None,
                # candidates=candidates,  # ← uncomment if your dataclass includes this field
            )

        except Exception as e:
            return PairwiseResult(
                track1=file1, track2=file2, offset_seconds=0.0, confidence=0.0,
                total_matches=0, best_count=0, tolerance_frames=0,
                processing_time=time.time() - start_time,
                error=str(e)
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