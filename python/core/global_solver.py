#!/usr/bin/env python3
"""
Core Global Solver Module
Handles reference selection and weighted least squares optimization
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from core.track_analyser import TrackInfo, PairwiseResult

@dataclass
class GlobalAlignment:
    """Container for global timeline alignment results"""
    reference_track: str
    track_offsets: Dict[str, float]  # Offset of each track relative to global timeline
    residual_error: float  # RMS error from least squares
    connectivity_score: float  # How well connected the tracks are
    solution_confidence: float  # Overall confidence in the solution

class ConflictDetector:
    """Detects triangle inconsistencies in pairwise relationships"""
    
    @staticmethod
    def detect_conflicts(pairwise_results: Dict[Tuple[str, str], PairwiseResult], 
                        min_confidence: float = 50.0) -> List[Dict]:
        """
        Detect triangle inconsistencies: A→B + B→C should ≈ A→C
        """
        conflicts = []
        
        # Get high-confidence pairs only
        high_confidence_pairs = {
            (r.track1, r.track2): r for r in pairwise_results.values() 
            if r.confidence >= min_confidence and r.error is None
        }
        
        # Find all tracks involved
        tracks = set()
        for t1, t2 in high_confidence_pairs.keys():
            tracks.add(t1)
            tracks.add(t2)
        
        track_list = list(tracks)
        
        # Check all possible triangles
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                for k in range(j + 1, len(track_list)):
                    t1, t2, t3 = track_list[i], track_list[j], track_list[k]
                    
                    # Get offsets (handle bidirectional pairs)
                    def get_offset(ta: str, tb: str) -> Optional[float]:
                        if (ta, tb) in high_confidence_pairs:
                            return high_confidence_pairs[(ta, tb)].offset_seconds
                        elif (tb, ta) in high_confidence_pairs:
                            return -high_confidence_pairs[(tb, ta)].offset_seconds
                        return None
                    
                    offset_12 = get_offset(t1, t2)
                    offset_13 = get_offset(t1, t3)
                    offset_23 = get_offset(t2, t3)
                    
                    # Check triangle consistency: offset_12 + offset_23 ≈ offset_13
                    if all(x is not None for x in [offset_12, offset_23, offset_13]):
                        expected_13 = offset_12 + offset_23
                        error = abs(expected_13 - offset_13)
                        
                        if error > 0.1:  # 100ms threshold
                            conflicts.append({
                                'type': 'triangle_inconsistency',
                                'tracks': [t1, t2, t3],
                                'offset_12': offset_12,
                                'offset_23': offset_23,
                                'offset_13_actual': offset_13,
                                'offset_13_expected': expected_13,
                                'error_seconds': error,
                                'severity': 'high' if error > 0.5 else 'medium'
                            })
        
        return conflicts

class GlobalSolver:
    """Solves for optimal global timeline alignment"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def select_reference_track(self, tracks: Dict[str, TrackInfo], 
                              pairwise_results: Dict[Tuple[str, str], PairwiseResult]) -> str:
        """
        Select optimal reference track based on quality + connectivity
        Primary: RMS-based quality score (70%)
        Secondary: Connectivity (30%)
        """
        if not tracks or not pairwise_results:
            raise ValueError("Need tracks and pairwise results for reference selection")
        
        track_scores = {}
        
        for track_path in tracks.keys():
            track_info = tracks[track_path]
            
            # Primary: Quality score (RMS-weighted)
            quality_score = track_info.quality_score
            
            # Secondary: Connectivity score
            connections = 0
            total_confidence = 0.0
            
            for (t1, t2), result in pairwise_results.items():
                if (t1 == track_path or t2 == track_path) and result.error is None and result.confidence >= 50.0:
                    connections += 1
                    total_confidence += result.confidence
            
            avg_confidence = total_confidence / max(connections, 1)
            connectivity_score = connections / max(len(tracks) - 1, 1)  # Normalize
            
            # Combined score: 70% quality, 30% connectivity
            combined_score = 0.70 * quality_score + 0.30 * connectivity_score * (avg_confidence / 100.0)
            
            track_scores[track_path] = {
                'combined_score': combined_score,
                'quality_score': quality_score,
                'connectivity_score': connectivity_score,
                'avg_confidence': avg_confidence
            }
        
        best_track = max(track_scores.keys(), key=lambda x: track_scores[x]['combined_score'])
        
        if self.verbose:
            print(f"📍 Reference Selection (Quality + Connectivity):")
            for track_path, scores in sorted(track_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True):
                name = Path(track_path).name
                print(f"  {name}: {scores['combined_score']:.3f} "
                      f"(Q: {scores['quality_score']:.3f}, C: {scores['connectivity_score']:.3f})")
            print(f"  ✅ Selected: {Path(best_track).name}")
        
        return best_track
    
    def solve_weighted_least_squares(self, tracks: Dict[str, TrackInfo], 
                                    pairwise_results: Dict[Tuple[str, str], PairwiseResult], 
                                    reference_track: str, 
                                    min_confidence: float = 50.0) -> GlobalAlignment:
        """
        Solve weighted least squares optimization
        
        Mathematical formulation:
        minimize: Σ w_ij * (t_j - t_i - offset_ij)²
        subject to: t_reference = 0
        """
        if reference_track not in tracks:
            raise ValueError(f"Reference track not found: {reference_track}")
        
        if self.verbose:
            print(f"🔢 Solving weighted least squares...")
            print(f"   Reference: {Path(reference_track).name}")
        
        # Create track index mapping
        track_list = list(tracks.keys())
        track_to_idx = {track: i for i, track in enumerate(track_list)}
        n_tracks = len(track_list)
        ref_idx = track_to_idx[reference_track]
        
        # Build constraint system: A * t = b with weights W
        A = []  # Constraint matrix
        b = []  # Constraint values  
        w = []  # Confidence weights
        
        # Add pairwise constraints: t_j - t_i = offset_ij
        for (t1, t2), result in pairwise_results.items():
            if result.error is None and result.confidence >= min_confidence:
                if t1 in track_to_idx and t2 in track_to_idx:
                    i, j = track_to_idx[t1], track_to_idx[t2]
                    
                    # Constraint: t_j - t_i = offset_ij
                    constraint = np.zeros(n_tracks)
                    constraint[i] = -1.0  # -t_i
                    constraint[j] = 1.0   # +t_j
                    
                    A.append(constraint)
                    b.append(result.offset_seconds)
                    w.append(result.confidence / 100.0)  # Normalize confidence to [0,1]
        
        if not A:
            raise ValueError("No valid constraints for least squares system")
        
        # Add reference constraint: t_reference = 0 (high weight)
        ref_constraint = np.zeros(n_tracks)
        ref_constraint[ref_idx] = 1.0
        A.append(ref_constraint)
        b.append(0.0)
        w.append(10.0)  # High weight for reference constraint
        
        # Convert to numpy arrays
        A = np.array(A)
        b = np.array(b)
        w = np.array(w)
        
        # Solve weighted least squares: minimize ||W(At - b)||²
        W = np.diag(w)
        WA = W @ A
        Wb = W @ b
        
        try:
            # Normal equations: (A^T W^T W A) t = A^T W^T W b
            track_offsets_array = np.linalg.lstsq(WA, Wb, rcond=None)[0]
            residual_error = np.linalg.norm(A @ track_offsets_array - b)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Least squares solution failed: {str(e)}")
        
        # Create result dictionary
        track_offsets = {track_list[i]: track_offsets_array[i] for i in range(n_tracks)}
        
        # Calculate connectivity score
        total_possible_pairs = len(track_list) * (len(track_list) - 1) // 2
        actual_pairs = len([1 for r in pairwise_results.values() 
                          if r.error is None and r.confidence >= min_confidence])
        connectivity_score = actual_pairs / max(total_possible_pairs, 1)
        
        # Calculate solution confidence
        solution_confidence = self._calculate_solution_confidence(track_offsets, residual_error, pairwise_results)
        
        global_alignment = GlobalAlignment(
            reference_track=reference_track,
            track_offsets=track_offsets,
            residual_error=residual_error,
            connectivity_score=connectivity_score,
            solution_confidence=solution_confidence
        )
        
        if self.verbose:
            print(f"   Residual error: {residual_error:.4f}s")
            print(f"   Solution confidence: {solution_confidence:.1f}%")
        
        return global_alignment
    
    def _calculate_solution_confidence(self, track_offsets: Dict[str, float], 
                                     residual_error: float,
                                     pairwise_results: Dict[Tuple[str, str], PairwiseResult]) -> float:
        """Calculate overall confidence in the global alignment solution"""
        # Factor 1: Low residual error is good
        error_score = max(0, 1 - (residual_error / 0.1))  # 100ms max expected
        
        # Factor 2: Consistency with pairwise measurements
        consistency_errors = []
        for (t1, t2), result in pairwise_results.items():
            if result.error is None and t1 in track_offsets and t2 in track_offsets:
                implied_offset = track_offsets[t2] - track_offsets[t1]
                measured_offset = result.offset_seconds
                consistency_errors.append(abs(implied_offset - measured_offset))
        
        consistency_score = max(0, 1 - (np.mean(consistency_errors) / 0.1)) if consistency_errors else 0
        
        # Combine factors
        overall_confidence = (0.6 * error_score + 0.4 * consistency_score) * 100.0
        return min(overall_confidence, 100.0)
    
    def construct_global_timeline(self, tracks: Dict[str, TrackInfo], 
                                 pairwise_results: Dict[Tuple[str, str], PairwiseResult]) -> GlobalAlignment:
        """
        Complete global timeline construction workflow:
        1. Select reference track
        2. Solve weighted least squares
        """
        if self.verbose:
            print(f"\n🔧 Global Timeline Construction")
            print("-" * 35)
        
        # Step 1: Reference track selection  
        reference_track = self.select_reference_track(tracks, pairwise_results)
        
        # Step 2: Solve weighted least squares
        global_alignment = self.solve_weighted_least_squares(tracks, pairwise_results, reference_track)
        
        if self.verbose:
            print(f"✅ Global timeline constructed")
            print(f"📍 Track positions (relative to reference):")
            sorted_tracks = sorted(global_alignment.track_offsets.items(), key=lambda x: x[1])
            for track_path, offset in sorted_tracks:
                name = Path(track_path).name
                print(f"   {name}: {offset:+.3f}s")
        
        return global_alignment