#!/usr/bin/env python3
"""
Video Match Against Multiple Unified References (Production-Ready Preparation Engine)
Complete workflow: Trim videos → Extract audio → Match → Create ABR renditions for all outputs.
"""

import argparse
import sys
import os
import time
import json
import glob
import subprocess
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Union
from fractions import Fraction

import ffmpeg

from core.track_analyser import TrackAnalyser

# --- Constants and Profiles ---
SHORT_SEC = 8.0
LONG_SEC  = 12.0

ABR_PROFILES = {
    "1080": {"vb": "5000k", "maxrate": "6000k", "bufsize": "7500k", "w": 1920},
    "720":  {"vb": "2500k", "maxrate": "3000k", "bufsize": "4000k", "w": 1280},
    "480":  {"vb": "1400k", "maxrate": "1700k", "bufsize": "2100k", "w": 854},
}

# -------------------------------
# Reporting
# -------------------------------
def write_compact_offsets_report(
    processed_dir: str, unified_name: str, results: List[Dict], unified_renditions: Optional[Dict],
    unified_duration: float, analysis_audio_name: str, args: argparse.Namespace, is_unified_audio: bool
) -> str:
    """
    Write a compact JSON with paths, durations, offsets, and the final consolidated metadata structure.
    """
    os.makedirs(processed_dir, exist_ok=True)

    items = []
    for r in results:
        if r.get('status') != 'success' or r.get('decision') not in ('green', 'yellow'):
            continue
        if args.reject_positive_offsets and r.get('raw_offset_seconds', 0) > 0:
            continue

        signed = r.get('final_offset_seconds', 0)
        display = -signed

        aligned_video_rel = []
        if r.get('aligned_renditions'):
            for height, data in r['aligned_renditions'].items():
                aligned_video_rel.append({
                    "resolution": f"{height}p",
                    "file": data['file'],
                    "width": data['width'],
                    "height": int(height)
                })

        item_data = {
            "video_file": r.get("video_file"),
            "decision": r.get("decision"),
            "accepted": r.get("decision") == "green",
            "confidence": round(float(r.get("confidence", 0.0)), 1),
            "final_offset_seconds": int(round(display)),
            "signed_offset_seconds": int(round(signed)),
            "aligned_video_rel": sorted(aligned_video_rel, key=lambda x: x['height'], reverse=True) if aligned_video_rel else r.get("aligned_video_file"),
            "aligned_audio_rel": r.get("aligned_audio_file"),
            "metadata": {"tags": r.get("tags", [])}
        }

        if r.get("durations"):
            item_data["durations"] = r["durations"]
        if "sanity_warning" in r:
            item_data["sanity_warning"] = r["sanity_warning"]
        
        items.append(item_data)

    unified_renditions_list = []
    if unified_renditions:
        for height, data in unified_renditions.items():
            unified_renditions_list.append({
                "resolution": f"{height}p",
                "file": data['file'],
                "width": data['width'],
                "height": int(height)
            })

    payload = {
        "unified_reference": {
            "reference_name": unified_name,
            "is_audio_only": is_unified_audio,
            "trimmed_rel": sorted(unified_renditions_list, key=lambda x: x['height'], reverse=True) if not is_unified_audio else None,
            "duration": unified_duration,
            "analysis_audio_rel": analysis_audio_name,
            "metadata": {}
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dir": os.path.basename(processed_dir) or "processed",
        "items": items,
    }
    out_path = os.path.join(processed_dir, f"offsets_{Path(unified_name).stem}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def export_batch_results(batch_result: Dict, output_dir: str, args: argparse.Namespace) -> str:
    """Exports the detailed, verbose JSON report for a single batch run (timestamped, atomic write)."""
    os.makedirs(output_dir, exist_ok=True)
    unified_name = Path(batch_result['unified_reference']['reference_name']).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    export_filename = f"export_detailed_{unified_name}_{timestamp}.json"
    export_path = os.path.join(output_dir, export_filename)

    run_parameters = vars(args).copy()
    if 'output_resolutions' in run_parameters and run_parameters['output_resolutions'] is not None:
        run_parameters['output_resolutions'] = sorted(int(r) for r in run_parameters['output_resolutions'])

    payload = dict(batch_result)
    payload['parameters'] = run_parameters

    tmp_path = export_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, export_path)

    return export_path

# -------------------------------
# Video & Audio Helpers
# -------------------------------
def get_video_info(video_path: str) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], Optional[str]]:
    """
    Probe a video and return: (duration_seconds, fps_float, width, height, fps_str)
    """
    try:
        probe = ffmpeg.probe(video_path)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            return None, None, None, None, None

        duration = fmt.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None
        if not duration:
            try:
                duration = float(video_stream.get("duration", 0.0))
            except (TypeError, ValueError):
                duration = None

        fps_str = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")

        fps_float = 0.0
        if fps_str:
            try:
                frac = Fraction(fps_str)
                if frac.numerator > 0 and frac.denominator > 0:
                    fps_float = float(frac)
            except (ValueError, ZeroDivisionError):
                try:
                    fps_float = float(fps_str)
                except (TypeError, ValueError):
                    fps_float = 0.0

        try:
            width, height = int(video_stream.get("width")), int(video_stream.get("height"))
        except (TypeError, ValueError):
            width = height = None

        return duration, fps_float, width, height, fps_str
    except Exception:
        return None, None, None, None, None


def extract_audio_from_video(
    video_path: str,
    output_path: str,
    audio_format: str,
    sample_rate: Optional[int],
    bitrate: Optional[str],
    channels: Optional[int],
    verbose: bool
) -> bool:
    """
    Extract audio, defaulting to 44.1 kHz mono 16-bit PCM for WAV.
    """
    if verbose:
        print(f"🔄 Extracting audio from {Path(video_path).name} as {audio_format}...")
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Defaults for analysis/final export if not provided
        if sample_rate is None:
            sample_rate = 44100
        if channels is None:
            channels = 1

        cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-i', video_path, '-vn', '-ar', str(sample_rate), '-ac', str(channels)]

        if audio_format == 'wav':
            cmd += ['-acodec', 'pcm_s16le']
        elif audio_format == 'mp3':
            cmd += ['-acodec', 'libmp3lame']
            cmd += ['-b:a', bitrate] if bitrate else ['-q:a', '0']
        elif audio_format == 'flac':
            cmd += ['-acodec', 'flac', '-compression_level', '8']
        elif audio_format == 'aac':
            cmd += ['-acodec', 'aac', '-b:a', bitrate or '320k']

        cmd.append(output_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            if verbose:
                print(f"   ✅ Success: {Path(output_path).name}")
            return True
        else:
            if verbose:
                print(f"   ❌ FFmpeg error: {result.stderr.strip()}")
            return False
    except Exception as e:
        if verbose:
            print(f"   ❌ Error: {str(e)}")
        return False


def get_audio_duration(file_path: str) -> float:
    try:
        import librosa
        return float(librosa.get_duration(path=file_path))
    except Exception:
        pass
    try:
        probe = ffmpeg.probe(file_path)
        dur = probe.get('format', {}).get('duration')
        return float(dur) if dur else 0.0
    except Exception:
        return 0.0


def discover_video_files(directory: str, pattern: str = "*.*", recursive: bool = False) -> List[str]:
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp', '.3g2', '.mpg', '.mpeg', '.m2v', '.divx', '.xvid', '.asf', '.rm', '.rmvb', '.vob', '.ogv', '.f4v', '.mts', '.m2ts', '.ts'}
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    pattern_path = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
    candidate_files = glob.glob(pattern_path, recursive=recursive)
    if pattern == "*.*":
        candidate_files = [f for f in candidate_files if Path(f).suffix.lower() in VIDEO_EXTENSIONS]
    if not candidate_files:
        raise ValueError(f"No video files found matching pattern in: {directory}")
    return sorted(candidate_files)


def discover_unified_references(unified_dir: str, pattern: str = "*.*", recursive: bool = False) -> List[str]:
    UNIFIED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp', '.3g2', '.mpg', '.mpeg', '.m2v', '.divx', '.xvid', '.asf', '.rm', '.rmvb', '.vob', '.ogv', '.f4v', '.mts', '.m2ts', '.ts'}
    if not os.path.exists(unified_dir):
        raise FileNotFoundError(f"Unified directory not found: {unified_dir}")
    if not os.path.isdir(unified_dir):
        raise NotADirectoryError(f"Path is not a directory: {unified_dir}")
    pattern_path = os.path.join(unified_dir, "**", pattern) if recursive else os.path.join(unified_dir, pattern)
    candidate_files = glob.glob(pattern_path, recursive=recursive)
    if pattern == "*.*":
        candidate_files = [f for f in candidate_files if Path(f).suffix.lower() in UNIFIED_EXTENSIONS]
    if not candidate_files:
        raise ValueError(f"No audio/video files found matching pattern in unified directory: {unified_dir}")
    return sorted(candidate_files)


def is_audio_file(file_path: str) -> bool:
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    return Path(file_path).suffix.lower() in AUDIO_EXTENSIONS


def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("❌ FFmpeg not found or not working. Please install and ensure it's in your system's PATH.")
        sys.exit(1)


def choose_r_value(fps_float: Optional[float], fps_str: Optional[str], default_fps: float = 30.0) -> Union[str, float]:
    if fps_str and "/" in fps_str:
        try:
            frac = Fraction(fps_str)
            if frac.numerator > 0 and frac.denominator > 0:
                return fps_str
        except (ValueError, ZeroDivisionError):
            pass
    return fps_float if (fps_float and fps_float > 0) else default_fps


def choose_g_value(fps_float: Optional[float], default_fps: float = 30.0) -> int:
    base = fps_float if (fps_float and fps_float > 0) else default_fps
    return max(1, int(round(base)))


def choose_r_g(fps_float: Optional[float], fps_str: Optional[str], default_fps: float = 30.0) -> Tuple[Union[str, float], int]:
    return choose_r_value(fps_float, fps_str, default_fps), choose_g_value(fps_float, default_fps)


def trim_video_precise(input_path: str, output_path: str, start_offset: float = 0.0, target_duration: Optional[float] = None, method: str = "floor") -> bool:
    duration, fps_float, _, _, fps_str = get_video_info(input_path)
    if duration is None:
        return False
    r_value, g_value = choose_r_g(fps_float, fps_str, default_fps=30.0)

    if target_duration is None:
        remaining = max(0.0, duration - start_offset)
        if method == "ceil":
            target_duration = math.ceil(remaining)
        elif method == "round":
            target_duration = round(remaining)
        else:
            target_duration = math.floor(remaining)
    if target_duration <= 0:
        return False

    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        inp = ffmpeg.input(input_path)
        output_ext = os.path.splitext(output_path)[1].lower()
        common_kwargs = dict(
            ss=start_offset, t=target_duration,
            vcodec="libx264", acodec="aac",
            r=r_value, vsync="cfr",
            g=g_value, sc_threshold=0, bf=0,
            pix_fmt="yuv420p", ar=48000, ac=1,
            preset="veryfast", crf=20,
            **{"force_key_frames": "expr:gte(t,n_forced*1)"},
            loglevel="error"
        )
        if output_ext == ".mov":
            try:
                ffmpeg.output(inp, output_path, **common_kwargs, **{"vtag": "avc1", "movflags": "+faststart"}).overwrite_output().run()
            except Exception:
                try:
                    ffmpeg.output(inp, output_path, **common_kwargs, **{"f": "mp4"}).overwrite_output().run()
                except Exception:
                    mp4_output = output_path.replace(".mov", ".mp4")
                    ffmpeg.output(inp, mp4_output, **common_kwargs, **{"movflags": "+faststart"}).overwrite_output().run()
                    output_path = mp4_output
        else:
            container_opts = {"movflags": "+faststart"} if output_ext == ".mp4" else {}
            ffmpeg.output(inp, output_path, **common_kwargs, **container_opts).overwrite_output().run()

        verify_duration, _, _, _, _ = get_video_info(output_path)
        if verify_duration and abs(verify_duration - target_duration) > 0.05:
            print(f"Warning: Duration verification failed: got {verify_duration:.3f}s, expected {target_duration:.3f}s")
            return False
        return True
    except Exception:
        return False


def trim_and_create_renditions(input_path: str, output_stem: str, start_offset: float, target_duration: float, resolutions: Set[int], verbose: bool) -> Dict[str, Dict]:
    _, fps_float, _, _, fps_str = get_video_info(input_path)
    r_value, g_value = choose_r_g(fps_float, fps_str, default_fps=30.0)

    if verbose:
        disp_r = r_value if isinstance(r_value, str) else f"{r_value:.3f}"
        print(f"Trimming & creating {len(resolutions)} renditions for {Path(input_path).name} (r={disp_r}, g={g_value}, keyframe@1s)")

    output_metadata: Dict[str, Dict] = {}
    out_dir = os.path.dirname(output_stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for height in sorted(list(resolutions), reverse=True):
        profile = ABR_PROFILES.get(str(height))
        if not profile:
            if verbose:
                print(f"   ⚠️ No ABR profile for {height}p — skipping.")
            continue

        output_filename = f"{os.path.basename(output_stem)}_{height}p.mp4"
        output_path = os.path.join(out_dir, output_filename) if out_dir else output_filename
        try:
            inp = ffmpeg.input(input_path, ss=start_offset)
            stream_v = inp.video.filter("scale", -2, height)
            ffmpeg.output(
                stream_v, inp.audio, output_path, t=target_duration,
                vcodec="libx264", acodec="aac", preset="veryfast", crf=21,
                pix_fmt="yuv420p",  # explicit on primary path
                **{"b:v": profile["vb"], "maxrate": profile["maxrate"], "bufsize": profile["bufsize"]},
                r=r_value, vsync="cfr", g=g_value, sc_threshold=0,
                **{"force_key_frames": "expr:gte(t,n_forced*1)"},
                **{"movflags": "+faststart"},
                ar=48000, ac=2, **{"b:a": "128k"}
            ).run(cmd="ffmpeg", quiet=not verbose, overwrite_output=True)
            output_metadata[str(height)] = {"file": output_filename, "width": profile["w"], "height": int(height)}
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Primary rendition for {height}p failed, trying fallback. Error: {e}")
            try:
                inp = ffmpeg.input(input_path, ss=start_offset)
                stream_v = inp.video.filter("scale", -2, height)
                ffmpeg.output(
                    stream_v, inp.audio, output_path, t=target_duration,
                    vcodec="libx264", acodec="aac",
                    r=r_value, vsync="cfr", pix_fmt="yuv420p", bf=0,
                    g=g_value, sc_threshold=0,
                    **{"b:v": profile["vb"], "maxrate": profile["maxrate"], "bufsize": profile["bufsize"]},
                    **{"force_key_frames": "expr:gte(t,n_forced*1)"},
                    **{"movflags": "+faststart"},
                    ar=48000, ac=2, **{"b:a": "128k"}
                ).run(cmd="ffmpeg", quiet=not verbose, overwrite_output=True)
                output_metadata[str(height)] = {"file": output_filename, "width": profile["w"], "height": int(height)}
                if verbose:
                    print(f"   ✅ Fallback for {height}p succeeded.")
            except Exception as e2:
                if verbose:
                    print(f"   ❌ Fallback for {height}p also failed. Error: {e2}")
                continue

    if output_metadata:
        if verbose:
            print(f"✅ Successfully created renditions: {', '.join(f'{k}p' for k in sorted(output_metadata.keys(), key=int, reverse=True))}")
    else:
        print("❌ Error creating all renditions.")
    return output_metadata

# -------------------------------
# Matching & Sanity Logic
# -------------------------------
def check_overlap_sanity(offset_seconds: float, track1_duration: float, track2_duration: float, verbose: bool = False) -> Dict:
    track1_start, track1_end = -offset_seconds, -offset_seconds + track1_duration
    track2_start, track2_end = 0.0, track2_duration
    overlap_duration = max(0.0, min(track1_end, track2_end) - max(track1_start, track2_start))
    min_required_overlap = max(0.1, min(track1_duration, track2_duration) * 0.1)
    sufficient_overlap = overlap_duration >= min_required_overlap
    if verbose:
        print(f"    DEBUG: Overlap duration: {overlap_duration:.2f}s (required: {min_required_overlap:.2f}s)")
    return {'has_overlap': overlap_duration > 0.0, 'sufficient_overlap': sufficient_overlap, 'sanity_check_passed': sufficient_overlap}


def progressive_match(analyzer: TrackAnalyser, new_track: str, unified_track: str, confidence_thresholds: List[float], verbose: bool = False):
    levels = [("Standard", 20, confidence_thresholds[0]), ("High", 35, confidence_thresholds[1]), ("Precise", 50, 0.0)]
    if verbose:
        print("🔄 Progressive Quality Matching (Standard → High → Precise)")
    last_result, level_used = None, "Precise"
    for level_name, pps, threshold in levels:
        result = analyzer.calculate_pairwise_offset(new_track, unified_track, peaks_per_second=pps)
        last_result = result
        if result.error:
            if verbose:
                print(f"   ❌ ERROR at {level_name}: {result.error}")
            continue
        if verbose:
            print(f"   📊 {level_name}: {result.confidence:.1f}% confidence, {result.offset_seconds:.3f}s offset")
        if result.confidence >= threshold:
            level_used = level_name
            if verbose:
                print(f"   ✅ ACCEPTED - stopping at {level_name} level")
            break
    return last_result, level_used

# -----------------------
# Core Processing
# -----------------------
def process_single_video(video_path: str, unified_track: str, args: argparse.Namespace, *, processed_dir: str) -> Dict:
    video_name, video_stem = Path(video_path).name, Path(video_path).stem

    try:
        original_duration, _, _, original_height, _ = get_video_info(video_path)
        if not original_duration:
            return {'video_file': video_name, 'status': 'failed', 'error': 'Could not read video info'}

        # Extract ANALYSIS audio as 44.1k mono 16-bit WAV (explicit)
        audio_path = os.path.join(processed_dir, f"{video_stem}_analysis.wav")
        if not extract_audio_from_video(video_path, audio_path, 'wav', args.sample_rate or 44100, None, 1, args.verbose):
            return {'video_file': video_name, 'status': 'failed', 'error': 'Audio extraction failed'}

        track_duration = get_audio_duration(audio_path)
        unified_duration = get_audio_duration(unified_track)

        analyzer = TrackAnalyser(target_sr=args.target_sr)
        if args.progressive:
            result, level_used = progressive_match(analyzer, audio_path, unified_track, [args.standard_threshold, args.fast_threshold], args.verbose)
        else:
            result = analyzer.calculate_pairwise_offset(audio_path, unified_track, peaks_per_second=args.peaks_per_second)
            level_used = f"Single ({args.peaks_per_second} peaks/sec)"

        if result.error:
            if args.cleanup_audio:
                os.remove(audio_path)
            return {'video_file': video_name, 'status': 'failed', 'error': result.error, 'confidence': 0}

        raw_offset, target_offset = result.offset_seconds, math.floor(result.offset_seconds)
        start_trim = raw_offset - target_offset
        target_duration = math.floor(original_duration - start_trim)

        if target_duration < 1:
            if args.cleanup_audio:
                os.remove(audio_path)
            return {'video_file': video_name, 'status': 'failed', 'error': 'Clip too short after alignment', 'confidence': result.confidence}

        sanity_check = check_overlap_sanity(raw_offset, track_duration, unified_duration, args.verbose)
        adjusted_confidence = result.confidence
        if not sanity_check['sanity_check_passed']:
            adjusted_confidence *= 0.5
            if args.verbose:
                print(f"    ⚠️ Sanity check failed! Confidence adjusted: {result.confidence:.1f}% → {adjusted_confidence:.1f}%")

        if track_duration < SHORT_SEC:
            low_conf, accept_conf, min_matches_red, min_matches_green, dom_thresh = 25.0, 47.5, 5, 8, 1.25
        elif track_duration >= LONG_SEC:
            low_conf, accept_conf, min_matches_red, min_matches_green, dom_thresh = 30.0, 60.0, 10, 15, 1.35
        else:
            low_conf, accept_conf, min_matches_red, min_matches_green, dom_thresh = 27.5, 52.5, 7, 12, 1.30

        dominance_ratio = None
        try:
            if hasattr(result, "candidates") and result.candidates:
                counts = sorted((int(c.get("count", 0)) for c in result.candidates), reverse=True)
                if len(counts) >= 2:
                    dominance_ratio = counts[0] / max(1, counts[1])
        except Exception:
            dominance_ratio = None

        total_matches = getattr(result, "total_matches", None)
        try:
            total_matches = int(total_matches) if total_matches is not None else None
        except Exception:
            total_matches = None

        if (not sanity_check['sanity_check_passed']) or (adjusted_confidence < low_conf) or (total_matches is not None and total_matches < min_matches_red):
            decision = "red"
        elif (adjusted_confidence >= accept_conf and (total_matches is None or total_matches >= min_matches_green) and (dominance_ratio is None or dominance_ratio >= dom_thresh)):
            decision = "green"
        else:
            decision = "yellow"

        aligned_renditions, aligned_video_file = None, None

        if args.output_resolutions:
            aligned_stem = os.path.join(processed_dir, f"{video_stem}_aligned_off{target_offset:+d}")
            if original_height:
                resolutions_to_create = {r for r in args.output_resolutions if r <= original_height}
            else:
                resolutions_to_create = set(args.output_resolutions)
            if not resolutions_to_create:
                resolutions_to_create.add(min(args.output_resolutions))
            aligned_renditions = trim_and_create_renditions(video_path, aligned_stem, start_trim, target_duration, resolutions_to_create, args.verbose)
        else:
            aligned_video_file = f"{video_stem}_aligned_off{target_offset:+d}{Path(video_path).suffix}"
            aligned_video_path = os.path.join(processed_dir, aligned_video_file)
            if not trim_video_precise(video_path, aligned_video_path, start_trim, target_duration):
                if args.cleanup_audio:
                    os.remove(audio_path)
                return {'video_file': video_name, 'status': 'failed', 'error': 'Video alignment trim failed', 'confidence': result.confidence}

        if not aligned_renditions and not aligned_video_file:
            if args.cleanup_audio:
                os.remove(audio_path)
            return {'video_file': video_name, 'status': 'failed', 'error': 'Video output creation failed', 'confidence': result.confidence}

        aligned_audio_path = None
        if args.export_audio:
            source_for_audio, aligned_stem_name = "", ""
            if aligned_renditions:
                highest_res = str(max(int(h) for h in aligned_renditions.keys()))
                source_for_audio = os.path.join(processed_dir, aligned_renditions[highest_res]['file'])
                aligned_stem_name = Path(aligned_renditions[highest_res]['file']).stem
            elif aligned_video_file:
                source_for_audio = os.path.join(processed_dir, aligned_video_file)
                aligned_stem_name = Path(aligned_video_file).stem

            if source_for_audio:
                export_name = f"{aligned_stem_name}.{args.format}"
                aligned_audio_path = os.path.join(processed_dir, export_name)
                # Final export uses the user-specified format
                extract_audio_from_video(source_for_audio, aligned_audio_path, args.format, args.sample_rate, args.bitrate, args.channels, args.verbose)

        if args.cleanup_audio:
            os.remove(audio_path)

        durations_block = {
            "original": float(original_duration),
            "final": float(target_duration),
        }

        return {
            'video_file': video_name,
            'status': 'success',
            'decision': decision,
            'confidence': adjusted_confidence,
            'raw_offset_seconds': raw_offset,
            'final_offset_seconds': target_offset,
            'aligned_renditions': aligned_renditions,
            'aligned_video_file': aligned_video_file,
            'aligned_audio_file': os.path.basename(aligned_audio_path) if aligned_audio_path else None,
            'tags': ["UGC"],
            'level_used': level_used,
            'durations': durations_block,
        }
    except Exception as e:
        return {'video_file': video_name, 'status': 'failed', 'error': str(e), 'confidence': 0}


def display_video_batch_results(results: List[Dict]):
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    print("\n📊 Video Batch Results Summary:")
    print(f"   Processed: {len(results)} videos | Successful: {len(successful)} | Failed: {len(failed)}")
    if successful:
        print("\n✅ Successful Matches:")
        print(f"{'Video File':<40} {'Offset':<10} {'Confidence':<12} {'Status'}")
        print("-" * 80)
        for r in successful:
            video_short = r['video_file'][:38] + ".." if len(r['video_file']) > 40 else r['video_file']
            status_display = f"{r['decision'].upper()}"
            print(f"{video_short:<40} {r['final_offset_seconds']:>+7.0f}s {r['confidence']:>8.1f}%   {status_display}")
    if failed:
        print("\n❌ Failed Videos:")
        for r in failed:
            print(f"   {r['video_file']}: {r['error']}")


def process_against_single_unified(video_files: List[str], unified_ref: str, args: argparse.Namespace) -> Dict:
    unified_name = Path(unified_ref).name
    print(f"\n🎯 Processing against unified reference: {unified_name}")
    print("=" * 60)

    is_unified_audio = is_audio_file(unified_ref)
    unified_dirname = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(unified_name).stem)
    processed_dir = os.path.join(args.output_dir, unified_dirname)
    os.makedirs(processed_dir, exist_ok=True)

    unified_renditions, unified_trimmed_duration = None, 0.0
    analysis_audio_name = f"{Path(unified_name).stem}_analysis_audio.wav"
    unified_track_path = os.path.join(processed_dir, analysis_audio_name)

    if not is_unified_audio:
        print(f"\nPreparing unified video reference: {unified_name}...")
        unified_duration, _, _, unified_height, _ = get_video_info(unified_ref)
        if not unified_duration:
            return {'status': 'failed', 'error': f'Could not read unified reference info: {unified_name}'}

        unified_trimmed_duration = math.floor(unified_duration)
        trimmed_unified_path = os.path.join(processed_dir, f"{Path(unified_name).stem}_trimmed.mp4")
        if not trim_video_precise(unified_ref, trimmed_unified_path, 0.0, unified_trimmed_duration):
            return {'status': 'failed', 'error': f'Failed to pre-trim unified reference: {unified_name}'}

        if args.output_resolutions:
            unified_stem = os.path.join(processed_dir, f"{Path(unified_name).stem}_trimmed_unified")
            if unified_height:
                unified_resolutions = {r for r in args.output_resolutions if r <= unified_height}
            else:
                unified_resolutions = set(args.output_resolutions)
            if not unified_resolutions:
                unified_resolutions.add(min(args.output_resolutions))
            unified_renditions = trim_and_create_renditions(trimmed_unified_path, unified_stem, 0.0, unified_trimmed_duration, unified_resolutions, args.verbose)
            if not unified_renditions:
                return {'status': 'failed', 'error': f'Failed to create renditions for unified reference: {unified_name}'}

        # Extract analysis audio for unified: explicit 44.1k mono WAV
        if not extract_audio_from_video(trimmed_unified_path, unified_track_path, 'wav', args.sample_rate or 44100, None, 1, args.verbose):
            return {'status': 'failed', 'error': f'Failed to extract audio from trimmed unified reference: {unified_name}'}
        if args.cleanup_trimmed:
            os.remove(trimmed_unified_path)

    else:  # Audio only unified
        unified_trimmed_duration = get_audio_duration(unified_ref)
        unified_track_path = unified_ref
        analysis_audio_name = unified_name

    results: List[Dict] = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {Path(video_file).name}")
        r = process_single_video(video_file, unified_track_path, args, processed_dir=processed_dir)
        results.append(r)
        if r['status'] == 'failed':
            print(f"   ❌ {r['error']}")

    display_video_batch_results(results)
    compact_path = write_compact_offsets_report(
        processed_dir, unified_name, results, unified_renditions,
        unified_trimmed_duration, analysis_audio_name, args, is_unified_audio
    )
    print(f"\n📄 Compact report generated: {compact_path}")

    batch_result = {
        'unified_reference': {
            "reference_name": unified_name, "is_audio_only": is_unified_audio,
            "trimmed_rel": unified_renditions, "duration": unified_trimmed_duration,
            "analysis_audio_rel": analysis_audio_name, "metadata": {}
        },
        'results': results,
        'summary': {
            'total': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'green': len([r for r in results if r.get('decision') == 'green']),
            'yellow': len([r for r in results if r.get('decision') == 'yellow']),
            'red': len([r for r in results if r.get('decision') == 'red']),
        }
    }
    detailed_path = export_batch_results(batch_result, args.output_dir, args)
    print(f"📄 Detailed report generated: {detailed_path}")

    if args.cleanup_unified and unified_track_path != unified_ref:
        os.remove(unified_track_path)
    return {'status': 'success'}

# ----------------
# CLI Entrypoint
# ----------------
def main():
    parser = argparse.ArgumentParser(description="Match videos and create ABR renditions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-dir', '-d', type=str, required=True, help='Directory of videos to process.')
    parser.add_argument('--pattern', default='*.*', help='File pattern for video search.')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search subdirectories recursively.')
    unified_group = parser.add_mutually_exclusive_group(required=True)
    unified_group.add_argument('--video-unified', help='Single video file as the master reference.')
    unified_group.add_argument('--unified-dir', help='Directory of master reference files.')
    unified_group.add_argument('--unified', help='Single pre-existing unified audio track.')
    parser.add_argument('--unified-pattern', default='*.*', help='Pattern for unified reference search.')
    parser.add_argument('--unified-recursive', action='store_true', help='Search unified directory recursively.')
    parser.add_argument('--output-dir', '-o', default='processed_videos', help='Root output directory.')
    parser.add_argument('--export-prefix', type=str, help='Prefix for exported result files.')
    parser.add_argument('--output-resolutions', type=lambda s: {int(r) for r in s.split(',') if r.strip()}, default=None, help='Enable ABR mode. Comma-separated list of heights (e.g., 1080,720). If not set, original single-file trim is used.')
    parser.add_argument('--reject-positive-offsets', action='store_true', help='Exclude results with positive raw offsets from the compact JSON report.')
    parser.add_argument('--export-audio', action='store_true', help='Export a final aligned audio file in the specified --format.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output (overrides verbose).')
    parser.add_argument('--format', '-f', choices=['wav', 'mp3', 'flac', 'aac'], default='wav', help='Audio format for final export.')
    parser.add_argument('--sample-rate', '-s', type=int, help='Target sample rate for analysis audio.')
    parser.add_argument('--bitrate', '-b', type=str, help='Audio bitrate for compressed formats.')
    parser.add_argument('--channels', '-c', type=int, choices=[1, 2], help='Number of audio channels.')
    parser.add_argument('--progressive', action='store_true', help='Use progressive quality (20→35→50 peaks/sec).')
    parser.add_argument('--peaks-per-second', type=int, default=20, help='Peak density for single match.')
    parser.add_argument('--target-sr', type=int, help='Target sample rate for audio processing in TrackAnalyser.')
    parser.add_argument('--fast-threshold', type=float, default=75.0, help='Confidence threshold for high level progressive match.')
    parser.add_argument('--standard-threshold', type=float, default=60.0, help='Confidence threshold for standard level progressive match.')
    parser.add_argument('--cleanup-audio', action='store_true', help='Delete per-video analysis audio files.')
    parser.add_argument('--cleanup-unified', action='store_true', help='Delete the unified reference analysis audio after a batch.')
    parser.add_argument('--cleanup-trimmed', action='store_true', help='Delete the intermediate trimmed unified video.')
    parser.add_argument('--cleanup-all', action='store_true', help='Enable all cleanup flags.')
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False
    if args.cleanup_all:
        args.cleanup_audio, args.cleanup_unified, args.cleanup_trimmed = True, True, True

    check_ffmpeg()

    try:
        video_files = discover_video_files(args.video_dir, args.pattern, args.recursive)
        print(f"🎥 Found {len(video_files)} video files to process.")
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"❌ Error discovering videos: {e}")
        sys.exit(1)

    unified_refs = []
    if args.video_unified:
        unified_refs.append(args.video_unified)
    elif args.unified:
        unified_refs.append(args.unified)
    elif args.unified_dir:
        try:
            unified_refs.extend(discover_unified_references(args.unified_dir, args.unified_pattern, args.unified_recursive))
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"❌ Error discovering unified references: {e}")
            sys.exit(1)

    if not unified_refs:
        print(f"❌ No unified reference files found.")
        sys.exit(1)

    if args.output_resolutions:
        print(f" ABR Rendition mode enabled. Resolutions to be generated: {sorted(list(args.output_resolutions), reverse=True)}")
    else:
        print(" Single-file trim mode enabled.")

    print(f"🎬 Processing {len(video_files)} videos against {len(unified_refs)} unified references.")

    start_time = time.time()
    for unified_ref in unified_refs:
        process_against_single_unified(video_files, unified_ref, args)

    print(f"\n\n⏱️ Total processing time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
