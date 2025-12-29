import time
import argparse
import pyaudio
import numpy as np
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

# Predefined command vocabulary and their variants
COMMANDS = {
    "FORWARD": ["まえ","マエ","前","すすめ","ススメ","進め","まい"],
    "STOP": ["とまれ","トマレ","止まれ","ストップ","stop"],
    "LEFT": ["ひだり","ヒダリ","左"],
    "RIGHT": ["みぎ","ミギ","右",],
    "BACK": ["うしろ", "ウシロ","後ろ", "バック", "back"],
}

def apply_preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def calculate_rms(data: bytes, apply_preemph: bool = False, preemph_coeff: float = 0.97) -> float:
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = audio_np - float(np.mean(audio_np))
    if apply_preemph:
        audio_np = apply_preemphasis(audio_np, coeff=preemph_coeff)
    return float(np.sqrt(np.mean(audio_np ** 2)))


def match_command(text: str, threshold: float = 0.55):
    cleaned = text.lower().replace(" ", "")
    best_cmd, best_score = None, 0.0
    for cmd, variants in COMMANDS.items():
        for phrase in variants:
            cand = phrase.lower().replace(" ", "")
            # exact substring boost
            if cand in cleaned:
                return cmd, 1.0
            score = SequenceMatcher(None, cleaned, cand).ratio()
            if score > best_score:
                best_score = score
                best_cmd = cmd
    if best_score >= threshold:
        return best_cmd, best_score
    return None, best_score


def main():
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description="Voice command recognition using Whisper")
    parser.add_argument("-m", "--model", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model size (default: small)")
    parser.add_argument("-p", "--playback", action="store_true",
                        help="Play recorded audio before transcription for verification")
    parser.add_argument("-t", "--threshold", type=float, default=0.55,
                        help="Command match threshold (0-1, default: 0.55)")
    args = parser.parse_args()

    # --- Settings ---
    model_size = args.model      # Use selected model from command line
    device = "cpu"             # use "cuda" if you have NVIDIA GPU
    compute_type = "int8"      # cpu: int8; gpu: float16/float32 recommended
    beam_size = 5
    temperature = 0.0

    sample_rate = 16000
    chunk_size = 1024
    max_record_sec = 6.0
    silence_duration_sec = 0.7
    gate_multiplier = 1.5
    enable_preemph = True
    preemph_coeff = 0.97

    print(f"Loading Whisper model '{model_size}'...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    # Noise calibration
    print("\nCalibrating noise floor... please stay quiet for 2 seconds.")
    calib_frames = []
    for _ in range(int(sample_rate / chunk_size * 2)):
        calib_frames.append(stream.read(chunk_size, exception_on_overflow=False))
    calib_audio = b"".join(calib_frames)
    base_rms = calculate_rms(calib_audio, apply_preemph=enable_preemph, preemph_coeff=preemph_coeff)
    noise_gate_rms = max(0.005, base_rms * gate_multiplier)
    silence_frames_needed = int(silence_duration_sec * sample_rate / chunk_size)
    max_frames = int(max_record_sec * sample_rate / chunk_size)
    print(f"Baseline RMS={base_rms:.4f}, gate={noise_gate_rms:.4f}")

    playback_mode = "(playback enabled)" if args.playback else ""
    print(f"\nReady. Say a command (e.g., '進め', '止まれ', '右', '左', 'バック'). Ctrl+C to exit. {playback_mode}")

    try:
        loop = 0
        while True:
            loop += 1
            frames = []
            is_recording = False
            silence_count = 0
            start_ts = None

            print(f"[{loop}] Waiting for speech...", end="", flush=True)

            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                frame_rms = calculate_rms(data, apply_preemph=enable_preemph, preemph_coeff=preemph_coeff)

                if not is_recording:
                    if frame_rms >= noise_gate_rms:
                        is_recording = True
                        start_ts = time.time()
                        silence_count = 0
                        print(f"\r[{loop}] Recording... ", end="", flush=True)
                else:
                    if frame_rms < noise_gate_rms:
                        silence_count += 1
                    else:
                        silence_count = 0

                    elapsed = time.time() - start_ts
                    print(f"\r[{loop}] Recording... ({elapsed:.1f}s) ", end="", flush=True)

                    # stop conditions
                    if silence_count >= silence_frames_needed:
                        print("\n  Detected silence. Stopping.")
                        break
                    if len(frames) >= max_frames:
                        print("\n  Reached max duration. Stopping.")
                        break

            if not is_recording:
                # didn't detect speech; skip
                frames.clear()
                continue

            audio_data = b"".join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_np = audio_np - float(np.mean(audio_np))
            if enable_preemph:
                audio_np = apply_preemphasis(audio_np, coeff=preemph_coeff)

            # Playback recorded audio if enabled
            if args.playback:
                print("  ▶ Playing back recorded audio...")
                try:
                    play_stream = audio.open(format=pyaudio.paInt16,
                                             channels=1,
                                             rate=sample_rate,
                                             output=True)
                    play_stream.write(audio_data)
                    play_stream.stop_stream()
                    play_stream.close()
                    print("  ✓ Playback complete.")
                except Exception as e:
                    print(f"  ✗ Playback failed: {e}")

            # Transcribe
            print("  Transcribing...", end="", flush=True)
            segments, _ = model.transcribe(
                audio_np,
                language="ja",
                beam_size=beam_size,
                temperature=temperature,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(seg.text for seg in segments).strip()
            print(f" done.\n  Raw text: {text}")

            cmd, score = match_command(text, threshold=args.threshold)
            if cmd:
                print(f"  Matched command: {cmd} (score={score:.2f})")
            else:
                print(f"  No command matched (best score={score:.2f}).")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
