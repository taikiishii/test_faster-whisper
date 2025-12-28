import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import io
import wave
import time


def apply_preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def normalize_audio(audio_np: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """音声を目標のdBレベルに正規化"""
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-7:
        return audio_np
    target_rms = 10 ** (target_db / 20.0)
    return audio_np * (target_rms / rms)


def calculate_frame_rms(data: bytes, apply_preemph: bool = False, preemph_coeff: float = 0.97) -> float:
    """単一フレームのRMSを計算"""
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = audio_np - float(np.mean(audio_np))
    if apply_preemph:
        audio_np = apply_preemphasis(audio_np, coeff=preemph_coeff)
    return float(np.sqrt(np.mean(audio_np ** 2)))


def measure_rms(stream, sample_rate: int, chunk_size: int, seconds: float = 2.0, apply_preemph: bool = False, preemph_coeff: float = 0.97) -> float:
    frames = []
    for _ in range(int(sample_rate / chunk_size * seconds)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
    audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = audio_np - float(np.mean(audio_np))
    if apply_preemph:
        audio_np = apply_preemphasis(audio_np, coeff=preemph_coeff)
    return float(np.sqrt(np.mean(audio_np ** 2)))

def main():
    # --- 設定 ---
    model_size = "small"  # tiny, base, small, medium, large （大きいほど精度向上、計算量増加）
    device = "cpu"        # GPU(NVIDIA)があるなら "cuda"
    compute_type = "int8" # CPU使用時。GPU時は "float16" や "float32" を推奨
    
    # === 精度向上パラメータ ===
    beam_size = 5         # 5 → 10 で精度向上（計算時間も増加）
    temperature = 0.0     # 0.0 = 最も確実な認識、高いほど多様な結果
    enable_audio_norm = True  # 音声レベルを正規化してSNRを改善
    normalize_target_db = -20.0  # 正規化の目標dB
    
    sample_rate = 16000   # Whisper推奨のサンプリングレート
    chunk_size = 1024     # バッファサイズ
    test_record_seconds = 3  # 開始時のテスト録音秒数
    enable_preemph = True   # 事前強調でSNRを少し改善
    preemph_coeff = 0.97
    input_device_index = None  # 必要なら入力デバイス番号を指定
    gate_multiplier = 1.5   # ノイズゲート閾値の乗数（低いほど感度が高い）
    gate_enabled = True     # ノイズゲートの有効/無効
    
    # VAD（音声活動検出）パラメータ
    silence_threshold_multiplier = 1.2  # 無音判定の基準（基準RMS × この値より小さい = 無音）
    silence_duration_sec = 0.8  # この秒数連続で無音なら認識実行
    max_record_duration = 30.0  # 最大録音時間（秒）
    
    # モデルのロード
    print(f"モデル '{model_size}' をロード中...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # PyAudioの初期化
    audio = pyaudio.PyAudio()
    
    # --- デバイス情報表示 ---
    print("\n=== 利用可能な音声入力デバイス ===")
    default_device = audio.get_default_input_device_info()
    print(f"デフォルトデバイス: {default_device['name']} (index: {default_device['index']})")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            print(f"  [{i}] {dev_info['name']} (channels: {dev_info['maxInputChannels']})")
    print("=" * 40)
    
    input_kwargs = dict(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    if input_device_index is not None:
        input_kwargs['input_device_index'] = input_device_index
    stream = audio.open(**input_kwargs)

    # --- 環境ノイズをキャリブレーション ---
    print("\n環境ノイズ測定中…（2秒、静かにしてください）")
    baseline_rms = measure_rms(stream, sample_rate, chunk_size, seconds=2.0, apply_preemph=enable_preemph, preemph_coeff=preemph_coeff)
    noise_gate_rms = max(0.005, baseline_rms * gate_multiplier)
    print(f"基準RMS={baseline_rms:.4f} → ゲートRMS={noise_gate_rms:.4f}")
    if not gate_enabled:
        print("(ノイズゲートは無効です)")

    # --- テスト録音フェーズ ---
    print(f"\nテスト録音を開始します。{test_record_seconds}秒マイクに向かって話してください...")
    test_frames = []
    test_start = time.time()
    for _ in range(int(sample_rate / chunk_size * test_record_seconds)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        test_frames.append(data)
    test_elapsed = time.time() - test_start
    
    test_audio_data = b"".join(test_frames)
    test_audio_np = np.frombuffer(test_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    test_audio_np = test_audio_np - float(np.mean(test_audio_np))
    test_rms = float(np.sqrt(np.mean(test_audio_np ** 2)))
    test_level = float(np.abs(test_audio_np).max())
    
    print(f"\n✓ テスト録音完了 ({test_elapsed:.2f}秒) | max={test_level:.3f}, rms={test_rms:.3f}")
    
    # テスト再生
    print("▶ テスト録音を再生します...")
    try:
        play_stream = audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=sample_rate,
                                 output=True)
        play_stream.write(test_audio_data)
        play_stream.stop_stream()
        play_stream.close()
        print("✓ テスト再生完了")
    except Exception as e:
        print(f"✗ テスト再生に失敗しました: {e}")
    
    # ユーザー確認
    print("\n問題がなければ Enterキーを押してください...")
    input()
    
    print("\n>>> 音声認識を開始します！ (Ctrl+Cで終了)")
    print("※ 無音時は待機、音声検出後に無音になったら認識を実行します\n")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            frames = []
            consecutive_silence_frames = 0
            silence_duration_frames = int(silence_duration_sec * sample_rate / chunk_size)
            max_frames = int(max_record_duration * sample_rate / chunk_size)
            is_recording = False
            record_start_time = None
            
            print(f"[{loop_count}] 音声を待機中...", end="", flush=True)
            
            # 音声検出→無音で自動終了のループ
            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # フレームのRMSを計算
                frame_rms = calculate_frame_rms(data, apply_preemph=enable_preemph, preemph_coeff=preemph_coeff)
                
                # 音声検出の判定
                if not is_recording:
                    # 待機中→音声検出で開始
                    if frame_rms >= noise_gate_rms:
                        is_recording = True
                        record_start_time = time.time()
                        consecutive_silence_frames = 0
                        print(f"\r[{loop_count}] 録音中... ", end="", flush=True)
                else:
                    # 録音中
                    if frame_rms < noise_gate_rms * silence_threshold_multiplier:
                        # 無音フレーム
                        consecutive_silence_frames += 1
                    else:
                        # 音声フレーム
                        consecutive_silence_frames = 0
                    
                    # 進捗表示
                    elapsed = time.time() - record_start_time
                    print(f"\r[{loop_count}] 録音中... ({elapsed:.1f}秒) ", end="", flush=True)
                
                # 終了条件チェック
                if is_recording:
                    # 条件1: 最大時間に到達
                    if len(frames) >= max_frames:
                        print(f"\n  最大録音時間に到達しました")
                        break
                    # 条件2: 十分な無音期間
                    if consecutive_silence_frames >= silence_duration_frames:
                        print(f"\n  無音を検出しました")
                        break
            
            # 記録した音声を処理
            record_time = time.time() - record_start_time if record_start_time else 0
            
            # メモリ上の音声データをWhisperが読める形式に変換
            audio_data = b"".join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            # DCオフセット除去
            audio_np = audio_np - float(np.mean(audio_np))
            # 事前強調（任意）
            if enable_preemph:
                audio_np = apply_preemphasis(audio_np, coeff=preemph_coeff)
            
            # 音声レベルのチェック
            audio_level = np.abs(audio_np).max()
            audio_rms = np.sqrt(np.mean(audio_np**2))
            
            # === 精度向上: 音声正規化 ===
            if enable_audio_norm:
                audio_np = normalize_audio(audio_np, target_db=normalize_target_db)
            
            clip_ratio = float(np.mean(np.abs(audio_np) >= 0.98))
            print(f"✓ 録音完了 ({record_time:.2f}秒) | max={audio_level:.3f}, rms={audio_rms:.3f}, clip={clip_ratio*100:.1f}%")
            
            # 音声が小さすぎる場合はスキップ
            if audio_rms < noise_gate_rms:
                print(f"  (音声が小さすぎます)")
                continue

            # クリッピングが多い場合は警告
            if clip_ratio > 0.01:
                print("  ⚠️ 入力が大きすぎてクリップしています。マイクレベルを下げてください。")
            
            # 文字起こし実行
            print("  認識中...", end="", flush=True)
            transcribe_start = time.time()
            segments, info = model.transcribe(
                audio_np, 
                language="ja", 
                beam_size=beam_size,      # 精度向上パラメータ
                temperature=temperature,   # 精度向上パラメータ
                vad_filter=True,  # VADフィルター有効化（無音部分をスキップ）
                vad_parameters=dict(min_silence_duration_ms=500)  # 500ms以上の無音で区切る
            )
            
            segment_list = list(segments)
            transcribe_time = time.time() - transcribe_start
            
            print(f" ({transcribe_time:.2f}秒)")
            
            if segment_list:
                for segment in segment_list:
                    print(f"  ✓ {segment.text}")
            else:
                print("  (認識結果なし)")

    except KeyboardInterrupt:
        print("\n終了します...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()