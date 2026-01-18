#!/bin/bash
# CUDA対応のCTranslate2を使用するためのライブラリパス設定
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH

# Pythonスクリプトを実行
/home/taiki/Documents/test_faster-whisper/.venv/bin/python /home/taiki/Documents/test_faster-whisper/faster-whisper.py "$@"
