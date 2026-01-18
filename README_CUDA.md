# CUDA対応faster-whisperのセットアップ完了

## 完了した作業

1. **CTranslate2のCUDA対応ビルド**
   - ソースコード: `/tmp/CTranslate2`
   - ビルドディレクトリ: `/tmp/CTranslate2/build`
   - インストール先: `/usr/local/ctranslate2`
   - CUDA アーキテクチャ: 87 (Jetson Orin Nano用)

2. **設定**
   - device: `cuda`
   - compute_type: `float16`
   - OpenBLAS使用（MKLの代わり）

## 実行方法

### 方法1: スクリプトを使用（推奨）
```bash
./run.sh
```

### 方法2: 環境変数を設定して実行
```bash
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
/home/taiki/Documents/test_faster-whisper/.venv/bin/python faster-whisper.py
```

## 永続的にライブラリパスを設定する方法

`~/.bashrc` に既に以下のような CUDA のパス設定がある場合:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

**この行を書き換えずに**、以下の行を追加してください:
```bash
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
```

または、既存の行を以下のように統合することもできます:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
```

その後、シェルを再起動するか以下を実行:
```bash
source ~/.bashrc
```

## インストール先の永続化（完了済み）

✅ すでに `/usr/local/ctranslate2` に移動済みです。

以下のファイルは既に正しいパスに更新されています:

1. **run.sh** のLD_LIBRARY_PATHを更新:
```bash
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
```

2. **~/.bashrc** のCTranslate2のパスを更新:
- 追加した行を以下のように変更:
```bash
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
```
- または、CUDA パスと統合している場合:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH
```

## ビルドコマンドの記録

```bash
# 必要なパッケージのインストール
sudo apt-get install -y build-essential git portaudio19-dev

# ソースのクローン
cd /tmp
git clone --recursive https://github.com/OpenNMT/CTranslate2.git

# CMake設定
cd CTranslate2
mkdir build && cd build
cmake -DWITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DWITH_MKL=OFF \
      -DWITH_OPENBLAS=ON \
      -DCMAKE_INSTALL_PREFIX=/tmp/ctranslate2-install \
      ..

# ビルド
make -j$(nproc)

# インストール
sudo make install

# Pythonバインディングのインストール
cd ../python
export CT2_INSTALL_PREFIX=/usr/local/ctranslate2
export CPLUS_INCLUDE_PATH=/usr/local/ctranslate2/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/ctranslate2/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/ctranslate2/lib:$LD_LIBRARY_PATH

pip install pybind11 wheel setuptools --upgrade
pip install . --no-build-isolation
```

## 注意事項

- CUDAバージョン: 12.6
- Jetson Orin Nano用にアーキテクチャ87でビルド
- マイクの感度が低い場合は、`gate_multiplier`を調整してください
