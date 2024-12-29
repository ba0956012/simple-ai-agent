
# simple-ai-agent

## 簡介

simple-ai-agent 功能：

- 聲音監聽與處理
- 語音轉文字（使用speech_recognition的google）
- 關鍵字檢測與操作 (sBERT 或是 Levenshtein)
- 現場人數檢測 ([yoloV5n](https://github.com/yakhyo/yolov5-onnx-inference/blob/main/models/yolov5.py))

---

## 環境需求

- **系統**: macOS (Apple Silicon, M1)
- **Python**: 3.8
- **環境管理工具**: miniforge3

---

## 安裝步驟

### 1. 安裝 Conda

如果尚未安裝 Conda，請先安裝 miniforge3.

### 2. 創建 Conda 環境


```bash
conda create -n audio_ai_env python=3.8
conda activate audio_ai_env
```

### 3. 安裝requirements.txt


```bash
pip install -r requirements.txt
```

### 4. 安裝 `cffi`

在 Mac M1 上，`cffi` 需要通過 `conda` 安裝：

```bash
conda install cffi
```

---

## 使用方式

1. 確保 `.env` 文件已設置（把.env.example改為.env，可自行調整參數）。
2. 運行主程序，第一次執行需要等待下載模型(請耐心等待)：
   ```bash
   python main.py
   ```
