<div align="center">

# 🎤 Development of Low-Resource ASR System for Children's Speech

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper--small-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![WER](https://img.shields.io/badge/WER-22.43%25-brightgreen?style=for-the-badge)](.)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)](.)

<br/>

**A B.Tech Major Project**
University of Petroleum & Energy Studies (UPES), Dehradun
School of Computer Science — B.Tech CSE CCVT

*Submitted in partial fulfillment of the requirements for the degree of*
*Bachelor of Technology in Computer Science & Engineering*

<br/>

| 👤 Team Member    | 🆔 Roll No.    |
|------------------|---------------|
| Omi Kumari       | R2142220340   |
| Shivam Singh     | R2142220940   |

**Guided by:** Kshitij Kumre &nbsp;|&nbsp; **Session:** 2025–2026

</div>

---

## 📌 Overview

Automatic Speech Recognition (ASR) systems achieve strong performance on adult speech but degrade significantly when applied to children's speech. The core challenges stem from children's **higher pitch**, **greater acoustic variability**, **inconsistent pronunciation**, and **non-uniform speaking rates** — characteristics that standard ASR models are not designed to handle.

This project bridges that gap by developing a **data-efficient, low-resource ASR system** specifically tailored for **Hindi and English children's speech**. The pipeline fine-tunes **OpenAI's Whisper-small** transformer model using **SpecAugment-based data augmentation** and **transfer learning**, achieving a significant reduction in Word Error Rate (WER) — from ~40% down to **22.43%**.

> 💡 *The system is designed for real-world deployment in educational tools, child-friendly voice interfaces, healthcare, and accessibility applications.*

---

## 🚀 Key Results

| Metric                | Before Augmentation | After Augmentation          |
|-----------------------|---------------------|-----------------------------|
| Word Error Rate (WER) | ~40%                | **22.43%** ✅                |
| Model                 | Whisper-small       | Whisper-small (fine-tuned)  |
| Languages Supported   | Hindi + English     | Hindi + English             |
| Augmentation Applied  | None                | SpecAugment + Pitch/Prosody |

- 📉 **~44% relative reduction** in Word Error Rate
- 🎯 Better robustness to high pitch and expressive child speech
- 🔊 Improved temporal generalization via time-stretching augmentation
- 🧠 End-to-end model — no external language model required

---

## 🛠️ Tech Stack

| Category              | Tools & Frameworks                                     |
|-----------------------|--------------------------------------------------------|
| **Core Language**     | Python 3.10+                                           |
| **Deep Learning**     | PyTorch (CUDA-enabled), Torchaudio                     |
| **ASR Model**         | OpenAI Whisper-small (244M parameters, multilingual)   |
| **Model Hub**         | Hugging Face Transformers, Datasets, Accelerate        |
| **Audio Processing**  | Librosa, Soundfile                                     |
| **Augmentation**      | SpecAugment (Time & Frequency Masking via Torchaudio)  |
| **Training Env**      | Google Colab (NVIDIA T4 / P100 / A100 GPU)             |
| **Frontend**          | React.js, HTML, CSS, JavaScript                        |
| **Backend**           | Flask (Python)                                         |
| **Optimizer**         | AdamW with low learning rate                           |
| **Loss Function**     | Sequence-to-Sequence Cross-Entropy (CTC)               |
| **Evaluation Metric** | Word Error Rate (WER) via `jiwer`                      |

---

## ⚙️ Methodology

### 1. Data Collection & Preprocessing
- Gathered Hindi child speech recordings and manually cleaned noisy or mislabeled samples.
- Resampled all audio to **16 kHz** and applied amplitude normalization.
- Standardized transcripts: lowercase conversion, punctuation removal, spelling normalization.
- Extracted **log-Mel spectrogram** features using Whisper's built-in feature extractor.

### 2. Data Augmentation — SpecAugment

SpecAugment operates directly on the **Mel-spectrogram**, making it computationally efficient and highly effective for low-resource settings — avoiding the need to record new samples.

| Technique                | Description                                         | Benefit for Child Speech                         |
|--------------------------|-----------------------------------------------------|--------------------------------------------------|
| **Time Masking**         | Randomly masks a block of time frames (→ zero)      | Simulates pauses, hesitations, breathing sounds  |
| **Frequency Masking**    | Randomly masks a range of frequency bins            | Robustness to high-pitch and pitch variability   |
| **Pitch Augmentation**   | Shifts the fundamental frequency (F0) of audio      | Better handling of high-frequency utterances     |
| **Time-Stretching**      | Speeds up or slows audio without changing pitch     | Improves temporal generalization                 |
| **Prosody Modification** | Adjusts speaking rhythm and stress patterns         | Robustness to expressive and inconsistent speech |

### 3. Model Selection — Whisper-small

Whisper's **encoder–decoder transformer architecture** was chosen for its:
- Strong **zero-shot multilingual** performance out of the box
- Robustness to noisy and low-resource audio environments
- No requirement for an external language model (end-to-end design)

| Size      | Layers | Width | Heads | Parameters | Multilingual |
|-----------|--------|-------|-------|------------|:------------:|
| tiny      | 4      | 384   | 6     | 39 M       | ✓            |
| base      | 6      | 512   | 8     | 74 M       | ✓            |
| **small** | **12** | **768** | **12** | **244 M** | **✓**      |
| medium    | 24     | 1024  | 16    | 769 M      | ✓            |
| large-v3  | 32     | 1280  | 20    | 1550 M     | ✓            |

> ✅ **Whisper-small** was selected as the optimal balance between parameter efficiency and multilingual accuracy for a low-resource setting.

### 4. Fine-Tuning Pipeline

```
Pretrained Whisper-small Checkpoint
           ↓
Log-Mel Feature Extraction + Text Tokenization (Whisper Processor)
           ↓
SpecAugment Applied (Training Split Only)
           ↓
Frozen Encoder Layers → Gradual Unfreezing for Deeper Adaptation
           ↓
AdamW Optimizer | Low Learning Rate | Seq2Seq Cross-Entropy Loss
           ↓
Batch Training over Multiple Steps / Epochs (Google Colab GPU)
           ↓
WER Evaluation on Validation Set
           ↓
Fine-Tuned Whisper Model → WER: 22.43%
```

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────┐
│             🎙️  Input: Child's Voice          │
└───────────────────────┬──────────────────────┘
                        │
           ┌────────────▼────────────┐
           │    Audio Preprocessing  │
           │  (16 kHz, Normalized)   │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  Log-Mel Spectrogram    │
           │   Feature Extraction    │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │      SpecAugment        │
           │ (Time + Freq Masking)   │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  Whisper-small Encoder  │
           │   (Transformer Stack)   │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │    Whisper Decoder      │
           │  (Attention + CTC)      │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │   📝 Transcribed Text   │
           └─────────────────────────┘
```

---

## 📂 Project Structure

```
📦 low-resource-asr-children/
│
├── 📁 data/
│   ├── raw/                    # Original child speech recordings
│   ├── processed/              # Cleaned & resampled audio (16 kHz)
│   └── augmented/              # SpecAugment-processed training data
│
├── 📁 models/
│   ├── whisper-small-base/     # Pretrained Whisper checkpoint
│   └── whisper-finetuned/      # Fine-tuned model weights
│
├── 📁 src/
│   ├── preprocess.py           # Audio preprocessing pipeline
│   ├── augment.py              # SpecAugment & pitch augmentation
│   ├── train.py                # Fine-tuning training loop
│   └── evaluate.py             # WER computation & evaluation
│
├── 📁 frontend/                # React.js web interface
│   ├── public/
│   └── src/
│       └── App.jsx
│
├── 📁 utils/
│   ├── audio_utils.py          # Audio I/O helper functions
│   └── metrics.py              # WER metric computation
│
├── app.py                      # Flask backend API
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ▶️ How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Colab account (for GPU training) or a local NVIDIA GPU

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/low-resource-asr-children.git
cd low-resource-asr-children
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess & Augment Data
```bash
python src/preprocess.py --input data/raw --output data/processed
python src/augment.py --input data/processed --output data/augmented
```

### 4. Fine-Tune the Model
```bash
python src/train.py \
  --model openai/whisper-small \
  --data data/augmented \
  --output models/whisper-finetuned \
  --epochs 10 \
  --batch_size 8
```

### 5. Evaluate the Model
```bash
python src/evaluate.py --model models/whisper-finetuned --data data/processed
```

### 6. Run the Backend (Flask)
```bash
python app.py
```

### 7. Run the Frontend (React)
```bash
cd frontend
npm install
npm start
```
> 🌐 The app will be available at `http://localhost:3000`

---

## 📋 Requirements

### Hardware (Training)

| Environment       | Specification                                    |
|-------------------|--------------------------------------------------|
| **Cloud (Colab)** | NVIDIA T4 / P100 / A100 — 12–15 GB VRAM         |
| **Local (min)**   | Any PC with internet access (for Colab sessions) |
| **Local (GPU)**   | NVIDIA GPU with 8+ GB VRAM recommended           |

### Python Dependencies (`requirements.txt`)
```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.24.0
librosa>=0.10.0
soundfile>=0.12.1
evaluate>=0.4.0
jiwer>=3.0.0
flask>=3.0.0
flask-cors>=4.0.0
```

---

## 📚 Literature Review Highlights

| Study | Augmentation Technique | Model Architecture | Result |
|-------|------------------------|-------------------|--------|
| Shahnawazuddin (2024) | Out-of-domain + In-domain data | RNN with CTC | CER 8.3% — 65% relative improvement |
| Yeung et al. (2021) | F0 perturbation + MFCCs | BLSTM + 4-gram LM | 19.3% WER |
| Kadyan et al. (2021) | VTLN + Prosody/Duration modification | DNN-HMM Hybrid | 32.1% improvement |
| Shivakumar & Narayanan (2022) | Speed perturbation + SpecAugment | ResNet + Transformer | 2.4% WER |
| **Ours (2025–26)** | **SpecAugment + Pitch/Prosody** | **Whisper-small (fine-tuned)** | **22.43% WER ✅** |

---

## 🔍 SWOT Analysis

|   | Strengths 💪 | Weaknesses ⚠️ |
|---|-------------|--------------|
|   | Data-efficient augmentation via SpecAugment | Limited Hindi child speech dataset size |
|   | Transfer learning from pretrained Whisper | Colab GPU session limit (12 hours) |
|   | End-to-end architecture, no external LM needed | Mobile inference speed not yet optimized |
|   | Multilingual support (Hindi + English) | High variability in child speech hard to fully capture |

|   | Opportunities 🌟 | Threats 🚨 |
|---|-----------------|------------|
|   | Expand to regional Indian languages (Punjabi, Tamil, etc.) | Noisy real-world audio environments |
|   | Integrate LLMs for contextual understanding & code-switching | Ethical concerns in collecting child voice data |
|   | Deploy as a scalable mobile/web application | Risk of overfitting on small datasets |
|   | Apply in speech therapy and healthcare monitoring | Data privacy regulations around minors |

---

## 💡 Applications

| Domain | Use Case |
|--------|----------|
| 🏫 **Education** | Voice-controlled learning platforms and tutoring assistants for children |
| ♿ **Accessibility** | Assistive speech interfaces for children with communication disabilities |
| 🏥 **Healthcare** | Speech therapy evaluation, progress monitoring, and clinical tools |
| 🌐 **Multilingual** | Extend to regional languages — Punjabi, Tamil, Bengali, and more |
| 📱 **Child-Friendly Apps** | Interactive storytelling, quizzes, and voice-enabled games for kids |

---

## 📈 Future Work

- [ ] Expand dataset to cover diverse regional Indian languages
- [ ] Integrate **Large Language Models (LLMs)** for contextual understanding and Hindi-English code-switching
- [ ] Optimize model inference for **real-time, edge/mobile deployment**
- [ ] Explore **child-to-child and teenager-to-child voice conversion** as additional augmentation
- [ ] Deploy as a production-grade, scalable **web and mobile application**
- [ ] Evaluate on standardized benchmarks — **MyST Corpus**, **CMU Kids Dataset**
- [ ] Investigate **Wav2Vec2** and **Conformer** architectures as alternative backbones

---

## 📖 References

1. Shahnawazuddin, S. (2024). Developing children's ASR system under low-resource conditions using end-to-end architecture. *Digital Signal Processing*, *146*, 104385.

2. Yeung, G., Fan, R., & Alwan, A. (2021). Fundamental frequency feature normalization and data augmentation for child speech recognition. *ICASSP 2021*, IEEE, pp. 6993–6997.

3. Kadyan, V., Shanawazuddin, S., & Singh, A. (2021). Developing children's speech recognition system for low-resource Punjabi language. *Applied Acoustics*, *178*, 108002.

4. Shivakumar, P. G., & Narayanan, S. (2022). End-to-end neural systems for automatic children speech recognition: An empirical study. *Computer Speech & Language*, *72*, 101289.

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is submitted as an academic major project at **UPES Dehradun** and is intended for educational and research purposes only.

---

<div align="center">

## 👨‍💻 Authors

| Name | Roll No. | GitHub |
|------|----------|--------|
| **Omi Kumari** | R2142220340 | [@omi](https://github.com/) |
| **Shivam Singh** | R2142220940 | [@shivam](https://github.com/) |

*B.Tech CSE CCVT — School of Computer Science, UPES Dehradun*
*Guided by: **Kshitij Kumre***

<br/>

*Made with ❤️ at UPES Dehradun, 2025–2026*

⭐ *If this project helped you, consider giving it a star!* ⭐

</div>