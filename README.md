<div align="center">

# 🎤 Development of Low-Resource ASR System for Children's Speech

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper--small-412991?style=flat&logo=openai&logoColor=white)
![WER](https://img.shields.io/badge/WER-22.43%25-success?style=flat)
![License](https://img.shields.io/badge/License-Academic-blue?style=flat)

**A B.Tech Major Project** — University of Petroleum & Energy Studies (UPES), Dehradun

*Submitted in partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science & Engineering*

---

| Team Member      | Roll No.      | Guided By        |
|------------------|---------------|------------------|
| Alveera Ahmad    | R2142221531   | **Kshitij Kumre** |
| Omi Kumari       | R2142220340   |                   |
| Sejal Kamboj     | R2142220637   |                   |
| Shivam Singh     | R2142220940   |                   |

</div>

---

## 📌 Overview

Automatic Speech Recognition (ASR) systems perform exceptionally well on adult speech but degrade significantly when applied to children's speech. The challenges stem from children's **higher pitch**, **greater acoustic variability**, **inconsistent pronunciation**, and **non-uniform speaking rates** — characteristics that standard ASR models are not trained to handle well.

This project addresses this gap by developing a **data-efficient, low-resource ASR system** specifically tailored for Hindi and English children's speech. The approach fine-tunes **OpenAI's Whisper-small** transformer model using **SpecAugment-based data augmentation** and **transfer learning**, achieving a significant reduction in Word Error Rate (WER) from ~40% down to **22.43%**.

> *The system is designed for real-world deployment in educational tools, child-friendly voice interfaces, healthcare, and accessibility applications.*

---

## 🚀 Key Results

| Metric                   | Before Augmentation | After Augmentation |
|--------------------------|---------------------|--------------------|
| Word Error Rate (WER)    | ~40%                | **22.43%**         |
| Model                    | Whisper-small       | Whisper-small (fine-tuned) |
| Dataset Language         | Hindi + English     | Hindi + English    |
| Augmentation Applied     | None                | SpecAugment        |

- ✅ ~44% relative reduction in WER
- ✅ Better robustness to pitch variability and expressive child speech
- ✅ Improved temporal generalization via time-stretching augmentation

---

## 🛠️ Tech Stack

| Category              | Tools & Frameworks                                   |
|-----------------------|------------------------------------------------------|
| **Core Language**     | Python 3.10+                                         |
| **Deep Learning**     | PyTorch (CUDA-enabled), Torchaudio                   |
| **ASR Model**         | OpenAI Whisper-small (244M parameters, multilingual) |
| **Model Hub**         | Hugging Face Transformers, Datasets, Accelerate      |
| **Audio Processing**  | Librosa, Soundfile                                   |
| **Augmentation**      | SpecAugment (via Torchaudio)                         |
| **Training Env**      | Google Colab (NVIDIA T4/P100/A100 GPU)               |
| **Frontend**          | React.js, HTML, CSS, JavaScript                      |
| **Backend**           | Flask (Python)                                       |
| **Optimizer**         | AdamW                                                |
| **Loss Function**     | Sequence-to-Sequence Cross-Entropy                   |

---

## ⚙️ Methodology

### 1. Data Collection & Preprocessing
- Gathered Hindi child speech recordings and manually cleaned noisy/mislabeled samples.
- Resampled all audio to **16 kHz** and normalized amplitude.
- Standardized transcripts: lowercase conversion, punctuation removal, spelling normalization.
- Extracted **log-Mel spectrogram** features using Whisper's feature extractor.

### 2. Data Augmentation — SpecAugment
SpecAugment operates directly on the Mel-spectrogram — making it computationally efficient and highly effective for low-resource settings.

| Technique           | Description                                                        | Benefit                                              |
|---------------------|--------------------------------------------------------------------|------------------------------------------------------|
| **Time Masking**    | Randomly masks a block of time frames (set to zero)               | Simulates missing audio; improves context robustness |
| **Frequency Masking** | Randomly masks a range of frequency bins                        | Improves robustness to pitch variation               |
| **Pitch Augmentation** | Shifts fundamental frequency of audio                          | Better handling of high-frequency utterances         |
| **Time-Stretching** | Speeds up or slows down audio without changing pitch              | Improves temporal generalization                     |
| **Prosody Modification** | Adjusts speaking rhythm and stress patterns                | Robustness to expressive and inconsistent speech     |

### 3. Model Selection — Whisper-small
Whisper's **encoder–decoder transformer architecture** was selected for its:
- Strong zero-shot multilingual performance
- Robustness to noisy, low-resource environments
- No requirement for an external language model (end-to-end)

| Size     | Layers | Width | Heads | Parameters | Multilingual |
|----------|--------|-------|-------|------------|--------------|
| tiny     | 4      | 384   | 6     | 39 M       | ✓            |
| base     | 6      | 512   | 8     | 74 M       | ✓            |
| **small**| **12** | **768**| **12**| **244 M** | **✓**        |
| medium   | 24     | 1024  | 16    | 769 M      | ✓            |
| large-v3 | 32     | 1280  | 20    | 1550 M     | ✓            |

### 4. Fine-Tuning Pipeline
```
Pretrained Whisper-small Checkpoint
        ↓
Log-Mel Feature Extraction + Text Tokenization
        ↓
SpecAugment Applied (Training Split Only)
        ↓
Frozen Encoder Layers → Gradual Unfreezing
        ↓
AdamW Optimizer | Low Learning Rate | Cross-Entropy Loss
        ↓
WER Evaluation on Validation Set
        ↓
Fine-Tuned Whisper Model (WER: 22.43%)
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────┐
│              Input: Child's Voice           │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Audio Preprocessing    │
        │  (16kHz, Normalized)    │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Log-Mel Spectrogram    │
        │  Feature Extraction     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    SpecAugment          │
        │ (Time + Freq Masking)   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Whisper-small Encoder  │
        │  (Transformer Layers)   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Whisper Decoder        │
        │  (Attention + CTC)      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Transcribed Text      │
        └─────────────────────────┘
```

---

## 📂 Project Structure

```
📦 low-resource-asr-children/
├── 📁 data/
│   ├── raw/                    # Original child speech recordings
│   ├── processed/              # Cleaned & resampled audio (16kHz)
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
├── 📁 frontend/                # React.js frontend interface
│   ├── public/
│   └── src/
│       └── App.jsx
│
├── 📁 utils/
│   ├── audio_utils.py          # Helper functions for audio I/O
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
- Google Colab account (for GPU training) or local NVIDIA GPU

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

### 5. Run the Backend (Flask)
```bash
python app.py
```

### 6. Run the Frontend (React)
```bash
cd frontend
npm install
npm start
```
> The app will be available at `http://localhost:3000`

---

## 📋 Requirements

### Hardware (Training)
- **Cloud:** Google Colab with NVIDIA T4 / P100 / A100 GPU (12–15 GB VRAM)
- **Local (minimum):** Any machine with internet access for Colab interaction

### Software
```txt
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.24.0
librosa>=0.10.0
soundfile>=0.12.1
evaluate>=0.4.0
jiwer>=3.0.0            # WER computation
```

---

## 📚 Literature Review Highlights

| Study | Augmentation | Model | WER / Accuracy |
|-------|-------------|-------|----------------|
| Shahnawazuddin (2024) | Out-of-domain + In-domain | RNN (CTC) | CER 8.3% — 65% relative improvement |
| Yeung et al. (2021) | F0 perturbation + MFCCs | BLSTM + 4-gram LM | 19.3% WER |
| Kadyan et al. (2021) | VTLN + Prosody/Duration modification | DNN-HMM Hybrid | 32.1% improvement |
| Shivakumar & Narayanan (2022) | Speed perturbation + SpecAugment | ResNet + Transformer | 2.4% WER |
| **Ours** | **SpecAugment + Pitch/Prosody** | **Whisper-small (fine-tuned)** | **22.43% WER** |

---

## 🔍 SWOT Analysis

| Strengths | Weaknesses |
|-----------|------------|
| Data-efficient augmentation via SpecAugment | Limited dataset size for Hindi child speech |
| Transfer learning from pretrained Whisper | Colab session time constraints (12-hour limit) |
| End-to-end architecture, no external LM needed | Inference speed not yet optimized for mobile |

| Opportunities | Threats |
|--------------|---------|
| Expand to regional Indian languages | Noisy real-world audio environments |
| Integrate LLMs for code-switching support | Ethical concerns in collecting child data |
| Deploy as scalable mobile/web application | Risk of overfitting on small datasets |

---

## 💡 Applications

- 🏫 **Educational Tools** — Voice-controlled learning platforms for children
- ♿ **Accessibility** — Assistive speech interfaces for children with disabilities
- 🏥 **Healthcare** — Speech therapy evaluation and monitoring tools
- 🌐 **Multilingual Learning** — Extend to regional languages (Punjabi, Tamil, etc.)
- 📱 **Child-Friendly Voice Apps** — Interactive storytelling and tutoring assistants

---

## 📈 Future Work

- [ ] Expand dataset coverage across diverse regional Indian languages
- [ ] Integrate Large Language Models (LLMs) to improve contextual understanding and code-switching
- [ ] Real-time ASR optimization for edge/mobile deployment
- [ ] Explore child-to-child and teenager-to-child voice conversion for augmentation
- [ ] Deploy as a production-grade, scalable web and mobile application
- [ ] Evaluate with standardized child speech benchmarks (MyST, CMU Kids)

---

## 📖 References

1. Shahnawazuddin, S. (2024). Developing children's ASR system under low-resource conditions using end-to-end architecture. *Digital Signal Processing*, *146*, 104385.

2. Yeung, G., Fan, R., & Alwan, A. (2021). Fundamental frequency feature normalization and data augmentation for child speech recognition. *ICASSP 2021*, IEEE, pp. 6993–6997.

3. Kadyan, V., Shanawazuddin, S., & Singh, A. (2021). Developing children's speech recognition system for low-resource Punjabi language. *Applied Acoustics*, *178*, 108002.

4. Shivakumar, P. G., & Narayanan, S. (2022). End-to-end neural systems for automatic children speech recognition: An empirical study. *Computer Speech & Language*, *72*, 101289.

---

## 👨‍💻 Authors

**Shivam Singh** · **Omi Kumari**

*B.Tech CSE CCVT — School of Computer Science, UPES Dehradun*
*Guided by: Kshitij Kumre*

---

<div align="center">

*Made with ❤️ at UPES Dehradun, 2025–2026*

</div>