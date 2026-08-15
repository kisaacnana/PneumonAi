# 🫁 PneumonAi

### Pneumonia Classification from Chest X-Ray Images Using Deep Learning

PneumonAi is a **computer vision and deep-learning project** that explores binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** classes using TensorFlow/Keras.

The project has evolved from an academic CNN experiment into a more reproducible ML workflow with model evaluation, explainability, automated tests, and an optional Streamlit demonstration interface.

> ⚠️ **Medical disclaimer:** PneumonAi is an educational/research project and is **not a medical diagnostic device**. Predictions must not be used to diagnose, treat, triage, or make clinical decisions about a patient.

---

## ✨ What the project demonstrates

- 🧠 CNN-based medical image classification
- 🔄 Reproducible training with a fixed random seed
- 🧪 Precision, recall, F1-score and ROC-AUC evaluation
- 🔥 Grad-CAM model explainability
- 🖥️ Optional Streamlit demonstration interface
- ✅ Automated smoke tests and GitHub Actions CI
- 📦 Dependency management and repository hygiene

---

## 🧠 Pipeline

```text
                  Chest X-Ray
                       │
                       ▼
              Image Preprocessing
                       │
                       ▼
              Data Augmentation
                       │
                       ▼
                CNN Classifier
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Probability         Grad-CAM
             │                   │
             ▼                   ▼
       Classification      Visual Explanation
```

The modern training pipeline uses 224×224 RGB inputs, convolutional feature extraction, pooling, dropout, binary cross-entropy, Adam optimization, early stopping, and checkpointing.

---

## 📊 Dataset

The original project uses the **Chest X-Ray Images (Pneumonia)** dataset published by Paul Mooney on Kaggle.

[Dataset on Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset is **not included in this repository**. Do not commit medical images, patient-identifiable information, trained model artifacts, or credentials.

For the modern training pipeline, arrange the data as:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/kisaacnana/PneumonAi.git
cd PneumonAi
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Kaggle dataset and arrange the folders using the structure above.

### 5. Train

```bash
python Code/train_modern.py
```

The best model is saved to `artifacts/pneumonai.keras` and training plots are written to `artifacts/`.

### 6. Evaluate

```bash
python Code/evaluate.py
```

The evaluation script reports precision, recall, F1-score, confusion matrix and ROC-AUC.

### 7. Generate a Grad-CAM explanation

Place a compatible X-ray at `sample_xray.png`, then run:

```bash
python Code/gradcam.py
```

The generated explanation is saved to `artifacts/gradcam.png`.

> Grad-CAM is an interpretability aid. A highlighted region does **not** prove that the model is using clinically valid features.

### 8. Run the demonstration app

```bash
streamlit run app.py
```

The app expects a trained model at `artifacts/pneumonai.keras`.

---

## 🧪 Testing

Run the smoke tests with:

```bash
pytest -q
```

GitHub Actions automatically runs compilation checks and the test suite on pushes and pull requests to `master`.

---

## 📈 Results

The repository retains the original project's training visualizations:

![Model Accuracy](Figures/accuracy_plot.png)

![Model Loss](Figures/loss_plot.png)

The original project reported approximately **96% accuracy** under its original evaluation setup. This number is preserved as historical project context only; it is **not** a result from the modern pipeline and must not be interpreted as clinical diagnostic accuracy.

When the modern pipeline is trained, report its actual test-set metrics here rather than copying the historical result.

---

## 🔬 Evaluation Philosophy

For a binary medical-image classifier, accuracy alone is insufficient. PneumonAi therefore exposes multiple metrics:

| Metric | Why it matters |
|---|---|
| Precision | How often positive predictions are correct |
| Recall / Sensitivity | How many positive cases are detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ranking/discrimination performance across thresholds |
| Confusion matrix | Shows the distribution of correct and incorrect predictions |

These metrics still do **not** establish clinical validity, safety, fairness, or regulatory approval.

---

## 🗂️ Project Structure

```text
PneumonAi/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Code/
│   ├── train.py             # Original training implementation
│   ├── test.py              # Original prediction implementation
│   ├── train_modern.py      # Reproducible modern training pipeline
│   ├── evaluate.py          # Evaluation metrics
│   └── gradcam.py           # Grad-CAM explainability
├── Figures/                 # Original project visualizations
├── tests/
│   └── test_smoke.py        # Model pipeline smoke tests
├── app.py                   # Streamlit demonstration
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🛠️ Technology Stack

**Machine Learning**  
Python • TensorFlow • Keras • scikit-learn

**Computer Vision**  
OpenCV • Pillow • CNNs • Grad-CAM

**Visualization**  
Matplotlib • Streamlit

**Engineering**  
Git • GitHub Actions • pytest

---

## ⚠️ Limitations & Responsible Use

- This is an educational/research project, not a clinical system.
- Dataset size and composition limit generalization.
- Dataset bias can affect model behavior across populations and imaging environments.
- A held-out test result does not establish real-world clinical performance.
- Accuracy, ROC-AUC and other aggregate metrics do not measure clinical safety.
- Grad-CAM visualizations should not be interpreted as clinical evidence.
- Real clinical deployment would require rigorous external validation, governance, privacy controls, risk management, human oversight, and applicable regulatory review.

---

## 🔮 Future Roadmap

- [x] Modernize training pipeline
- [x] Add reproducible dependencies
- [x] Add multi-metric evaluation
- [x] Add Grad-CAM explainability
- [x] Add Streamlit demonstration
- [x] Add automated tests and CI
- [ ] Add a documented data-splitting/preprocessing protocol
- [ ] Add confusion-matrix and ROC plots as generated artifacts
- [ ] Benchmark transfer-learning models such as ResNet/EfficientNet
- [ ] Add experiment tracking
- [ ] Add model versioning and reproducibility metadata
- [ ] Add a deployment-ready API only after rigorous validation

---

## 👤 Author

**Nana Ike**  
Computer Engineer • Creative Developer • UI/UX & Motion Designer

- GitHub: [@kisaacnana](https://github.com/kisaacnana)
- LinkedIn: [Nana Ike](https://linkedin.com/in/nana-ike-476a201b)
- Email: kisaacnana@gmail.com

---

## 📄 License

See [`LICENSE`](LICENSE) for license information.

> **Build. Design. Create. Solve.**
