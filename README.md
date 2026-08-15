# 🫁 PneumonAi

### Pneumonia Detection from Chest X-Ray Images Using Convolutional Neural Networks

PneumonAi is a **deep-learning computer vision project** that explores the use of a Convolutional Neural Network (CNN) to classify chest X-ray images as **pneumonia** or **normal**.

The project was developed as an exploration of how machine learning can be applied to medical image classification and automated decision-support workflows.

> ⚠️ **Medical disclaimer:** PneumonAi is an educational/research project and is **not a medical diagnostic device**. Its predictions should not be used to diagnose, treat, or make clinical decisions about a patient.

---

## 🎯 Project Objective

Pneumonia can require rapid assessment, and chest X-ray imaging is commonly used as part of the diagnostic process. PneumonAi investigates whether a CNN can learn visual patterns in chest X-ray images that distinguish pneumonia cases from normal cases.

The original project reported approximately **96% accuracy** on its evaluation setup. This figure should be interpreted in the context of the dataset, preprocessing pipeline, model architecture, and evaluation methodology used in the project; it should **not** be interpreted as clinical diagnostic accuracy.

---

## 🧠 How It Works

The workflow is broadly:

```text
Chest X-Ray Image
       ↓
Image Preprocessing
       ↓
CNN Feature Extraction
       ↓
Classification Layer
       ↓
Pneumonia / Normal
```

The model uses:

- Convolutional layers for feature extraction
- Max pooling for spatial downsampling
- A fully connected layer for classification
- A sigmoid output for binary classification
- Data augmentation during training
- Binary cross-entropy loss
- Adam optimization

The training implementation is available in [`Code/train.py`](Code/train.py), while [`Code/test.py`](Code/test.py) demonstrates loading the trained model and making a prediction on a new image.

---

## 📊 Dataset

The model was trained using the **Chest X-Ray Images (Pneumonia)** dataset published by Paul Mooney on Kaggle.

The original project used approximately:

- **5,200 training images**
- **620 test images**
- Two classes: **NORMAL** and **PNEUMONIA**

Dataset source:

[Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset is not included in this repository because of its size and licensing/distribution considerations.

---

## 📈 Results

The repository includes training visualizations for model accuracy and loss:

### Model Accuracy

![Model Accuracy](Figures/accuracy_plot.png)

### Model Loss

![Model Loss](Figures/loss_plot.png)

These plots provide a visual representation of the model's training and validation performance over the training process.

---

## 🗂️ Project Structure

```text
PneumonAi/
├── Code/
│   ├── train.py          # Train and save the CNN model
│   └── test.py           # Load the model and run predictions
├── Figures/
│   ├── accuracy_plot.png # Accuracy visualization
│   └── loss_plot.png     # Loss visualization
├── LICENSE
└── README.md
```

---

## 🛠️ Technology Stack

- **Python**
- **Keras / TensorFlow**
- **NumPy**
- **Matplotlib**
- **Convolutional Neural Networks (CNN)**
- **Computer Vision**

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kisaacnana/PneumonAi.git
cd PneumonAi
```

### 2. Install the required dependencies

The original project does not currently include a pinned `requirements.txt`. For a reproducible setup, the dependencies used by the code include:

```bash
pip install tensorflow keras numpy matplotlib
```

### 3. Prepare the dataset

Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and arrange the training and test directories according to the paths expected by `Code/train.py`.

### 4. Train the model

```bash
python Code/train.py
```

The training script saves the trained model as `pneumonia_model.h5` and generates accuracy/loss plots.

### 5. Run a prediction

Place a compatible image at the location expected by `Code/test.py`, ensure the trained model is available, and run:

```bash
python Code/test.py
```

---

## 🔬 Limitations

This project has several important limitations:

- The model is trained on a relatively small public dataset.
- The architecture is intentionally simple and is not intended to represent a production-grade medical imaging system.
- The evaluation setup does not establish clinical validity or generalization to real-world hospital populations.
- Accuracy alone is insufficient for evaluating a medical classification model; metrics such as sensitivity, specificity, precision, recall, F1-score, ROC-AUC, and calibration would provide a more complete assessment.
- The preprocessing and training configuration should be reproduced and independently validated before drawing conclusions from the reported performance.

---

## 🔮 Future Improvements

Potential next steps include:

- Add a reproducible `requirements.txt` or modern dependency configuration
- Introduce a dedicated validation set
- Report a full set of classification metrics
- Add a confusion matrix and ROC curve
- Experiment with transfer learning using architectures such as ResNet or EfficientNet
- Improve image preprocessing and augmentation
- Add explainability methods such as Grad-CAM
- Build a simple web interface for demonstration purposes
- Add automated tests and CI with GitHub Actions

---

## 📚 Background

The original project was developed as a practical exploration of **machine learning, convolutional neural networks, and medical image classification**.

For background information on pneumonia, see the [American Thoracic Society patient resources](https://www.thoracic.org/patients/patient-resources/resources/top-pneumonia-facts.pdf).

---

## 👤 Author

**Nana Ike**

Computer Engineer • Creative Developer • UI/UX & Motion Designer

- GitHub: [@kisaacnana](https://github.com/kisaacnana)
- LinkedIn: [Nana Ike](https://linkedin.com/in/nana-ike-476a201b)
- Email: kisaacnana@gmail.com

---

## 📄 License

See the [`LICENSE`](LICENSE) file for license information.
