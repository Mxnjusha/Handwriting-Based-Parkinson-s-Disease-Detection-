# Handwriting-Based Parkinson’s Disease Detection  
### A Hybrid ResNet50 + SVM Approach

👩‍🎓 **Author:** Manjusha K  
🎓 **Program:** MSc Big Data Analytics  
🏫 **Institution:** St. Joseph’s University, Bengaluru  

---

## 📌 Abstract
Parkinson’s Disease (PD) is a progressive neurological disorder that affects motor control,
often reflected through handwriting and drawing impairments.
This project presents a **hybrid deep learning framework** that combines
**ResNet50-based feature extraction** with a **Support Vector Machine (SVM) classifier**
to accurately detect Parkinson’s Disease using handwriting images.

The proposed system achieves high classification performance and is deployed
as a **Flask-based web application** for real-time clinical assistance.

---

## 🧠 Methodology
- Image preprocessing and augmentation
- Transfer learning using **ResNet50 (ImageNet pretrained)**
- Extraction of deep feature vectors
- Classification using **SVM with RBF kernel**
- Performance evaluation using accuracy, sensitivity, specificity, and AUC
- Web deployment using **Flask**

---

## 📂 Project Structure
parkinsons-handwriting-resnet50-svm/
│
├── paper/ # Research paper
├── src/ # Training & preprocessing scripts
│ ├── Augmented_data.py
│ ├── train_model.py
│ └── model.py
│
├── app/ # Flask application
│ ├── app.py
│ ├── templates/
│ └── static/
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 📊 Dataset
- **HandPD Dataset**
- Publicly available handwriting and drawing dataset for Parkinson’s Disease

🔗 Dataset Link:  
https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/

⚠️ *Dataset is not included in this repository due to size and licensing constraints.*

---

## 🏆 Results
- **Accuracy:** 97.47%
- **Sensitivity:** 97.60%
- **Specificity:** 97.35%
- **AUC:** 0.9891

The hybrid ResNet50 + SVM model outperforms traditional CNN classifiers
in robustness and generalization.

---

## 🚀 Web Application
A Flask-based web interface allows users to:
- Upload handwriting or drawing images
- Receive instant Parkinson’s Disease predictions
- Assist clinicians in early-stage diagnosis

---

## ⚙️ Installation & Usage

```bash
pip install -r requirements.txt
python app/app.py
