# Handwritten Digit Recognition using LeNet-5

## 📌 Overview
This project implements a **Handwritten Digit Recognition System** using the **LeNet-5** deep learning architecture in **PyTorch**. The model is trained on the **MNIST dataset**, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The goal is to accurately classify handwritten digits using a convolutional neural network (CNN).

## 🚀 Features
- **Deep Learning Model:** Uses the **LeNet-5** CNN architecture for digit classification.
- **High Accuracy:** Achieves **98%+ accuracy** on the MNIST dataset.
- **Data Augmentation:** Enhances model performance with image transformations.
- **Performance Metrics:** Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
- **Optimized Training:** Implements **batch normalization and dropout** to prevent overfitting.

## 🏗️ Tech Stack
- **Programming Language:** Python
- **Frameworks & Libraries:** PyTorch, NumPy, Matplotlib, Torchvision
- **Dataset:** MNIST (Modified National Institute of Standards and Technology)

## 📂 Project Structure
```
├── dataset/                  # Contains MNIST dataset
├── models/                   # Stores trained models
├── notebooks/                # Jupyter Notebooks for training and evaluation
├── src/
│   ├── train.py              # Script for model training
│   ├── evaluate.py           # Script for model evaluation
│   ├── predict.py            # Script for digit prediction
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies list
```

## 📊 Model Architecture (LeNet-5)
LeNet-5 is a **convolutional neural network** designed for digit recognition, consisting of:
- **Convolutional Layers**: Extract important features from images.
- **Pooling Layers**: Reduce spatial size and computational complexity.
- **Fully Connected Layers**: Classify the extracted features into digit categories (0-9).

## 🔧 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kartikay02/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the training script:**
   ```bash
   python src/train.py
   ```
4. **Evaluate the model:**
   ```bash
   python src/evaluate.py
   ```
5. **Make a prediction:**
   ```bash
   python src/predict.py --image path/to/image.png
   ```

## 📈 Model Performance
| Metric     | Score  |
|------------|--------|
| Accuracy   | 98%+   |
| Precision  | 97%    |
| Recall     | 96%    |
| F1-Score   | 96.5%  |

## 🎯 Future Improvements
- Implement real-time digit recognition.
- Train on a larger dataset for better generalization.
- Experiment with different CNN architectures for improved accuracy.

## 🤝 Contribution
Feel free to **fork**, **contribute**, or suggest improvements. If you find any issues, please open a GitHub **issue**.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Author:** Kumar Kartikay  
📧 **Contact:** [Kartikaykumar550@gmail.com]  
🔗 **GitHub:** [(https://github.com/Kartikay02)]  
