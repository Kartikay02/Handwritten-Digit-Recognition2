# Handwritten Digit Recognition using LeNet-5

## 📌 Overview
The Handwritten Digit Recognition project focuses on building a deep learning model capable of accurately identifying digits (0–9) from handwritten images. This model was developed using the LeNet-5 architecture, a pioneering convolutional neural network (CNN), implemented with the PyTorch deep learning framework.

The system leverages the MNIST dataset, a widely-used benchmark dataset consisting of 60,000 training images and 10,000 testing images of grayscale handwritten digits. The model learns hierarchical representations of the digits through convolutional and pooling layers, followed by fully connected layers that perform the classification.

Key processes in the project included data preprocessing, model design using PyTorch, training the CNN, evaluating model performance, and testing accuracy. The project demonstrated the ability of CNNs to effectively learn spatial hierarchies in image data, achieving high accuracy in digit recognition.

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

Here is the Deployed project online you can check the demo of this project (https://handwritten-digit-recognition2-gdmxrww9vakauv5lt3kj5r.streamlit.app/)
