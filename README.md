# 🌸 Flower Classifier

A deep learning model that classifies 102 types of flowers using transfer learning with ResNet-18.  
Built with PyTorch and trained on the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

---

## 📁 Project Overview

This project demonstrates:
- Loading and preprocessing image datasets for classification
- Fine-tuning a pretrained ResNet model using PyTorch
- Visualizing predictions
- Evaluating model accuracy on a test set

---

## 🧠 Model Architecture

- **Base model:** ResNet-18 (pretrained)
- **Modified final layer:** `Linear(in_features, 102)` for 102 flower classes
- **Training:** 1 epoch using Adam optimizer and CrossEntropy loss

---

## 📊 Performance

- **Test Accuracy (1 epoch):** ~71%  
- **Device:** Trained and evaluated on CPU

---

## 🚀 How to Run

### 1. Clone and install requirements
```bash
git clone https://github.com/your-username/flower-classifier.git
cd flower-classifier
pip install -r requirements.txt
```

### 2. Train the model

```python src/train.py```

- Downloads the dataset
- Fine-tunes the model for 1 epoch
- Saves model weights to outputs/flower_resnet18.pth

### 3. Predict and evaluate accuracy

```python src/predict.py```

- Loads model
- Displays a prediction on 1 random flower image
- Prints test accuracy over full test set

### Sample Output
## Prediction Example

```
Actual: ball moss
Predicted: purple coneflower
```

### Project Structure
```
flower-classifier/
├── data/               # Flower dataset (ignored in Git)
├── outputs/            # Saved model + predictions
├── src/
│   ├── train.py        # Trains model
│   └── predict.py      # Evaluates & visualizes predictions
├── requirements.txt
├── .gitignore
└── README.md
```

### 📚 Dataset Info

- Oxford 102 Flower Dataset
- 8,189 images of flowers, 102 categories
- Split: Train/Test provided by torchvision

### Author
Christian Kang
Computer Science Graduate | AI Projects & Backend Engineering
GitHub • LinkedIn