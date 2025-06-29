# 👗🧥 Image Classification with CNN (Fashion MNIST)

This project implements a **Convolutional Neural Network (CNN)** to classify clothing items in the **Fashion MNIST dataset**, enabling automated recognition of apparel categories such as shirts, shoes, and bags.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

**Fashion MNIST** is a dataset of Zalando's article images consisting of **70,000 grayscale images** in **10 categories**, with 60,000 training images and 10,000 test images. Each image is **28x28 pixels**, representing clothing items.

This project trains a **CNN model** to classify these clothing items into their respective categories with high accuracy.



## 📚 Dataset

* **Dataset:** [Fashion MNIST]()
* **Classes (10 total):**

  * 0: T-shirt/top
  * 1: Trouser
  * 2: Pullover
  * 3: Dress
  * 4: Coat
  * 5: Sandal
  * 6: Shirt
  * 7: Sneaker
  * 8: Bag
  * 9: Ankle boot



## ✨ Features

✅ Load and preprocess Fashion MNIST dataset

✅ Build CNN model using Keras (TensorFlow backend)

✅ Train the model with validation

✅ Evaluate performance on test data

✅ Visualize sample predictions with labels



## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* `numpy`
* `matplotlib`
* `seaborn`
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Image-Classification-with-CNN-Fashion-MNIST.git
cd Image-Classification-with-CNN-Fashion-MNIST
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `Fashion_MNIST_CNN.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Import libraries and load dataset
   * Normalize image pixel values
   * Build and compile the CNN model
   * Train the model on training data
   * Evaluate the model on test data
   * Visualize predictions on sample test images



## 🏗️ Model Architecture

Sample CNN architecture used:

* **Conv2D Layer:** 32 filters, (3x3) kernel, ReLU activation
* **MaxPooling2D Layer:** (2x2) pool size
* **Conv2D Layer:** 64 filters, (3x3) kernel, ReLU activation
* **MaxPooling2D Layer:** (2x2) pool size
* **Flatten Layer**
* **Dense Layer:** 128 units, ReLU activation
* **Output Dense Layer:** 10 units (classes), Softmax activation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```



## 📁 Project Structure

```
Image-Classification-with-CNN-Fashion-MNIST/
 ┣ Fashion_MNIST_CNN.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

* **Training Accuracy:** *e.g. 93%*
* **Test Accuracy:** *e.g. 90%*

The model achieves high accuracy in classifying clothing items, demonstrating the effectiveness of CNNs for image recognition tasks.



## 📊 Example Prediction

```python
import numpy as np

# Predict on a single image
index = 15
img = x_test[index].reshape(1,28,28,1)
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
print("Predicted class:", class_names[predicted_label])
```



## 🤝 Contributing

Contributions are welcome to:

* Implement data augmentation for improved accuracy
* Experiment with deeper CNN architectures or transfer learning
* Deploy as a TensorFlow Lite model for mobile apps

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐️ If you find this project useful, please give it a star!

