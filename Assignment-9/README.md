# Image Classification using Convolutional Neural Networks (CNN)

This project implements an end-to-end Deep Learning pipeline to classify images into Cats and Dogs for an animal welfare organization. The model is built using **TensorFlow/Keras** and structured inside a Jupyter Notebook with complete data preprocessing, training, evaluation, and visualizations.

---

## 🎯 Objective
The primary objective of this assignment is to build, train, and evaluate a Convolutional Neural Network (CNN) that automatically classifies images as either **Cat** or **Dog**. Automating this process helps animal welfare organizations categorize pet photos for adoption portals quickly and efficiently.

---

## 📊 Dataset Link
The project uses the standard Kaggle dataset for Cats and Dogs:
* **Kaggle Dataset**: [Cats and Dogs Dataset (tongpython/cat-and-dog)](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
* **Dataset Structure**: The dataset contains 10,000 images divided into:
  - **Training Set**: 8,000 images (4,000 cats and 4,000 dogs).
  - **Test Set**: 2,000 images (1,000 cats and 1,000 dogs).

---

## 🛠️ Libraries Used
The implementation utilizes the following standard Python libraries:
* **TensorFlow/Keras**: For building the CNN sequential model, compiling, and training.
* **Kagglehub**: For programmatically downloading the dataset from Kaggle.
* **Pillow (PIL)**: For image file verification and processing.
* **Pandas & NumPy**: For file path structuring, data splitting, and numerical matrix operations.
* **Scikit-Learn**: For dataset splitting and extracting classification performance metrics (precision, recall, F1-score).
* **Matplotlib & Seaborn**: For plotting training curves, sample images, and the confusion matrix.

---

## ⚙️ Methodology

1. **Data Understanding**:
   - Downloaded the dataset programmatically via `kagglehub`.
   - Verified folder structures and file integrity, removing corrupted/zero-byte images.
   - Identified the number of classes (2: Cat, Dog) and logged dimensions of sample images.
   - Displayed exactly 5 sample images with their labels.

2. **Data Preprocessing**:
   - Loaded images and resized them to a uniform dimension of $128 \times 128$ pixels.
   - Normalized pixel intensity values from $[0, 255]$ to the range $[0.0, 1.0]$ to optimize gradient descent.
   - Split a stratified subset of 2,000 images (1,000 cats and 1,000 dogs) into an **80% Training Set** (1,600 images) and a **20% Testing Set** (400 images).
   - Constructed Keras `ImageDataGenerator` data loaders for both sets.

3. **Model Development**:
   - Built a feedforward CNN with three convolutional blocks, max pooling, flattening, a dense hidden layer, and a single sigmoid output node.
   - Compiled the model with the **Adam optimizer**, **Binary Crossentropy** loss, and **Accuracy** as the evaluation metric.
   - Trained the network for exactly **10 epochs**.

4. **Model Evaluation**:
   - Measured final test accuracy, precision, recall, and F1-score on the test dataset.
   - Plotted a Confusion Matrix using Seaborn to visualize classification counts.
   - Generated learning curves showing training vs. validation accuracy and loss over 10 epochs.
   - Summarized structural performance observations.

---

## 🧠 CNN Architecture

The model is built using the Keras `Sequential` API with the following layers:

| Layer Number | Layer Type | Details (Filters, Kernel Size, Neurons) | Activation | Output Shape | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Conv2D** | 32 filters, $3\times3$ | ReLU | (126, 126, 32) | 896 |
| 2 | **MaxPooling2D** | Pool size $2\times2$ | - | (63, 63, 32) | 0 |
| 3 | **Conv2D** | 64 filters, $3\times3$ | ReLU | (61, 61, 64) | 18,496 |
| 4 | **MaxPooling2D** | Pool size $2\times2$ | - | (30, 30, 64) | 0 |
| 5 | **Conv2D** | 128 filters, $3\times3$ | ReLU | (28, 28, 128) | 73,856 |
| 6 | **MaxPooling2D** | Pool size $2\times2$ | - | (14, 14, 128) | 0 |
| 7 | **Flatten** | Reshapes $14\times14\times128$ to 1D vector | - | (25088) | 0 |
| 8 | **Dense** | Hidden layer with 128 units | ReLU | (128) | 3,211,392 |
| 9 | **Dense (Output)**| Binary classification unit | Sigmoid | (1) | 129 |

* **Total Trainable Parameters**: 3,304,769

---

## 📈 Results

* **Test Dataset Accuracy**: **~68.8%**
* **Precision**: **~73.6%**
* **Recall**: **~58.5%**
* **F1-Score**: **~65.2%**

### Learning Curves & Visualizations
The model output visualizations are saved in the project structure:
1. **Sample Images Plot**: [images/sample_images.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-9/images/sample_images.png)
2. **Confusion Matrix Heatmap**: [images/confusion_matrix.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-9/images/confusion_matrix.png)
3. **Training & Validation Curves**: [images/learning_curves.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-9/images/learning_curves.png)

### Key Observations
* **Overfitting**: The training accuracy reaches ~90% by the 10th epoch, while the validation/test accuracy peaks around 68.8%. The rising validation loss in later epochs is a clear indicator of overfitting.
* **Balanced Classification**: The confusion matrix exhibits an even distribution of errors (similar rates of False Positives and False Negatives), verifying that the model does not have a bias toward either class.
* **Feature Representation**: Early layers capture simple features like edges and color contrasts, while deeper Conv2D layers learn complex patterns such as ear shapes, eyes, and fur textures.

---

## 🏁 Conclusion

This assignment successfully developed a CNN image classifier distinguishing between cats and dogs, achieving ~68.8% accuracy on our dataset subset. The model uses **convolution layers** to extract hierarchical spatial features (like edges, shapes, and textures) and **pooling layers** to downsample the feature maps, reducing dimensions and providing translation invariance.

A primary advantage of CNNs over standard Feedforward ANNs is their **parameter sharing and local connectivity**. In a CNN, filters scan local receptive fields, preserving the grid-like spatial relationships of pixels. An ANN, by contrast, requires flattening the image into a 1D vector, destroying 2D spatial context and ballooning the number of parameters. A key limitation of CNNs is their **susceptibility to overfitting on small datasets**. Without techniques like dropout, batch normalization, or data augmentation, CNNs easily memorize training patterns rather than generalizing, requiring large volumes of annotated data to achieve optimal test scores.
