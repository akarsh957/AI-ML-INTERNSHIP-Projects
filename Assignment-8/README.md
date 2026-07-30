# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

This project implements an end-to-end Deep Learning pipeline to recognize and classify handwritten digits (0–9) using the **MNIST dataset**. Automating this process allows postal services to efficiently sort mail and read postal codes. The model is built using **TensorFlow/Keras** and structured within a Jupyter Notebook with comprehensive preprocessing, training, evaluation, and documentation.

---

## 🎯 Objective
The primary objective of this assignment is to develop an Artificial Neural Network (ANN) that automatically recognizes handwritten digits from postal codes. Using the MNIST dataset, we aim to build, train, and evaluate a multi-class classifier that maps $28 \times 28$ grayscale images of handwritten digits to their corresponding numerical values (0–9).

---

## 📊 Dataset Reference
The project utilizes the standard MNIST database of handwritten digits:
* **Official Website**: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
* **CSV Format Source**: [Kaggle MNIST in CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
* **Dataset Structure**: The dataset contains 70,000 samples. Each record consists of a class label (0–9) and 784 pixel values representing a flattened $28 \times 28$ grayscale image with pixel intensity values ranging from 0 (black) to 255 (white).

---

## 🛠️ Libraries Used
The implementation relies on the following standard machine learning and visualization libraries:
* **TensorFlow/Keras**: Model creation (Sequential API), compilation, training, and one-hot encoding.
* **Pandas**: Data loading, structure analysis, and tabular summary statistics.
* **NumPy**: Matrix transformations, flattening, and numeric operations.
* **Scikit-Learn**: Dataset splitting (`train_test_split`), evaluation metrics (accuracy, confusion matrix, classification report).
* **Matplotlib**: Image plotting (visualizing sample digits) and generating learning curves.
* **Seaborn**: Confusion matrix heatmap visualization.

---

## ⚙️ Methodology

1. **Data Understanding**:
   * Downscaled and converted the Keras MNIST dataset into a local CSV format (`mnist_dataset.csv`) to satisfy the requirement of loading via Pandas.
   * Loaded the dataset using `pd.read_csv()` and inspected the first 5 rows.
   * Identified target (`label`) and input features (`pixel1` to `pixel784`).
   * Logged dataset dimensions ($70,000 \text{ rows} \times 785 \text{ columns}$) and verified summary info.
   * Rendered a sample digit using Matplotlib.

2. **Data Preprocessing**:
   * Verified that the dataset contains no missing values (`df.isnull().sum()`).
   * Separated features ($X$) and targets ($y$).
   * Normalized grayscale pixel values (divided by $255.0$) to scale features to the range $[0.0, 1.0]$.
   * Partitioned the dataset using an 80% training and 20% testing split, stratifying the target labels.
   * Transformed the integer labels into a binary class matrix using One-Hot Encoding.

3. **Model Development**:
   * Constructed a feedforward ANN with two fully connected hidden layers and an output layer.
   * Compiled the network using the Adam optimizer, Categorical Crossentropy loss function, and Accuracy metric.
   * Trained the model for 10 epochs with a batch size of 64 and a 10% validation split.
   * Generated class predictions on the test dataset.

4. **Model Evaluation**:
   * Measured final test accuracy.
   * Plotted Accuracy vs. Epoch and Loss vs. Epoch curves.
   * Visualized the Confusion Matrix using a Seaborn heatmap.
   * Printed a detailed Classification Report containing Precision, Recall, and F1-Score for each digit.
   * Recorded key performance observations.

---

## 🧠 Model Architecture

The Artificial Neural Network is built using the Keras Sequential API with the following structure:

| Layer Name | Type | Neurons / Units | Activation Function | Input Dimension | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | Flatten / Dense Input | - | - | 784 | 0 |
| **Hidden Layer 1** | Dense | 128 | ReLU | 784 | 100,480 |
| **Hidden Layer 2** | Dense | 64 | ReLU | 128 | 8,256 |
| **Output Layer** | Dense | 10 | Softmax | 64 | 650 |

* **Total Trainable Parameters**: 109,386

---

## 📈 Results

* **Test Dataset Accuracy**: **~97.7%**
* **Training and Validation curves**: 
  * The learning curves (Accuracy vs. Epoch and Loss vs. Epoch) show smooth convergence, indicating that the Adam optimizer and ReLU activation layers successfully minimize error without overfitting.
* **Confusion Matrix Insights**:
  * Visually similar digits exhibit occasional misclassification (e.g., 4 misclassified as 9, 7 as 9, and 5 as 3).
  * High-contrast digits like 1 and 0 exhibit near-perfect precision and recall.

### Visualizations

The generated visualizations are stored under the [images/](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-8/images) directory:
* **Sample MNIST Digit**: [images/sample_digit.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-8/images/sample_digit.png)
* **Confusion Matrix Heatmap**: [images/confusion_matrix.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-8/images/confusion_matrix.png)
* **Model Learning Curves**: [images/learning_curves.png](file:///d:/AI%20ML%20INTERNSHIP%20Projects/Assignment-8/images/learning_curves.png)

---

## 🏁 Conclusion

In this assignment, we successfully built and evaluated a feedforward Artificial Neural Network (ANN) to automate handwritten digit recognition, achieving a test accuracy of **~97.7%**. This demonstrates the viability of ANNs in streamline-processing mail and postal code reading. 

Hidden layers are critical in ANNs because they perform feature extraction. The initial hidden layer identifies localized elements like lines and edges, while the second hidden layer combines them into complex shapes (curves, loops). A key advantage of Deep Learning over traditional Machine Learning is feature representation learning; the network automatically extracts representations from raw data without manual feature engineering. However, a major limitation of ANNs is their "black-box" nature, offering poor interpretability compared to traditional models like decision trees, making it difficult to explain specific classification errors.
