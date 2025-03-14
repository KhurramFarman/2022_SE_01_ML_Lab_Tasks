# **Report: Classification of MNIST Handwritten Digits Using Machine Learning**

---

## **1. Introduction**

The MNIST dataset is a widely used benchmark in the field of machine learning and computer vision. It consists of 28x28 grayscale images of handwritten digits (0-9), each flattened into a 1D vector of 784 features. The dataset is split into training and testing sets, stored in CSV files (`mnist_train.csv` and `mnist_test.csv`). The objective of this lab is to experiment with various machine learning models, evaluate their performance, and identify the best-performing model for classifying handwritten digits.

This report documents the methodology, results, and analysis of the classification task using Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Neural Network (MLP) models. The models were trained, tuned, and evaluated based on accuracy, precision, recall, and F1-score.

---

## **2. Methodology**

### **2.1 Dataset Preparation**
The dataset was loaded into Pandas DataFrames for easy manipulation and analysis. The training and testing datasets were split into features (`X_train`, `X_test`) and labels (`y_train`, `y_test`). The dataset was already preprocessed, with images flattened into 784 features and labels provided for each image.

```python
# Loading the dataset
train_file_path = "mnist_train.csv"
test_file_path = "mnist_test.csv"
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Splitting into features and labels
X_train = df_train.drop(columns=['label'])
y_train = df_train[['label']]
X_test = df_test.drop(columns=['label'])
y_test = df_test[['label']]
```

### **2.2 Models Used**
Four machine learning models were implemented and tuned:
1. **Logistic Regression**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **Neural Network (MLP)**

For each model, hyperparameter tuning was performed to identify the best configuration.

#### **2.2.1 Logistic Regression**
- **Hyperparameters Tuned:**
  - `max_iter`: [5, 10, 20]
  - `solver`: ["lbfgs", "saga"]
- **Best Configuration:**
  - `max_iter`: 10
  - `solver`: "saga"
- **Accuracy:** 92.64%

#### **2.2.2 Random Forest**
- **Hyperparameters Tuned:**
  - `n_estimators`: [5, 10, 20]
  - `max_depth`: [None, 20, 30]
- **Best Configuration:**
  - `n_estimators`: 20
  - `max_depth`: 30
- **Accuracy:** 96.10%

#### **2.2.3 K-Nearest Neighbors (KNN)**
- **Hyperparameters Tuned:**
  - `n_neighbors`: [1, 3, 5]
- **Best Configuration:**
  - `n_neighbors`: 3
- **Accuracy:** 97.05%

#### **2.2.4 Neural Network (MLP)**
- **Hyperparameters Tuned:**
  - `hidden_layer_sizes`: [(64,), (128, 64), (256, 128, 64), (512, 256, 128), (1024, 512, 256)]
  - `max_iter`: 50
- **Best Configuration:**
  - `hidden_layer_sizes`: (1024, 512, 256)
  - `max_iter`: 50
- **Accuracy:** 98.13%

### **2.3 Evaluation Metrics**
The performance of each model was evaluated using the following metrics:
- **Accuracy:** Percentage of correctly classified samples.
- **Precision:** Ratio of true positives to the total predicted positives.
- **Recall:** Ratio of true positives to the total actual positives.
- **F1-Score:** Harmonic mean of precision and recall.

Confusion matrices were also plotted to visualize the misclassifications.

---

## **3. Results**

### **3.1 Model Performance Comparison**
The table below summarizes the performance of the models:

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 92.64%   | 0.9263    | 0.9264 | 0.9263   |
| Random Forest          | 96.10%   | 0.9610    | 0.9610 | 0.9610   |
| K-Nearest Neighbors    | 97.05%   | 0.9707    | 0.9705 | 0.9705   |
| Neural Network (MLP)   | 98.13%   | 0.9814    | 0.9813 | 0.9813   |

### **3.2 Confusion Matrix Insights**
The confusion matrices revealed common misclassifications, such as:
- **7** predicted as **1**
- **4** predicted as **9**
- **9** predicted as **0**
- **7** predicted as **4**

These misclassifications are likely due to the similarity in shapes and strokes of certain digits.

### **3.3 Model Performance Ranking**
1. **Neural Network (MLP):** Best-performing model with an accuracy of 98.13%.
2. **K-Nearest Neighbors (KNN):** Second-best with an accuracy of 97.05%.
3. **Random Forest:** Performed well with an accuracy of 96.10%.
4. **Logistic Regression:** Weakest model with an accuracy of 92.64%.

---

## **4. Discussion**

### **4.1 Best Performing Model**
The **Neural Network (MLP)** achieved the highest accuracy (98.13%) and F1-score (0.9813), making it the most effective model for this classification task. Its ability to capture complex patterns in the data, combined with the optimal hyperparameters, contributed to its superior performance.

### **4.2 Trade-offs**
- **Neural Network (MLP):** While it offers the best performance, it is computationally expensive and less interpretable.
- **K-Nearest Neighbors (KNN):** Provides strong performance with relatively simple implementation but can be slow for large datasets.
- **Random Forest:** Balances interpretability and performance, making it a good choice for applications where understanding the model's decisions is important.
- **Logistic Regression:** Simple and efficient but less effective for complex datasets like MNIST.

### **4.3 Misclassification Analysis**
The confusion matrices highlighted that digits with similar shapes (e.g., 7 and 1, 4 and 9) were occasionally misclassified. This suggests that future work could focus on improving feature extraction or using more advanced models like Convolutional Neural Networks (CNNs) to better distinguish between similar digits.

---

## **5. Conclusion**

In this lab, four machine learning models were trained and evaluated on the MNIST dataset. The **Neural Network (MLP)** emerged as the best-performing model with an accuracy of 98.13%. The **K-Nearest Neighbors (KNN)** and **Random Forest** models also demonstrated strong performance, while **Logistic Regression** lagged behind due to its simplicity.

The results indicate that more complex models like Neural Networks are better suited for tasks involving high-dimensional data like image classification. However, simpler models like Random Forest and KNN remain viable options when interpretability and computational efficiency are prioritized.

### **5.1 Future Work**
- Experiment with Convolutional Neural Networks (CNNs) for improved feature extraction.
- Explore data augmentation techniques to reduce misclassifications.
- Investigate ensemble methods to combine the strengths of multiple models.

### **5.2 Saving the Best Model**
The best-performing model (Neural Network) was saved for future use:

```python
import joblib
joblib.dump(best_mlp_model, "best_mlp_model.pkl")
```

--- 

This concludes the report on the classification of MNIST handwritten digits using machine learning.