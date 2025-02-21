# 🍏🍊 Apple vs Orange Classification using K-Nearest Neighbors (KNN)

## 📌 Overview
This project demonstrates the classification of apples and oranges using the K-Nearest Neighbors (KNN) algorithm. It explores different values of K (3, 5, and 7) and evaluates the model's performance using accuracy, classification reports, and confusion matrices.

## ✨ Features
- ✅ Data preprocessing and splitting into training and testing sets
- 🤖 Implementation of KNN classifier with different values of K
- 📊 Model evaluation using accuracy score, classification report, and confusion matrix
- 🎨 Visualization of confusion matrices using heatmaps
- 📍 Scatter plot representation of data points and new predictions

## 🛠️ Technologies Used
- 🐍 Python
- 🔢 NumPy
- 🗃️ Pandas
- 📚 Scikit-learn
- 🎨 Seaborn
- 📈 Matplotlib

## 📂 Dataset
The dataset consists of weight and size attributes with corresponding class labels (apple or orange). The data is preprocessed and split into training and testing sets (80%-20% split).

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AbhishekMudaliyar/apple-vs-orange-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd apple-vs-orange-classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
1. 📥 Load and preprocess the dataset:
   ```python
   df = pd.read_csv("apples_and_oranges.csv")
   ```
2. 🎯 Train the model using different K values:
   ```python
   KNN = KNeighborsClassifier(n_neighbors=3)
   KNN.fit(X_train, y_train)
   ```
3. 📊 Evaluate the model:
   ```python
   accuracy_test = accuracy_score(y_test, y_pred_test)
   print(classification_report(y_test, y_pred_test))
   ```
4. 🔍 Visualize confusion matrices:
   ```python
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.show()
   ```
5. 🎨 Predict a new data point and visualize:
   ```python
   new_data_point = np.array([[71, 4.5]])
   plt.scatter(new_data_point[0][0], new_data_point[0][1], color="blue", label="New Data Point")
   plt.show()
   ```

## 📈 Results
The trained KNN model provides different accuracy results for various values of K (3, 5, and 7). The model's performance is evaluated using confusion matrices and classification reports. The new data point is visualized in a scatter plot to observe its classification.

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for any improvements.

## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
For any questions or suggestions, reach out via [abhishekmudaliyar2003@gmail.com].

