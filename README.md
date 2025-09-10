```markdown
# 📊 Customer Churn Prediction with Deep Learning (ANN)

This project implements a **Customer Churn Prediction model** using an **Artificial Neural Network (ANN)** in TensorFlow/Keras.  
The app is deployed with **Streamlit**, where users can input customer details and predict whether a customer is likely to churn.

---

## 🚀 Project Structure

```

Churn\_ANN/
1. │── Churn\_Modeling.ipynb    # Google Colab training notebook
2. │── churn\_model.keras       # Trained ANN model
3. │── scaler.pkl              # StandardScaler for numeric features
4. │── app.py                  # Streamlit app for deployment
5. │── requirements.txt        # Dependencies
6. │── Churn\_Modelling.csv     # Original dataset
7. │── README.md               # Project documentation

```

---

## 📂 Dataset

We use the **Churn_Modelling.csv** dataset, which contains 10,000 customers' information including demographics, account details, and churn status (`Exited`).  

During preprocessing:
- Removed: `RowNumber`, `CustomerId`, `Surname`, `Exited`
- Encoded: `Gender` (binary) and `Geography` (one-hot)
- Scaled: Only **8 numeric features**

---

## 🧠 Model Training (Notebook)

The model was trained in **Google Colab** with the following pipeline:

1. **Data Preprocessing**  
   - Dropped unnecessary columns  
   - One-hot encoded categorical variables  
   - Standardized numeric features  

2. **Feature Order Used in Training**  

The `StandardScaler` was trained **only on these 8 numeric features in this exact order**:
```

\['CreditScore', 'Age', 'Tenure', 'Balance',
'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

````

⚠️ **Important**: Any prediction must follow this same order, otherwise the scaler will raise an error.

3. **Model Architecture**  
   - Input layer: 11 neurons  
   - Hidden layers: 2 dense layers (ReLU activation)  
   - Output layer: 1 neuron (Sigmoid activation for churn probability)  
   - Optimizer: Adam  
   - Loss: Binary Crossentropy  

4. **Saving Artifacts**  
   ```python
   import joblib
   joblib.dump(sc, "scaler.pkl")         # Save scaler
   classifier.save("churn_model.keras")  # Save ANN model
````

---

## 🌐 Streamlit App (`app.py`)

The Streamlit app takes user input (credit score, age, balance, etc.), preprocesses it in the **same way as training**, and makes churn predictions.

### Preprocessing in App:

* **Scale only the 8 numeric features** using `scaler.pkl`
* **Append categorical encodings** (`Gender`, `Geography`)
* Final input vector = 12 features:

  ```
  [Scaled numeric (8)] + [Gender (1)] + [Geography one-hot (3)]
  ```

---

## ⚙️ Installation & Setup

1. Clone this repo:

   ```bash
   git clone https://github.com/vj220803/Churn_ANN.git
   cd Churn_ANN
   ```

2. Create virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this project to GitHub.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → "New app".

3. Connect your GitHub repo, select `app.py`.

4. Add dependencies in `requirements.txt`.
   Example:

   ```
   streamlit
   numpy
   tensorflow
   scikit-learn
   joblib
   ```

5. Deploy 🚀

---

## 📊 Example Prediction

* Input:

  ```
  CreditScore = 600
  Gender = Male
  Age = 40
  Tenure = 3
  Balance = 60000
  NumOfProducts = 2
  HasCrCard = 1
  IsActiveMember = 1
  EstimatedSalary = 50000
  Geography = France
  ```

* Output:

  ```
  ✅ This customer is not likely to churn. (Probability: 0.23)
  ```

---

## 📌 Notes

* Always ensure **feature order** matches the training pipeline.
* If you retrain the model, re-save both `churn_model.keras` and `scaler.pkl`.
* For reproducibility, use the same preprocessing pipeline as in the notebook.

---

## ✨ Author

Developed by **Vijayan Naidu** 👨‍💻
M.Sc. Data Science | Fergusson College

---


