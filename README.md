# 📊 Customer Churn Prediction using ANN

This project predicts **customer churn** (whether a customer will leave the bank) using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras. The model is deployed with **Streamlit** for an interactive web app.  

---

## 📂 Project Structure

Churn_ANN/
│── app.py # Streamlit app
│── churn_model.keras # Trained ANN model
│── scaler.pkl # StandardScaler used during training
│── Churn_Modeling.ipynb # Google Colab notebook (model training)
│── requirements.txt # Python dependencies
│── Churn_Modelling.csv # Dataset (Kaggle bank churn dataset)
│── README.md # Project documentation


---

## ⚙️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** → ANN model  
- **Scikit-learn** → Data preprocessing (scaling, encoding)  
- **Streamlit** → Web app deployment  
- **NumPy & Pandas** → Data handling  
- **Matplotlib & Seaborn** → Visualization (EDA in notebook)  

---

## 📑 Dataset

The dataset used is the **Churn Modelling Dataset** (commonly available on Kaggle).  

### Features:
- **CreditScore** – Numerical credit score of customer  
- **Geography** – Country (France, Germany, Spain)  
- **Gender** – Male/Female  
- **Age** – Customer age  
- **Tenure** – Number of years with the bank  
- **Balance** – Account balance  
- **NumOfProducts** – Number of bank products held  
- **HasCrCard** – Has credit card (0/1)  
- **IsActiveMember** – Active membership status (0/1)  
- **EstimatedSalary** – Customer salary  

### Target:
- **Exited** → `1 = Churn` , `0 = Not Churn`  

### Dropped Columns:
- `RowNumber`, `CustomerId`, `Surname` → identifiers, not useful for training.  

---

## 🧠 Model Training

1. **Data Preprocessing**  
   - Encoded categorical variables:  
     - Gender → Binary (Male=1, Female=0)  
     - Geography → One-hot (France, Germany, Spain)  
   - Standardized numerical features using `StandardScaler`.  

2. **Artificial Neural Network (ANN)**  
   - Input layer: 12 features  
   - Hidden Layers: Dense layers with ReLU activation  
   - Output Layer: Sigmoid activation (binary classification)  

3. **Compilation**  
   - Optimizer: `adam`  
   - Loss: `binary_crossentropy`  
   - Metric: `accuracy`  

4. **Saving Model & Scaler**  
   ```python
   import joblib
   # Save scaler
   joblib.dump(sc, "scaler.pkl")

   # Save trained ANN model
   classifier.save("churn_model.keras")


## Running the Streamlit App
1️⃣ Install dependencies
Make sure you are in the project folder and run:
pip install -r requirements.txt

2️⃣ Run the app
streamlit run app.py

3️⃣ Use the app
Fill in customer details (Credit Score, Age, Geography, Gender, Balance, etc.)
The app will preprocess the input (scaling + encoding).
The ANN model will predict Churn Probability.

✅ Not likely to churn ⚠️ Likely to churn

🌐 Deployment on Streamlit Cloud
Push your project folder to GitHub (include all required files).

Go to Streamlit Cloud.

Sign in with GitHub and select your repository.

Deploy 🚀

✅ Ensure that your repo has at least:

app.py

requirements.txt

churn_model.keras

scaler.pkl

## Future Enhancements
1. Add SHAP/LIME for interpretability of churn predictions
2. Support additional countries and datasets
3. Build an API for integration with real banking systems
4. Deploy on Docker + AWS/GCP for production use

## 👩‍💻 Author
Vijayan Naidu
M.Sc. Data Science | Fergusson College, Pune