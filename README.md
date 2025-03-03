# 📰 AI-Based Fake News Detector

This repository contains an **AI-based Fake News Detection System** that classifies news articles as **Fake** or **Real** using **Naïve Bayes Classifier** based on **Bayes' Theorem**. The model is trained using **TF-IDF vectorization** and can be deployed for real-time detection.

---

## 📌 Features

- **Dataset**: Uses the Fake & Real News Dataset from Kaggle
- **Algorithm**: Naïve Bayes (Bayes' Theorem)
- **Libraries Used**: Scikit-learn, Pandas, NumPy, Matplotlib
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score
- **Model Deployment**: Save & Load model using Pickle

---

## 📂 Dataset

The dataset consists of real and fake news articles, with labeled categories:

- `fake.csv`: Contains fake news articles.
- `true.csv`: Contains real news articles.

To download the dataset directly from Kaggle, use:

```python
!pip install kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
!kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset --unzip
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```python
python train.py
```

### 4️⃣ Test the Model

```python
python test.py
```

### 5️⃣ Save & Load the Model

```python
import pickle

# Save Model
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Load Model
with open("fake_news_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
```

---

## 📊 Data Visualization

To visualize the dataset, use the following code:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Data distribution
plt.figure(figsize=(6,4))
sns.countplot(y=["Fake"]*len(df_fake) + ["Real"]*len(df_real))
plt.title("Fake vs Real News Distribution")
plt.show()
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Feel free to open an issue or submit a pull request.

---

## 📝 Author

**Mann Jadhav**

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repository!

