# 🌦 Weather Prediction & Climate Analysis with LSTM + XGBoost Ensemble

## 📌 Overview

This project predicts weather conditions using **deep learning (LSTM)** for temporal patterns and **XGBoost** for tabular features, combined via **ensemble learning** to boost performance.
It also includes **climate trend analysis**, **environmental impact assessment**, and **spatial visualisations**.

With this pipeline, we achieved:
* **XGBoost accuracy**: **85%**
* **LSTM accuracy**: \~59%
* **Ensemble accuracy**: **95%** (massive boost by combining models)

---

## 🚀 Features

* **Data Preprocessing**

  * Handles **categorical** and **numerical** features
  * Scales numeric values with **MinMaxScaler**
  * Encodes categorical features for models
* **Time Series Modelling**

  * LSTM sequences weather data for each location
  * Sequence length configurable (default: 24 timesteps)
* **Tabular Modeling**

  * XGBoost handles rich categorical + numeric features
  * Strong baseline for classification
* **Ensemble Learning**

  * Weighted probability averaging between LSTM & XGBoost
  * Significantly higher accuracy than individual models
* **Exploratory Data Analysis**

  * Climate trends (monthly/yearly temperature changes)
  * Air quality correlation heatmaps
  * Feature importance with XGBoost
  * Geographical mapping with **Folium**
* **Outlier Detection**

  * Boxplots for pre/post cleaning
* **Visualization**

  * Climate patterns
  * Air quality relationships
  * Country-level temperature comparisons

---

## 🧠 Models

### 1️⃣ **LSTM**

* Input: `(seq_length, num_features)`
* Captures temporal dependencies
* Architecture:

  ```python
  LSTM(64) → Dropout(0.2) → Dense(64, relu) → Dropout(0.2) → Dense(num_classes, softmax)
  ```
* Loss: `categorical_crossentropy`
* Optimizer: `adam`

### 2️⃣ **XGBoost**

* Handles mixed feature types
* Fast, interpretable, high-performing on tabular data
* Feature importance extraction

### 3️⃣ **Ensemble**

* Weighted probability blending:

  ```python
  final_probs = α * p_lstm + (1 - α) * p_xgb
  final_preds = argmax(final_probs)
  ```
* Optimal weight (`α`) tuned via validation set

---

## 📊 Results

| Model    | Accuracy |
| -------- | -------- |
| LSTM     | \~59%    |
| XGBoost  | \~85%    |
| Ensemble | **95%**  |

Classification report (Ensemble):

* **Precision**: 0.99 (macro)
* **Recall**: 0.89 (macro)
* **Weighted F1**: 0.95

---

## 🌍 Climate & Environmental Analysis

### 📈 Climate Trends

* Monthly temperature changes per location
* Long-term patterns visualisation

### 🌫 Air Quality Analysis

* Correlation heatmap between weather & air quality:

  * CO, Ozone, NO₂, SO₂, PM2.5, PM10

### 📌 Feature Importance

* XGBoost feature importance plot
* Shows top predictors for weather classification

### 🗺 Spatial Patterns

* Interactive **Folium** map
* Red dots: warmer locations, Blue dots: cooler locations

### 🌡 Geographical Patterns

* Average temperature per country
* Top/bottom 15 countries shown with horizontal bars

---

## 🛠 Installation

```bash
# Clone repository
https://github.com/SrishtiTurki/Weather-Prediction-Forecast.git

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
tensorflow
xgboost
matplotlib
seaborn
folium
```

---

## ▶️ Usage

```python
# 1. Preprocess data
# 2. Train LSTM & XGBoost
# 3. Blend predictions
# 4. Run analysis & plots
```

Outputs:

* Ensemble predictions
* Visualizations (`/images`)
* Interactive map (`weather_map.html`)

---
