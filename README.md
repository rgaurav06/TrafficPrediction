# Traffic Volume Prediction Using Ensemble Machine Learning

A complete, production-ready Smart City / IoT web dashboard built with Streamlit for predicting hourly traffic volume using meteorological and temporal features.

---

## 1. Project Title
**Traffic Volume Prediction Using Ensemble Machine Learning**

---

## 2. Project Objective
Urban traffic congestion leads to substantial economic losses, increased carbon emissions, and longer commute times. The objective of this project is to develop an intelligent transportation analytics system that predicts hourly interstate traffic volume using historical traffic counts, weather observations, and temporal calendar attributes. By leveraging ensemble machine learning, traffic management authorities and city planners can proactively forecast congestion patterns, optimize signal timings, and dispatch transit resources efficiently.

---

## 3. Dataset
The project utilizes the **Metro Interstate Traffic Volume Dataset** made available by the Minnesota Department of Transportation (MnDOT) via the UCI Machine Learning Repository:
- **Location:** Westbound Interstate 94 (I-94) connecting Minneapolis and Saint Paul, Minnesota.
- **Records:** Over 48,000 hourly observations spanning from 2012 to 2018.
- **Attributes:** Hourly traffic volume counts paired with local weather conditions (temperature, precipitation, cloud cover, weather classifications) and calendar metadata.

---

## 4. Machine Learning Methodology
The project workflow follows standard end-to-end data science practices:

```
Raw Temporal & Weather Inputs
           │
           ▼
Feature Engineering (Peak-Hour, Weekend, Date Deconstruction)
           │
           ▼
Data Preprocessing Pipeline (SimpleImputer + OneHotEncoder)
           │
           ▼
Ensemble Machine Learning Estimator (XGBoost Regressor)
           │
           ▼
Hourly Traffic Volume Forecast (vehicles/hour)
```

1. **Feature Extraction:** Date and time inputs are transformed into temporal attributes (`hour`, `day`, `month`, `year`, `day_of_week`, `is_weekend`, `is_peak_hour`).
2. **Preprocessing Pipeline:** Encapsulated in a Scikit-Learn `ColumnTransformer`:
   - Numerical attributes (`temp`, `rain_1h`, `snow_1h`, `clouds_all`, etc.) are processed through a median imputer.
   - Categorical attributes (`holiday`, `weather_main`, `weather_description`) are processed through a most-frequent imputer followed by `OneHotEncoder(handle_unknown='ignore')`.
3. **Model Inference:** The engineered feature vector is passed to the trained ensemble regression model to produce hourly volume predictions without retraining.

---

## 5. Features

The model expects the exact 14 features described below:

| Feature Name | Type | Description / Logic |
| :--- | :--- | :--- |
| `holiday` | Categorical | Specific US holiday name or `'None'` for regular days |
| `temp` | Numeric | Average ambient temperature in Kelvin (K) |
| `rain_1h` | Numeric | Rainfall accumulation in the past hour (mm) |
| `snow_1h` | Numeric | Snowfall accumulation in the past hour (mm) |
| `clouds_all` | Numeric | Percentage of cloud coverage (0% - 100%) |
| `weather_main` | Categorical | Primary weather class (e.g., `Clear`, `Clouds`, `Rain`, `Snow`, `Thunderstorm`) |
| `weather_description` | Categorical | Granular weather condition (e.g., `scattered clouds`, `light rain`, `sky is clear`) |
| `hour` | Integer | Hour of the day (`0` to `23`) extracted from selected time |
| `day` | Integer | Day of the month (`1` to `31`) extracted from selected date |
| `month` | Integer | Month of the year (`1` to `12`) extracted from selected date |
| `year` | Integer | Year extracted from selected date |
| `day_of_week` | Integer | Day index (`Monday=0` through `Sunday=6`) |
| `is_weekend` | Binary | `1` if Saturday or Sunday, else `0` |
| `is_peak_hour` | Binary | `1` if hour is `7, 8, 9, 16, 17, 18` (commuter rush hours); otherwise `0` |

---

## 6. Models
This project evaluates multiple ensemble regression paradigms suited for tabular time-indexed data:
- **Random Forest Regressor:** Bagging ensemble of randomized decision trees that minimizes variance.
- **Gradient Boosting Regressor:** Sequential boosting that iteratively corrects residual errors.
- **XGBoost Regressor (Extreme Gradient Boosting):** Scalable, regularized gradient boosted decision tree framework optimized for execution speed and accuracy.
- **Stacking Regressor:** Heterogeneous ensemble utilizing a meta-learner to combine base regressors.

The deployed application loads the serialized model artifact:
- **File:** `traffic_volume_model.joblib`
- **Trained with:** Scikit-Learn `1.6.1` and `xgboost`

---

## 7. Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git (optional)

### Setup Instructions

1. **Clone or Navigate to the Project Directory:**
   ```bash
   cd "traffic-volume-prediction"
   ```

2. **(Recommended) Create and Activate a Virtual Environment:**
   - On Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - On macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

> **Important Compatibility Note:** The model was serialized with Scikit-Learn `1.6.1`. Ensure `scikit-learn==1.6.1` and a compatible `xgboost` version (such as `2.1.4`) are installed as specified in `requirements.txt`.

---

## 8. How to Run

Launch the Streamlit dashboard by executing:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at:
`http://localhost:8501`

---

## 9. Project Structure

```
traffic-volume-prediction/
│
├── app.py                          # Main Streamlit web application & UI
├── Metro_Interstate_Traffic_Volume.csv # Real MnDOT / UCI traffic dataset (48,204 records)
├── traffic_volume_model.joblib      # Pre-trained Scikit-Learn + XGBoost pipeline
├── requirements.txt                # Pinned dependencies (scikit-learn==1.6.1, xgboost, etc.)
├── README.md                       # Comprehensive project documentation
├── .gitignore                      # Git ignore rules for Python & Streamlit
└── assets/
    └── style.css                   # Smart City UI theme and styling rules
```

---

## 10. Future Scope
- **Real-Time Weather API Integration:** Automatically fetch current hourly weather conditions using OpenWeatherMap or Tomorrow.io APIs.
- **Multi-Route Network Expansion:** Expand the prediction architecture to cover multiple highway corridors and arterial intersections.
- **Deep Learning / Spatio-Temporal Models:** Evaluate Temporal Fusion Transformers (TFT) or Graph Neural Networks (GNNs) for city-wide spatio-temporal forecasting.
- **Automated Incident Alerting:** Trigger automated congestion alerts and rerouting recommendations when forecasted traffic exceeds highway capacity.
