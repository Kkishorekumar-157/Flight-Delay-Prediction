# Flight-Delay-Prediction
Machine Learning-based flight delay prediction using historical congestion, probability calibration and cost-sensitive classification.
# Reliable and Cost-Sensitive Flight Delay Prediction Using Historical Airport Congestion and Probability Calibration

## Project Overview

Flight delays are a major challenge in the aviation industry, affecting passengers, airlines, airport operations, and scheduling efficiency.

This project develops a Machine Learning-based flight delay prediction system using historical US flight data from 2024. It investigates how historical airport, airline, and route congestion information can improve prediction performance.

The project also explores probability calibration and cost-sensitive classification to improve the reliability and practical usefulness of flight delay predictions.

## Research Objectives

- Develop a Machine Learning model to predict whether a flight will arrive at least 15 minutes late.
- Investigate the contribution of historical airport, airline, and route congestion features.
- Compare probability calibration techniques to improve prediction reliability.
- Optimize classification thresholds based on different false-positive and false-negative cost ratios.
- Evaluate the model using classification and probability-based performance metrics.

## Dataset

**Dataset:** US Flight Data – 2024

**Target Variable:** `ArrDel15`

- `0` – Flight arrived without a delay of 15 minutes or more.
- `1` – Flight arrived 15 minutes or more late.

The dataset contains flight scheduling information, airline details, origin and destination airports, and engineered historical congestion features.

## Methodology

### 1. Data Preprocessing and Feature Engineering

- Data cleaning and missing-value handling.
- Feature selection and preparation.
- Encoding categorical variables using One-Hot Encoding.
- Scaling numerical features using Standard Scaling.
- Construction of historical congestion features.

### 2. Historical Congestion-Based Feature Ablation

Five feature configurations were evaluated:

| Configuration | Features |
|---|---:|
| Flight Only | 11 |
| Flight + Airport | 19 |
| Flight + Airline | 15 |
| Flight + Route | 15 |
| Flight + All Historical | 27 |

The experiment investigates the contribution of different historical congestion feature groups.

### 3. Probability Calibration

Three probability estimation approaches were compared:

- Raw XGBoost probabilities
- Platt Scaling
- Isotonic Regression

Evaluation metrics:
- Brier Score
- Log Loss
- Expected Calibration Error (ECE)

### 4. Cost-Sensitive Threshold Optimization

Different false-negative and false-positive cost ratios were evaluated:

- 1:1
- 2:1
- 5:1
- 10:1
- 20:1

The objective is to identify classification thresholds that minimize operational misclassification costs.

## Machine Learning Model

**Primary Algorithm:** XGBoost Classifier

The same model configuration was used across the five feature-ablation experiments to maintain consistency.

## Experimental Results

### Feature Ablation Results

| Feature Configuration | Accuracy | ROC-AUC | PR-AUC | F1 Score |
|---|---:|---:|---:|---:|
| Flight Only | 0.8133 | 0.7419 | 0.4571 | 0.1988 |
| Flight + Airport | 0.8189 | 0.7538 | 0.4839 | 0.2617 |
| Flight + Airline | 0.8152 | 0.7440 | 0.4633 | 0.2235 |
| Flight + Route | 0.8142 | 0.7448 | 0.4593 | 0.2267 |
| Flight + All Historical | **0.8193** | **0.7559** | **0.4867** | **0.2710** |

**Observation:** Combining all historical congestion features produced the best overall performance. Airport-level historical features provided the strongest individual improvement.

### Probability Calibration Results

| Calibration Method | Brier Score | Log Loss | ECE |
|---|---:|---:|---:|
| Raw XGBoost | 0.13457 | 0.42837 | 0.01618 |
| Platt Scaling | 0.13439 | 0.42873 | 0.01717 |
| Isotonic Regression | **0.13397** | **0.42684** | **0.00244** |

**Observation:** Isotonic Regression achieved the best results across all three calibration metrics.

### Cost-Sensitive Classification

At a 5:1 false-negative to false-positive cost ratio, the Platt Scaling model achieved:

| Metric | Result |
|---|---:|
| Optimal Threshold | 0.15 |
| False Positives | 109,189 |
| False Negatives | 18,580 |
| Recall | 74% |
| F1 Score | 45% |
| Expected Cost per Flight | 0.5642 |

**Observation:** Increasing the cost of missing a delayed flight reduces the optimal classification threshold, allowing the model to identify more potentially delayed flights.

## Technology Stack

- **Programming Language:** Python
- **Environment:** Jupyter Notebook
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Visualization:** Matplotlib, Seaborn

## Repository Structure

```text
flight-delay-prediction/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── flight_delay_prediction.ipynb
│
├── results/
│   ├── feature_ablation_results.csv
│   └── calibration_results.csv
│
└── images/
    ├── feature_comparison.png
    ├── reliability_diagram.png
    ├── confusion_matrix.png
    └── threshold_sensitivity.png
```

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/flight-delay-prediction.git
```

2. Navigate to the project directory:

```bash
cd flight-delay-prediction
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Place the dataset in the appropriate data directory.

5. Open Jupyter Notebook:

```bash
jupyter notebook
```

6. Open and execute the project notebook.

## Limitations and Future Work

- Conduct chronological train-test evaluation to assess future-flight prediction performance.
- Further investigate the temporal validity of historical congestion features.
- Explore additional Machine Learning and Deep Learning algorithms.
- Evaluate model performance across different airlines, airports, and routes.
- Investigate more realistic operational cost structures.

## Research Contribution

This project combines three connected investigations:

1. Historical congestion-aware flight delay prediction.
2. Probability calibration for more reliable delay-risk estimates.
3. Cost-sensitive threshold optimization for operational decision-making.

The objective is to move beyond conventional binary classification and investigate more reliable and operationally meaningful flight delay prediction.

## Disclaimer

This project is developed for academic and research purposes. The reported experimental results are based on the current dataset split and evaluation setup. Further chronological validation is required before making claims about real-world future-flight deployment.
