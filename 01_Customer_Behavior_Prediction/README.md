# Customer Behavior Prediction - E-commerce Dataset

## Project Overview
This project demonstrates a machine learning classification model built with real-world e-commerce customer data from Kaggle. The model predicts customer purchase intent using behavioral and demographic features, achieving **82% accuracy**.

## Dataset Information
- **Source:** Kaggle E-commerce Dataset
- **Total Records:** 12,330 customer session events
- **Features:** 18 behavioral and demographic variables
- **Target Variable:** Purchase (Binary: 0=No Purchase, 1=Purchase)
- **Features:**
  - Administrative Pages Visited
  - Informational Pages Visited
  - Product Pages Visited
  - Bounce Rate
  - Exit Rate
  - Page Values
  - Special Day
  - Month
  - Operating System
  - Browser Type
  - Region
  - Traffic Type
  - Visitor Type
  - Weekend (Boolean)
  - Session Duration

## Project Structure
```
01_Customer_Behavior_Prediction/
├── README.md
├── customer_prediction.ipynb
├── customer_prediction.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── ecommerce_data.csv
│   └── processed/
│       └── features_engineered.csv
├── models/
│   └── best_model.pkl
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## Key Achievements
✅ **82% Accuracy** - RandomForest Classification  
✅ **0.80 Precision** - Low false positives  
✅ **0.84 Recall** - Good positive detection  
✅ **0.88 ROC-AUC** - Excellent discrimination  
✅ **Feature Engineering** - RFM scores, session quality metrics  
✅ **Class Imbalance Handling** - SMOTE application  

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical analysis of customer behavior
- Distribution analysis of features
- Correlation matrix and heatmaps
- Missing value assessment
- Class balance analysis

### 2. Data Preprocessing
- Handling missing values
- Outlier detection and treatment
- Feature scaling (StandardScaler)
- Categorical encoding (LabelEncoder)

### 3. Feature Engineering
- **RFM Scores:** Recency, Frequency, Monetary segments
- **Behavioral Metrics:** Session quality, engagement score
- **Temporal Features:** Day of week, seasonality
- **Device Features:** Browser and OS encoding

### 4. Model Development
- Train-test split (80-20)
- SMOTE for class imbalance
- Random Forest with GridSearchCV
- Hyperparameter tuning
- Cross-validation (5-fold)

### 5. Model Evaluation
- Classification metrics
- Confusion matrix analysis
- ROC-AUC curve
- Feature importance analysis

## Results & Insights

### Model Performance
- Accuracy: 82%
- Precision: 80%
- Recall: 84%
- F1-Score: 82%
- ROC-AUC: 0.88

### Top Predictive Features
1. Page Values (importance: 24%)
2. Product Pages Visited (importance: 18%)
3. Session Duration (importance: 16%)
4. Bounce Rate (importance: 14%)
5. Visitor Type (importance: 12%)

### Business Insights
- High page values strongly indicate purchase intent
- Product page engagement is critical
- Session duration > 5 minutes correlates with 70% purchase probability
- Weekend visitors have 15% higher purchase rate
- Returning visitors have 35% higher conversion rate

## Technologies Used
- **Python 3.9+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **Imbalanced-learn** - SMOTE for class balance
- **Jupyter Notebook** - Interactive analysis

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
```

### Execute Analysis
```bash
# Option 1: Run Jupyter Notebook
jupyter notebook customer_prediction.ipynb

# Option 2: Run Python script
python customer_prediction.py
```

## Files Description

| File | Description |
|------|-------------|
| `customer_prediction.ipynb` | Complete interactive analysis notebook |
| `customer_prediction.py` | Reproducible Python script |
| `requirements.txt` | All dependencies |
| `data/raw/ecommerce_data.csv` | Original Kaggle dataset |
| `data/processed/features_engineered.csv` | Processed features |
| `models/best_model.pkl` | Trained RandomForest model |

## Future Improvements
- XGBoost and LightGBM model comparison
- Deep learning neural network approach
- Feature interaction analysis
- Ensemble methods stacking
- Real-time prediction API

## Author
Nawab Singh Rawat | Data Science & Analytics

## References
- Kaggle: E-commerce Dataset
- Scikit-learn Documentation
- SMOTE: Synthetic Minority Over-sampling Technique
