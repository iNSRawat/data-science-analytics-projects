import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CustomerBehaviorPredictor:
    def __init__(self, data_path='data/raw/customer_data.csv'):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load customer data"""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        return self.df
    
    def exploratory_analysis(self):
        """Perform EDA"""
        print("\n=== Exploratory Data Analysis ===")
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nTarget Distribution:\n{self.df['Purchase'].value_counts()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
    
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n=== Data Preprocessing ===")
        
        # Separate features and target
        X = self.df.drop('Purchase', axis=1)
        y = self.df['Purchase']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Processed features shape: {X.shape}")
        print(f"Target distribution - Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        
        return X, y, scaler, le_dict
    
    def train_model(self, X, y):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n=== Model Training ===")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X_train_smote, y_train_smote)
        
        self.model = grid_search.best_estimator_
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:\n{confusion_matrix(self.y_test, y_pred)}")
        print(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}
    
    def feature_importance(self):
        """Get feature importance"""
        print("\n=== Feature Importance ===")
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10))
        return feature_importance
    
    def predict(self, new_data):
        """Make predictions on new data"""
        predictions = self.model.predict(new_data)
        probabilities = self.model.predict_proba(new_data)
        return predictions, probabilities


if __name__ == '__main__':
    # Initialize predictor
    predictor = CustomerBehaviorPredictor()
    
    # Load data
    predictor.load_data()
    
    # EDA
    predictor.exploratory_analysis()
    
    # Preprocess
    X, y, scaler, le_dict = predictor.preprocess_data()
    
    # Train model
    predictor.train_model(X, y)
    
    # Evaluate
    metrics = predictor.evaluate_model()
    
    # Feature importance
    importance_df = predictor.feature_importance()
    
    print("\n=== Prediction Complete ===")
