import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic customer churn data"""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'num_products': np.random.randint(1, 5, n_samples),
        'has_support_tickets': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'contract_type': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        'payment_delay_days': np.random.randint(0, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    churn_probability = (
        0.1 +
        (df['payment_delay_days'] > 10) * 0.3 +
        (df['tenure_months'] < 12) * 0.2 +
        (df['has_support_tickets'] == 1) * 0.15 +
        (df['contract_type'] == 0) * 0.25
    )
    
    df['churned'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    logger.info("Starting data preprocessing...")
    
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    churn_count = y.sum()
    no_churn_count = len(y) - churn_count
    logger.info(f"Class distribution - No Churn: {no_churn_count}, Churned: {churn_count}")
    
    return X, y

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the model"""
    logger.info("Training Random Forest model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    logger.info(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores]}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
    

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = model.score(X_test_scaled, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL PERFORMANCE METRICS")
    logger.info(f"{'='*60}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(test_accuracy),
        'roc_auc': float(roc_auc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    }


def main():
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    logger.info("Starting Customer Churn Prediction Pipeline...")
    
    logger.info("Generating synthetic customer data...")
    df = generate_synthetic_data(n_samples=2000)
    logger.info(f"Generated {len(df)} customer records")
    
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()