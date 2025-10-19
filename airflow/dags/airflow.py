"""
Airflow DAG to run Iris model training and evaluation of pipeline.
"""
import sys
sys.path.append('/opt/airflow/src')
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import joblib
import os
from lab import load_data, preprocess_data, train_models, evaluate_models, save_artifacts

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

TEMP_DIR = "/opt/airflow/temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def load_data_callable(**kwargs):
    X, y, feature_names, target_names = load_data()
    joblib.dump(X, os.path.join(TEMP_DIR, 'X.joblib'))
    joblib.dump(y, os.path.join(TEMP_DIR, 'y.joblib'))
    ti = kwargs['ti']
    ti.xcom_push(key='feature_names', value=feature_names)
    ti.xcom_push(key='target_names', value=target_names.tolist())
    print("[DAG] Dataset loaded and saved to temp directory")


def preprocess_data_callable(**kwargs):
    X = joblib.load(os.path.join(TEMP_DIR, 'X.joblib'))
    y = joblib.load(os.path.join(TEMP_DIR, 'y.joblib'))
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    joblib.dump(X_train, os.path.join(TEMP_DIR, 'X_train.joblib'))
    joblib.dump(X_test, os.path.join(TEMP_DIR, 'X_test.joblib'))
    joblib.dump(y_train, os.path.join(TEMP_DIR, 'y_train.joblib'))
    joblib.dump(y_test, os.path.join(TEMP_DIR, 'y_test.joblib'))
    joblib.dump(scaler, os.path.join(TEMP_DIR, 'scaler.joblib'))    
    print("[DAG] Data preprocessed and saved to temp directory")

def train_models_callable(**kwargs):
    X_train = joblib.load(os.path.join(TEMP_DIR, 'X_train.joblib'))
    y_train = joblib.load(os.path.join(TEMP_DIR, 'y_train.joblib'))
    trained_models = train_models(X_train, y_train)
    joblib.dump(trained_models, os.path.join(TEMP_DIR, 'trained_models.joblib'))
    print("[DAG] Models trained and saved to temp directory")


def evaluate_models_callable(**kwargs):
    X_test = joblib.load(os.path.join(TEMP_DIR, 'X_test.joblib'))
    y_test = joblib.load(os.path.join(TEMP_DIR, 'y_test.joblib'))
    trained_models = joblib.load(os.path.join(TEMP_DIR, 'trained_models.joblib'))
    
    metrics = evaluate_models(trained_models, X_test, y_test)
    joblib.dump(metrics, os.path.join(TEMP_DIR, 'metrics.joblib'))
    ti = kwargs['ti']
    ti.xcom_push(key='metrics', value=metrics)
    print("[DAG] Models evaluated and metrics saved")

def save_artifacts_callable(**kwargs):
    trained_models = joblib.load(os.path.join(TEMP_DIR, 'trained_models.joblib'))
    metrics = joblib.load(os.path.join(TEMP_DIR, 'metrics.joblib'))
    scaler = joblib.load(os.path.join(TEMP_DIR, 'scaler.joblib'))
    
    metrics_path = save_artifacts(trained_models, metrics, scaler, output_dir="/opt/airflow/model")
    
    print(f"[DAG] Pipeline complete. Metrics saved at: {metrics_path}")
    
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    return metrics_path


with DAG(
    dag_id="iris_training_pipeline",
    default_args=default_args,
    description="Train, evaluate, and save multiple Iris models",
    schedule_interval=None,
    start_date=datetime(2025, 10, 1),
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data_callable,
        provide_context=True,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data_callable,
        provide_context=True,
    )

    train_task = PythonOperator(
        task_id="train_models",
        python_callable=train_models_callable,
        provide_context=True,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models_callable,
        provide_context=True,
    )

    save_task = PythonOperator(
        task_id="save_artifacts",
        python_callable=save_artifacts_callable,
        provide_context=True,
    )

    # Define task dependencies
    load_data_task >> preprocess_task >> train_task >> evaluate_task >> save_task