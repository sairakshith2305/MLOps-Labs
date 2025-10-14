"""
Airflow DAG to run Iris model training and evaluation pipeline.
"""
import sys
sys.path.append('/opt/airflow/src')
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from lab import run_pipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def iris_training_callable(**kwargs):
    print("[DAG] Starting Iris training pipeline...")
    metrics_path = run_pipeline(output_dir="/opt/airflow/model")
    print(f"[DAG] Pipeline complete. Metrics saved at: {metrics_path}")
    return metrics_path


with DAG(
    dag_id="iris_training_pipeline",
    default_args=default_args,
    description="Train, evaluate, and save multiple Iris models",
    schedule_interval=None,
    start_date=datetime(2025, 10, 1),
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_and_evaluate_models",
        python_callable=iris_training_callable,
        provide_context=True,
    )

    train_task
