import pickle
from datetime import datetime, timedelta

import mlflow as mlflow
import pandas as pd
from prefect.flow_runners import SubprocessFlowRunner

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    logger = get_run_logger()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    x_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger = get_run_logger()
    logger.info(f"The shape of x_train is {x_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    x_val = dv.transform(val_dicts)
    y_pred = lr.predict(x_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)

    logger = get_run_logger()
    logger.info(f"The MSE of validation is: {mse}")


@task
def get_paths(date):
    if date is None:
        date = datetime.now()
    else:
        date = datetime.strptime(date, "%Y-%m-%d")

    train_date = (date - timedelta(days=62)).strftime('%Y-%m-%d')
    val_date = (date - timedelta(days=31)).strftime('%Y-%m-%d')
    train_path = f'./data/fhv_tripdata_{train_date[:7]}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date[:7]}.parquet'

    logger = get_run_logger()
    logger.info(f"Train data path: {train_path}")
    logger.info(f"Val data path: {val_path}")

    return train_path, val_path


# @flow(task_runner=SequentialTaskRunner())
@flow
def main(date=None):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("homework-experiment")

    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)
    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f"models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open(f"models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    with mlflow.start_run():
        mlflow.log_artifact(f"models/model-{date}.bin", artifact_path="models")
        mlflow.log_artifact(f"models/dv-{date}.b", artifact_path="models")

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name="cron-schedule-homework",
    flow=main,
    parameters={"date": "2021-08-15"},
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York")
)
