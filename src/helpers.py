from boto3 import client
import os
import pickle


def download_model(
    model_s3_path: str,
    model_local_path: str
):
    s3 = client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    s3.download_file(os.getenv('BUCKET'), model_s3_path, model_local_path)
    with open(model_local_path, "rb") as f:
        return pickle.load(f)


def upload_model(
    model_local_path: str,
    model_s3_path: str
):
    s3 = client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    s3.upload_file(model_local_path, os.getenv('BUCKET'), model_s3_path)
