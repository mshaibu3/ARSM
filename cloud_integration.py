import boto3

def upload_to_s3(file_path, bucket_name, object_name):
    """
    Upload a file to an S3 bucket.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
    except Exception as e:
        print(f"Error uploading to S3: {e}")

def download_from_s3(bucket_name, object_name, file_path):
    """
    Download a file from an S3 bucket.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, object_name, file_path)
    except Exception as e:
        print(f"Error downloading from S3: {e}")
