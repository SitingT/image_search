from google.cloud import storage
from google.cloud import aiplatform


def list_buckets():
    client = storage.Client()
    buckets = list(client.list_buckets())
    print("Buckets in your project:")
    for bucket in buckets:
        print(bucket.name)
    client = aiplatform.gapic.PredictionServiceClient()
    print("Vertex AI client initialized successfully")


def get_bucket_metadata(bucket_name):
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' metadata retrieved successfully.")
        print(f"Location: {bucket.location}")
        print(f"Storage Class: {bucket.storage_class}")
    except Exception as e:
        print(f"Error: {e}")


def list_objects(bucket_name):
    client = storage.Client()
    try:
        blobs = client.list_blobs(bucket_name)
        print(f"Objects in bucket '{bucket_name}':")
        for blob in blobs:
            print(blob.name)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_buckets()
    list_objects("image_search_pic")
    # list_objects("image_search_tst")
    # from google.cloud import aiplatform

    # 初始化 Vertex AI 客户端
