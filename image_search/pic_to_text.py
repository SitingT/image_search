import io
import json

import vertexai
from vertexai.preview.vision_models import Image, ImageTextModel
from google.cloud import storage

PROJECT_ID = "imagesearch-450821"
BUCKET_NAME = "image_search_pic"
LOCATION = "us-central1"  # or the region where your Vertex AI is enabled


def main():
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Create a GCS client and get the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # List all objects in the bucket
    blobs = list(bucket.list_blobs(prefix=""))

    # Load the Vertex AI Image-to-text model
    model = ImageTextModel.from_pretrained("imagetext@001")

    with open("image_captions.json", "w", encoding="utf-8") as out_file:
        for blob in blobs:
            if blob.content_type and blob.content_type.startswith("image/"):
                print(f"Processing: {blob.name}")

                # Download the blob as bytes
                image_data = blob.download_as_bytes()

                # Use image_bytes to construct the Image object
                source_img = Image(image_bytes=image_data)

                # Generate captions
                captions = model.get_captions(
                    image=source_img,
                    language="en",
                    number_of_results=2,
                )
                print(f"Captions for {blob.name}: {captions}")

                # Write results to JSON lines
                out_data = {
                    "file": blob.name,
                    "captions": captions
                }
                out_file.write(json.dumps(out_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
