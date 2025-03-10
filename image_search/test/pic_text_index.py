from absl import app
from google.cloud import aiplatform
import base64
from google.cloud import storage
from google.protobuf import struct_pb2
import typing
import json


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]   # Text embedding
    image_embedding: typing.Sequence[float]  # Image embedding


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client for the multimodal embedding model."""

    def __init__(self, project: str,
                 location: str = "us-central1",
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        client_options = {"api_endpoint": api_regional_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        self.location = location
        self.project = project

    def get_embedding(self, text: str = None, image_bytes: bytes = None):
        """Get text/image embeddings using the 'multimodalembedding@001' model."""
        if not text and not image_bytes:
            raise ValueError(
                'At least one of "text" or "image_bytes" must be specified.')

        # Build the instance for the request
        instance = struct_pb2.Struct()
        if text:
            instance.fields['text'].string_value = text
        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields['image'].struct_value
            image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )

        response = self.client.predict(endpoint=endpoint, instances=[instance])

        # Extract embeddings
        text_embedding = None
        if text:
            text_emb_value = response.predictions[0].get('textEmbedding', [])
            text_embedding = [float(v) for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0].get('imageEmbedding', [])
            image_embedding = [float(v) for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding
        )


def describe_image(image_bytes: bytes) -> str:
    """
    Placeholder function for image captioning.
    Replace with your real model or API to turn the image into descriptive text.
    """
    # For demo, we just return a hard-coded caption:
    return "A placeholder description for this image."


def main(argv):
    # Same GCP project & region you provided
    project_id = "dark-foundry-445301-f4"
    location = "us-central1"
    bucket_name = "image_search_tst"

    # 1) Initialize the embedding client
    embedding_client = EmbeddingPredictionClient(
        project=project_id, location=location
    )

    # 2) Connect to the GCS bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    files = bucket.list_blobs(prefix="")  # List objects in bucket root

    # 3) Open a JSON file for appending or create new
    with open("indexDataTextDebug.json", "a", encoding="utf-8") as f:

        # 4) Loop over all images
        for file in files:
            if "image" in file.content_type:  # Checking if file is an image
                # (a) Download the raw image
                image_bytes = file.download_as_bytes()

                # (b) Generate a text description (caption) of the image
                description = describe_image(image_bytes)
                print(f"[INFO] File: {file.name}")
                print(f"       Caption: {description}")

                # Print the description before switching it to index
                print(f"[DEBUG] Description before indexing: {description}")

                # (c) Get a text embedding for the description
                embedding_response = embedding_client.get_embedding(
                    text=description
                )

                # (d) Write JSON line
                # Using a dictionary so that each line in the file is a valid JSON record
                result_item = {
                    "id": file.name,
                    "caption": description,
                    "embedding": embedding_response.text_embedding
                }
                # Convert dict to JSON string
                f.write(json.dumps(result_item))
                f.write("\n")

    print("Indexing completed. Results are appended to indexData.json.")


if __name__ == "__main__":
    app.run(main)
