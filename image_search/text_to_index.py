import io
import json
import base64
import typing

import vertexai
from vertexai.preview.vision_models import Image, ImageTextModel
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2

# Define a response type to hold the embeddings.


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]  # for text embeddings
    image_embedding: typing.Sequence[float]  # for image embeddings (if needed)

# This client wraps the Vertex AI Prediction service for embeddings.


class EmbeddingPredictionClient:
    def __init__(self, project: str,
                 location: str = "us-central1",
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        client_options = {"api_endpoint": api_regional_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options)
        self.location = location
        self.project = project

    def get_embedding(self, text: str = None, image_bytes: bytes = None):
        if not text and not image_bytes:
            raise ValueError(
                'At least one of text or image_bytes must be specified.')
        instance = struct_pb2.Struct()
        if text:
            instance.fields['text'].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields['image'].struct_value
            image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

        instances = [instance]
        endpoint = (f"projects/{self.project}/locations/{self.location}"
                    "/publishers/google/models/multimodalembedding@001")
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]['textEmbedding']
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]['imageEmbedding']
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding
        )


def main():
    PROJECT_ID = "imagesearch-450821"
    BUCKET_NAME = "image_search_pic"
    LOCATION = "us-central1"  # Change if needed

    # Initialize Vertex AI for the image-to-text model.
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Initialize a GCS client and retrieve the bucket.
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=""))

    # Load the image captioning model.
    model = ImageTextModel.from_pretrained("imagetext@001")

    # Initialize the embedding client (from Code 1).
    embedding_client = EmbeddingPredictionClient(
        project=PROJECT_ID, location=LOCATION)

    # Open a JSON lines file to store your index data.
    with open("image_index.json", "w", encoding="utf-8") as out_file:
        for blob in blobs:
            if blob.content_type and blob.content_type.startswith("image/"):
                print(f"Processing: {blob.name}")
                # Download the image bytes.
                image_data = blob.download_as_bytes()

                # Create an Image object for captioning.
                source_img = Image(image_bytes=image_data)

                # Generate captions.
                captions = model.get_captions(
                    image=source_img,
                    language="en",
                    number_of_results=2,
                )
                print(f"Captions for {blob.name}: {captions}")

                # Choose one caption (e.g. the first) to generate its text embedding.
                caption_text = captions[0] if captions else ""
                if caption_text:
                    embedding_response = embedding_client.get_embedding(
                        text=caption_text)
                    text_embedding = embedding_response.text_embedding
                else:
                    text_embedding = None

                # Prepare the index record.
                index_record = {
                    "id": blob.name,
                    "caption": caption_text,
                    "embedding": text_embedding
                }
                # Write the record as a JSON line.
                out_file.write(json.dumps(
                    index_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
