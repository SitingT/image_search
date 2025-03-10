import io
import json
import base64
import typing
import os

import vertexai
from vertexai.preview.vision_models import Image, ImageTextModel
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import openai  # Added for GPT-based caption refinement

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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


def refine_caption(initial_caption: str) -> str:
    """
    Uses GPT (via OpenAI API) to generate a more accurate and detailed image description.
    """
    prompt = (
        "You are an expert image describer. Improve the following image caption "
        "to be more descriptive and accurate: \n\n"
        f"Original caption: '{initial_caption}'\n\n"
        "Refined caption:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or another GPT model you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates clear and accurate image descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=60
    )
    refined_caption = response.choices[0].message.content.strip()
    return refined_caption


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

    # Initialize the embedding client.
    embedding_client = EmbeddingPredictionClient(
        project=PROJECT_ID, location=LOCATION)

    # Open a JSON lines file to store your index data.
    with open("image_index_gpt.json", "w", encoding="utf-8") as out_file:
        for blob in blobs:
            if blob.content_type and blob.content_type.startswith("image/"):
                print(f"Processing: {blob.name}")
                # Download the image bytes.
                image_data = blob.download_as_bytes()

                # Create an Image object for captioning.
                source_img = Image(image_bytes=image_data)

                # Generate initial captions.
                captions = model.get_captions(
                    image=source_img,
                    language="en",
                    number_of_results=2,
                )
                print(f"Initial captions for {blob.name}: {captions}")

                # Choose one caption (e.g. the first) and refine it using GPT.
                initial_caption = captions[0] if captions else ""
                refined_caption = ""
                if initial_caption:
                    try:
                        refined_caption = refine_caption(initial_caption)
                    except Exception as e:
                        print(f"Error refining caption for {blob.name}: {e}")
                        refined_caption = initial_caption  # Fallback to initial caption
                print(f"Refined caption for {blob.name}: {refined_caption}")

                # Generate text embeddings using the refined caption.
                if refined_caption:
                    embedding_response = embedding_client.get_embedding(
                        text=refined_caption)
                    text_embedding = embedding_response.text_embedding
                else:
                    text_embedding = None

                # Prepare the index record.
                index_record = {
                    "id": blob.name,
                    "caption": refined_caption,
                    "embedding": text_embedding
                }
                # Write the record as a JSON line.
                out_file.write(json.dumps(
                    index_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
