import streamlit as st
# Assuming your client is defined here
from indexing import EmbeddingPredictionClient
import json
from google.oauth2 import service_account
from google.cloud import aiplatform_v1beta1
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
import io

# Streamlit Page Configuration
st.set_page_config(
    layout="wide",
    page_title="JS Lab",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"
)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "A place for me to experiment with different LLM use cases, models, application frameworks, etc."
)

# Main Page Title
st.title("Performing Semantic Searches for Images with Vertex AI")
st.subheader(
    "Powered by Vertex AI Multimodal Embeddings and Vertex AI Matching Engine")

# Initialize the Embedding Client (to convert text queries to embeddings)
client = EmbeddingPredictionClient(project="imagesearch-450821")

# Service Account Credentials
scopes = ["https://www.googleapis.com/auth/cloud-platform"]
sa_file_path = "/Users/sitingtang/Desktop/keystore/imagesearch-450821-1d4044714a84.json"

try:
    credentials = service_account.Credentials.from_service_account_file(
        sa_file_path, scopes=scopes
    )
    st.write("✅ Service account credentials loaded successfully.")
except Exception as e:
    st.write(f"❌ Error loading credentials: {e}")

client_options = {
    "api_endpoint": "683845784.us-central1-788343963324.vdb.vertexai.goog"
}

# Initialize GCS Client and Access the Bucket
try:
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("image_search_pic")
    st.write("✅ GCS client initialized successfully.")
except Exception as e:
    st.write(f"❌ Error initializing GCS client: {e}")

# Initialize the Vertex AI Matching Engine Client
try:
    vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
        credentials=credentials,
        client_options=client_options,
    )
    st.write("✅ Vertex AI client initialized successfully.")
except Exception as e:
    st.write(f"❌ Error initializing Vertex AI client: {e}")

# Build the FindNeighborsRequest for vector search
request = aiplatform_v1beta1.FindNeighborsRequest(
    index_endpoint="projects/788343963324/locations/us-central1/indexEndpoints/8289961431663640576",
    deployed_index_id="img_search_1739525334428"
)

# List to Store the Retrieved Image Results along with their distances
# Each entry is a tuple: (distance, image)
results = []

# Search Input from User
search_term = ' ' + st.text_input('Search:')

if search_term != " ":
    try:
        # Convert the search query to an embedding vector
        converted_query = client.get_embedding(text=search_term)
        st.write("✅ Query converted to embedding.")
    except Exception as e:
        st.write(f"❌ Error converting query to embedding: {e}")

    # Create an IndexDatapoint using the generated text embedding.
    dp1 = aiplatform_v1beta1.IndexDatapoint(
        datapoint_id="0",  # Arbitrary id for the query point
        feature_vector=converted_query.text_embedding
    )

    # Append the query to the request
    query = aiplatform_v1beta1.FindNeighborsRequest.Query(datapoint=dp1)
    request.queries.append(query)

    # Perform the Similarity Search on the deployed index
    try:
        response = vertex_ai_client.find_neighbors(request)
        st.write("✅ Similarity search performed successfully.")
    except Exception as e:
        st.write(f"❌ Error performing similarity search: {e}")

    # Debug: Log the number of nearest neighbors found
    st.write(f"🔍 Found {len(response.nearest_neighbors)} nearest neighbors.")

    for r in response.nearest_neighbors:
        for n in r.neighbors:
            neighbor_id = n.datapoint.datapoint_id
            st.write(f"🆔 Neighbor ID: {neighbor_id}")

            # Use the neighbor id directly as the blob path.
            path = neighbor_id
            st.write(f"📁 Blob path: {path}")

            # Display the distance (similarity measure)
            distance = n.distance
            st.write(f"📏 Distance: {distance}")

            # (Optionally, set a threshold on distance if desired)
            if distance >= 0.5:
                try:
                    blob = bucket.blob(path)
                    st.write(f"📝 Blob content type: {blob.content_type}")

                    # Download the image bytes from GCS
                    image_data = blob.download_as_bytes()
                    st.write(f"✅ Downloaded {len(image_data)} bytes.")

                    # Open the image to verify it can be processed
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        # Save both the image and its distance for later sorting.
                        results.append((distance, img))
                    except UnidentifiedImageError:
                        st.write(f"❌ Unable to identify image at path: {path}")
                except Exception as e:
                    st.write(f"❌ Error downloading blob {path}: {e}")

    # Sort the results so that images with lower distance (more similar) come first
    if results:
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        # Extract only the images for display
        sorted_images = [img for distance, img in sorted_results]
        st.write(
            "### These are the most relevant results matching your search query:")
        st.image(sorted_images, width=200)
    else:
        st.write("Sorry! There are no images matching your query. Please try again.")
elif search_term == "a picture of ":
    st.write("Please type in a search query above.")
