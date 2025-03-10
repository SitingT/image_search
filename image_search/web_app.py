import streamlit as st
from indexing import *
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

# 主题
st.title("Performing Semantic Searches for Images with Vertex AI")
st.subheader(
    "Powered by Vertex AI Multimodal Embeddings and Vertex AI Matching Engine"
)

# Initialize Client to Access Multimodal-Embeddings Model
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
    "api_endpoint": "89982829.us-central1-788343963324.vdb.vertexai.goog"
}

# Client to Access GCS Bucket
try:
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("image_search_pic")
    st.write("✅ GCS client initialized successfully.")
except Exception as e:
    st.write(f"❌ Error initializing GCS client: {e}")

# Vertex AI Client for Similarity Matching
try:
    vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
        credentials=credentials,
        client_options=client_options,
    )
    st.write("✅ Vertex AI client initialized successfully.")
except Exception as e:
    st.write(f"❌ Error initializing Vertex AI client: {e}")

# Initialize Request Object
request = aiplatform_v1beta1.FindNeighborsRequest(
    index_endpoint="projects/788343963324/locations/us-central1/indexEndpoints/4583041541500567552",
    deployed_index_id="img_direct_search_1740184720365"
)

# List to Store Image Results
allResults = []

# Search Input
search_term = 'a picture of ' + st.text_input('Search: ')

if search_term != "a picture of ":
    try:
        converted_query_to_embedding = client.get_embedding(text=search_term)
        st.write("✅ Query converted to embedding.")
    except Exception as e:
        st.write(f"❌ Error converting query to embedding: {e}")

    dp1 = aiplatform_v1beta1.IndexDatapoint(
        datapoint_id="0",
        feature_vector=converted_query_to_embedding[0]
    )

    # Create Query and Append to Request
    query = aiplatform_v1beta1.FindNeighborsRequest.Query(
        datapoint=dp1,
    )
    request.queries.append(query)

    # Perform Similarity Search
    try:
        response = vertex_ai_client.find_neighbors(request)
        st.write("✅ Similarity search performed successfully.")
    except Exception as e:
        st.write(f"❌ Error performing similarity search: {e}")

    # Debugging: Log the number of neighbors found
    st.write(f"🔍 Found {len(response.nearest_neighbors)} nearest neighbors.")

    for r in response.nearest_neighbors:
        for n in r.neighbors:
            id = n.datapoint.datapoint_id
            st.write(f"🆔 Neighbor ID: {id}")

            # Extract Path and Debug the Path
            try:
                path = id.strip("b'").strip("'")
                st.write(f"📁 Blob path: {path}")
            except IndexError as e:
                st.write(f"❌ Error parsing blob path from ID: {e}")
                continue

            # Check Distance Threshold
            distance = n.distance
            st.write(f"📏 Distance: {distance}")

            if distance >= 0.5:
                try:
                    blob = bucket.blob(path)
                    st.write(f"📝 Blob content type: {blob.content_type}")

                    # Download Image Data
                    image_data = blob.download_as_bytes()
                    st.write(f"✅ Downloaded {len(image_data)} bytes.")

                    # Check If Image Can Be Opened
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        # allResults.append(img)
                        allResults.append(
                            {"distance": distance, "image": img, "path": path})
                    except UnidentifiedImageError:
                        st.write(f"❌ Unable to identify image at path: {path}")
                except Exception as e:
                    st.write(f"❌ Error downloading blob {path}: {e}")

# Display Results
if len(allResults) > 1:
    sortedResults = sorted(
        allResults, key=lambda result: result["distance"], reverse=True)
    st.write("### These are the most relevant results matching your search query:")
    # st.image(allResults, width=200)
    for result in sortedResults:
        st.image(result["image"])
elif search_term == "a picture of ":
    st.write("Please type in a search query above.")
else:
    st.write("Sorry! There are no images matching your query. Please try again.")
