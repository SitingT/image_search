#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from absl import app
from google.cloud import aiplatform
import typing
from google.protobuf import struct_pb2


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
    def __init__(self, project: str, location: str = "us-central1",
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        from google.cloud import aiplatform as vertex_ai
        client_options = {"api_endpoint": api_regional_endpoint}
        self.client = vertex_ai.gapic.PredictionServiceClient(
            client_options=client_options)
        self.project = project
        self.location = location

    def get_embedding(self, text: str = None) -> "EmbeddingResponse":
        if not text:
            raise ValueError('请提供有效的查询文本。')

        instance = struct_pb2.Struct()
        instance.fields['text'].string_value = text

        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )

        response = self.client.predict(endpoint=endpoint, instances=[instance])
        print("[DEBUG] Model Response:", response)  # Debugging print

        if not response or not response.predictions:
            print("[ERROR] 模型未返回任何结果，请检查部署或网络连接。")
            return EmbeddingResponse(text_embedding=[], image_embedding=[])

        text_emb_value = response.predictions[0].get('textEmbedding', [])
        text_embedding = [float(v)
                          for v in text_emb_value] if text_emb_value else []

        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=[]
        )


def main(argv):
    PROJECT_ID = "dark-foundry-445301-f4"
    LOCATION = "us-central1"
    INDEX_ENDPOINT_ID = "1391361596206350336"
    DEPLOYED_INDEX_ID = "test_1736501085412"

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        endpoint_resource_name)

    embedding_client = EmbeddingPredictionClient(
        project=PROJECT_ID,
        location=LOCATION
    )

    user_query = input("请输入要搜索的文字描述：")
    embedding_response = embedding_client.get_embedding(text=user_query)
    query_vector = embedding_response.text_embedding
    if not query_vector:
        print("[ERROR] 无法获取文本向量，请检查模型或输入。")
        return

    neighbor_count = 5  # 返回 Top-5
    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=neighbor_count
    )

    print("[DEBUG] Index Search Response:", response)  # Debugging print

    if not response or not isinstance(response, list) or len(response) == 0:
        print("[ERROR] 索引端点未返回任何结果，请确认 IndexEndpoint 是否正常。")
        return

    neighbors = response[0] if isinstance(response[0], list) else []

    print(f"\n=== 与“{user_query}”最相似的 Top {neighbor_count} 项 ===")
    for neighbor in neighbors:
        print(f"ID: {neighbor.id}, distance: {neighbor.distance}")

    print("\n检索结束！")


if __name__ == "__main__":
    app.run(main)
