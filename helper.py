import requests
from opensearchpy import OpenSearch

def get_embedding(prompt, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings/"
    data = {"prompt": prompt, "model": model}

    response = requests.post(url, json=data)
    response.raise_for_status()

    return response.json().get("embedding",None)

def get_opensearch_client(host,port):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    if client.ping():
        print("Connected to OpenSearch")

    return client


if __name__ == "__main__":
    get_opensearch_client("localhost",9200)