from helper import get_embedding

def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.
    """
    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists")
        client.indices.delete(index=index_name)

    # Define correct mapping using knn_vector
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "embedding": {"type": "knn_vector", "dimension": 768}
            }
        },
        "settings": {
            "index": {
                "knn": True
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def prepare_chunks_for_ingestion(chunks):
    """
    Prepare chunks for ingestion by adding embeddings.
    """
    prepared_chunks = []

    for idx, chunk in enumerate(chunks):
        if not chunk.get("content"):
            print(f"Skipping Chunk {idx} due to missing content")
            continue

        try:
            # Generate embedding
            embedding = get_embedding(chunk["content"])
            if len(embedding) != 768:
                raise ValueError(f"Invalid embedding dimension: {len(embedding)}")

            chunk_data = {
                "content": chunk.get("content", ""),
                "content_type": chunk.get("content_type", "text"),
                "filename": chunk.get("filename", None),
                "embedding": embedding
            }

            prepared_chunks.append(chunk_data)

        except Exception as e:
            print(f"Error in chunk {idx}: {e}")
            continue

    return prepared_chunks


def ingest_chunks_into_opensearch(client, index_name, chunks):
    """
    Ingest prepared chunks into the specified OpenSearch index.
    """
    from opensearchpy import helpers

    actions = []
    for chunk in chunks:
        action = {
            "_index": index_name,
            "_source": chunk
        }
        actions.append(action)

    try:
        helpers.bulk(client, actions)
        print(f"Ingested {len(actions)} chunks into index '{index_name}'.")
    except Exception as e:
        print(f"Error ingesting chunks into index '{index_name}': {e}")
        raise


def ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name):
    """
    Ingest all content into OpenSearch.
    """
    from helper import get_opensearch_client

    # Create OpenSearch client
    client = get_opensearch_client("localhost", 9200)

    # Create index
    create_index_if_not_exists(client, index_name)

    # Prepare and ingest images
    image_chunks = prepare_chunks_for_ingestion(processed_images)
    ingest_chunks_into_opensearch(client, index_name, image_chunks)

    # Prepare and ingest tables
    table_chunks = prepare_chunks_for_ingestion(processed_tables)
    ingest_chunks_into_opensearch(client, index_name, table_chunks)

    # Prepare and ingest semantic chunks
    semantic_chunks_data = prepare_chunks_for_ingestion(semantic_chunks)
    ingest_chunks_into_opensearch(client, index_name, semantic_chunks_data)


if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    from chunking import process_images_with_caption, process_tables_with_description, create_semantic_chunks

    pdf_file_path = "files/rag survey.pdf"

    # 1. Extract raw chunks
    raw_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )

    # 2. Process images
    processed_images = process_images_with_caption(raw_chunks, use_gemini=True)

    # 3. Process tables
    processed_tables = process_tables_with_description(raw_chunks, use_gemini=True)

    # 4. Re-partition for semantic chunks
    text_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        min_chars_to_combine=500,
        chars_before_new_chunk=1500,
    )

    # 5. Semantic text chunks
    semantic_chunks = create_semantic_chunks(text_chunks)

    # 6. Ingest all to OpenSearch
    index_name = "pdf_content_index"
    ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name)
