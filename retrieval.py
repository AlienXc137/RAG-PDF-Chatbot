from helper import get_embedding, get_opensearch_client


def keyword_search(query_text, top_k=20,indexname:str="pdf_content_index"): #default
    """
    Perform keyword search using OpenSearch.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client = get_opensearch_client("localhost", 9200)
    index_name = indexname

    try:
        # Create a keyword search query
        search_query = {
            "size": top_k,
            "query": {"match": {"content": query_text}},
            "_source": ["content", "content_type", "token_count"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


def semantic_search(query_text, top_k=20,indexname:str="pdf_content_index"):
    """
    Perform semantic search using vector embeddings.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client = get_opensearch_client("localhost", 9200)
    index_name = indexname

    try:
        # Get embedding for the query
        query_embedding = get_embedding(query_text)

        # Create a semantic search query
        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": ["content", "content_type", "token_count"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


def hybrid_search(query_text, top_k=20,indexname:str="pdf_content_index"):
    """
    Perform hybrid search using both keyword and semantic search.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client = get_opensearch_client("localhost", 9200)
    index_name = indexname

    try:
        # Get embedding for the query
        query_embedding = get_embedding(query_text)

        # Create a hybrid search query
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                        {"match": {"content": query_text}},
                    ]
                }
            },
            "_source": ["content", "content_type", "token_count"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Hybrid search error: {e}")
        # Fall back to keyword search
        try:
            fallback_query = {
                "size": top_k,
                "query": {"match": {"content": query_text}},
                "_source": ["content", "content_type", "token_count"],
            }
            response = client.search(index=index_name, body=fallback_query)
            return response["hits"]["hits"]
        except Exception as e2:
            print(f"Fallback search error: {e2}")
            return []


if __name__ == "__main__":
    from pprint import pprint

    query = "attention architecture"
    #results = keyword_search(query, top_k=10,indexname="attention_content")
    results = semantic_search(query, top_k=10,indexname="attention_content")
    #results = hybrid_search(query, top_k=10,indexname="attention_content")
    pprint(results)