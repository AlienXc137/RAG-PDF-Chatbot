import gradio as gr
import os
import fitz  # PyMuPDF
from ingestion import ingest_all_content_into_opensearch
from chunking import process_images_with_caption, process_tables_with_description, create_semantic_chunks
from unstructured.partition.pdf import partition_pdf
from generation import generate_rag_response
from opensearchpy import OpenSearch

# Extract index name from PDF metadata or filename
def get_index_name_from_pdf(file_path_str):
    tmp_path = file_path_str  # it's already a string path from gr.File

    doc = fitz.open(tmp_path)
    metadata = doc.metadata
    doc.close()

    title = metadata.get("title")
    if title and title.strip():
        index_name = title.strip().lower().replace(" ", "_")
    else:
        base_name = os.path.splitext(os.path.basename(tmp_path))[0]
        index_name = base_name.lower().split('20')[0].replace(" ", "_")

    return tmp_path, index_name

# Check if index already exists
def index_exists(index_name):
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False
    )
    return client.indices.exists(index=index_name)

# Ingest PDF into OpenSearch
def ingest_pdf(file_path_str, force):
    pdf_path, index_name = get_index_name_from_pdf(file_path_str)

    if force or not index_exists(index_name):
        # 1. Raw chunks
        raw_chunks = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Figure", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy=None,
        )

        # 2. Process images
        processed_images = process_images_with_caption(raw_chunks, use_gemini=True)

        # 3. Process tables
        processed_tables = process_tables_with_description(raw_chunks, use_gemini=True)

        # 4. Re-partition text for semantic chunks
        text_chunks = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            chunking_strategy="by_title",
            max_characters=2000,
            min_chars_to_combine=500,
            chars_before_new_chunk=1500,
        )
        semantic_chunks = create_semantic_chunks(text_chunks)

        # 5. Ingest into OpenSearch
        ingest_all_content_into_opensearch(
            processed_images, processed_tables, semantic_chunks, index_name
        )

        return index_name, "✅ Ingestion completed successfully!"
    else:
        return index_name, f"⚠️ Index `{index_name}` already exists. Skipping ingestion."

# Generate RAG answer with streaming
def answer_query(query, index_name, search_method, model):
    full_response = ""
    for chunk in generate_rag_response(query, index_name, search_method, 5, model, stream=True):
        full_response += chunk
        yield full_response + "▌"
    yield full_response

# Gradio UI
with gr.Blocks(title="Local QnA RAG", theme="huggingface") as demo:
    gr.Markdown("# PDF Question Answering using RAG")
    gr.Markdown("Upload a PDF on the left, ask a question on the right, and get instant AI-powered answers!")
    gr.Markdown(
    """
    ⚠️ **Note:**

    - The **DeepSeek model** is running locally, so **response generation may take some time** depending on your system resources.
    - Additionally, **embedding generation** for PDF content during ingestion is a **computationally intensive** process, especially for large documents.

      Please be patient while the system processes your input.
    """
)

    with gr.Row():
        #Left Column: Compact PDF Ingestion + Search Settings
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### 📥 Upload PDF")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)
                force_reingest = gr.Checkbox(label="🔁 Force Re-ingest", value=False)

            ingest_btn = gr.Button("Ingest PDF", size="sm",variant="primary")

            with gr.Group():
                gr.Markdown("#### Status of embeddings in VectorDB")
                index_display = gr.Textbox(label="Index Name", interactive=False, max_lines=1)
                ingest_status = gr.Textbox(label="Status", interactive=False, max_lines=2)

            with gr.Group():
                gr.Markdown("#### RAG Search Settings")
                search_method = gr.Dropdown(
                    ["semantic", "keyword", "hybrid"],
                    value="hybrid",
                    label="Search Method"
                )
                model_choice = gr.Dropdown(
                    ["gemini-2.5-flash", "deepseek-r1:1.5b"],
                    value="gemini-2.5-flash",
                    label="Model"
                )

        # Right Column: Question and Answer ===
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("#### 💬 Ask a Question")
                query_input = gr.Textbox(lines=2, placeholder="Ask something about the document...")
            
            query_btn = gr.Button("Ask Query", variant="primary")
            
            with gr.Group():
                gr.Markdown("#### RAG Response")
                response_output = gr.Textbox(label="Answer", lines=20, interactive=False)

    #State variable to hold index name ===
    index_state = gr.State("")

    # Ingestion Logic
    def handle_ingestion(pdf_input, force_reingest):
        if pdf_input is None:
            return "", "!!! Please upload a PDF file.", ""
        index_name, status = ingest_pdf(pdf_input, force_reingest)
        return index_name, status, index_name

    ingest_btn.click(
        fn=handle_ingestion,
        inputs=[pdf_input, force_reingest],
        outputs=[index_display, ingest_status, index_state]
    )

    # Query Logic
    query_btn.click(
        fn=answer_query,
        inputs=[query_input, index_state, search_method, model_choice],
        outputs=response_output
    )

demo.launch()
