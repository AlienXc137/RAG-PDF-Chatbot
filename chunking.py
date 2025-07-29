import base64
import os
import google.generativeai as genai
from dotenv import load_dotenv
from unstructured.documents.elements import Element,Text,Image,FigureCaption,Table,CompositeElement

load_dotenv()

#processing images
def process_images_with_caption(raw_chunks,use_gemini=True):
    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)
    
    # Extract images and their captions from the raw chunks
    processed_images = []
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):
            # check idx + 1 is figure caption
            if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
                #Checks if the next chunk (idx + 1) exists and is a FigureCaption
                caption = raw_chunks[idx + 1].text
            else:
                caption = None

            image_data=({
                "caption": caption if caption else "No caption",
                "image_text": chunk.text,
                "base64_image": chunk.metadata.image_base64,
                "content": chunk.text, #if gemini model doesnt run this will be saved
                "content_type":"image",
                "filename": chunk.metadata.filename
            })

            if use_gemini:
                model = genai.GenerativeModel("gemini-2.5-flash") 

                image_binary = base64.b64decode(image_data["base64_image"])

                prompt = (
                    f"Describe the image in detail. The caption is: {image_data['caption']}."
                    f"The image text is: {image_data['image_text']}" 
                    f"Directly analyze the image and provide a detailed description without any additional text."
                )

                response = model.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": image_binary},
                ])
                image_data["content"]=response.text
            
            processed_images.append(image_data)

    return processed_images

def process_tables_with_description(raw_chunks,use_gemini=True):

    # Configure Gemini API
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        genai.configure(api_key=api_key)

    # Extract tables from the raw chunks
    processed_tables = []
    for idx, element in enumerate(raw_chunks):
        if isinstance(element, Table):
            table_data=({
                "table_as_html": element.metadata.text_as_html,
                "table_text": element.text,
                "content": element.text,  # Fallback content
                "content_type": "table",
                "filename": element.metadata.filename
            })

            if use_gemini:
                model = genai.GenerativeModel("gemini-2.5-flash") 

                prompt = (
                    "Analyze the following table and provide a detailed description of its contents, "
                    "including the structure, key data points, and any notable trends or insights."
                    f"Here is the table in HTML format: {table_data['table_as_html']}"
                    "Directly analyze the table and provide a detailed description without any additional text."
                )

                response = model.generate_content([prompt])
                table_data["content"]=response.text
            
            processed_tables.append(table_data)

    return processed_tables

def create_semantic_chunks(text_chunks):
    process_chunks=[]
    for idx, chunk in enumerate(text_chunks):
        if isinstance(chunk,CompositeElement):
            chunk_data={
                "content": chunk.text,
                "content_type": "text",
                "filename": chunk.metadata.filename
            }
            process_chunks.append(chunk_data)
        
    return process_chunks


#chunking.py

if __name__=="__main__":
    from unstructured.partition.pdf import partition_pdf

    pdf_file_path="files/rag survey.pdf"
    raw_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True, #includes all the tables present in pdf
        extract_image_block_types=["Image", "Figure", "Table"], 
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )

    # processed_images=process_images_with_caption(raw_chunks,use_gemini=True)

    # for image in processed_images:
    #     print(image) 
    
    # process_tables=process_tables_with_description(raw_chunks,use_gemini=True)
    # for table in process_tables:
    #     print(table)

    text_chunks=partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_character=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500
    )

    semantic_chunks=create_semantic_chunks(text_chunks)
    for chunk in semantic_chunks:
        print(chunk)