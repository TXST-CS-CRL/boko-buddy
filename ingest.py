import os
import sys
from pathlib import Path

import pandas as pd

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import PptxReader, PDFReader, MarkdownReader 
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter

from llama_index.core import Settings

# Set the global ceiling to your maximum expected chunk size
Settings.chunk_size = 8192 
Settings.chunk_overlap = 100


def material_handler(data_dir, calendar_map, material_type):
    all_nodes = []

    # readers for lecture slides, handouts and textbook, and tutorials
    pptx_reader = PptxReader()
    pdf_reader = PDFReader()
    md_reader = MarkdownReader()
    
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Determine which reader to use based on extension
        if filename.endswith(".pptx"):
            reader = pptx_reader
        elif filename.endswith(".pdf"):
            reader = pdf_reader
        elif filename.endswith(".md"):
            reader = md_reader
        else:
            reader = SimpleDirectoryReader(input_files=[file_path])
            
        if filename in calendar_map:
            meta = calendar_map[filename]
            if material_type == "lectures":
                extra_info = {
                    "source": meta.get('source'), 
                    "date": str(meta.get('date')),
                    "topic": meta.get('topic'),
                    "lecture_id": meta.get('lecture_id'), 
                    "file_type": "PowerPoint" if filename.endswith(".pptx") else "PDF",
                    "priority": 2
                }
            elif material_type == "labs":
                extra_info = {
                    "source": meta.get('source'), 
                    "date": str(meta.get('date')),
                    "topic": meta.get('topic'),
                    "lab_id": meta.get('lab_id'), 
                    "file_type": "PowerPoint" if filename.endswith(".pptx") else "PDF"
                }
            elif material_type == "tutorials":
                extra_info = {
                    "source": meta.get('source'), 
                    "date": str(meta.get('date')),
                    "topic": meta.get('topic'),
                    "tutorial_id": meta.get('tutorial_id'), 
                    "file_type": "Markdown" if filename.endswith(".md") else "Text"
                }
            elif material_type == "code":
                extra_info = {
                    "source": meta.get('source'), 
                    "date": str(meta.get('date')),
                    "topic": meta.get('topic'),
                    "lecture_id": meta.get('lecture_id'), 
                    "file_type": "C++" if filename.endswith(".cpp") else "Text"
                }
            elif material_type == "textbook":
                raw_topics = meta.get('topic', '')
                topic_list = [t.strip() for t in str(raw_topics).split(',')]
                extra_info = {
                    "source": meta.get('source'), 
                    "topic": topic_list,
                    "file_type": "PDF",
                    "priority": 1
                }
            else:
                continue
            
            print(f"Ingesting {filename} ...")
            documents = reader.load_data(file_path)

            # 1. "Personalities"
            textbook_parser = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
            tutorial_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            for doc in documents:
                doc.metadata = {}
                doc.metadata = extra_info
                doc.metadata["source_file"] = filename 
                doc.metadata["page_label"] = doc.metadata.get("page_label", "N/A")
                doc.metadata.update(extra_info)
                print(f"TEXT LENGTH: { (int) (len(doc.text) / 4) } tokens")
                
                doc.excluded_llm_metadata_keys = [] # Let the LLM see everything else (date, topic)
                doc.excluded_embed_metadata_keys = ["file_name", "source_file"] # Don't waste vector space on filename

                doc.metadata_template = "{key}: {value}"
                doc.text_template = "Metadata: {metadata_str}\n\nContent: {content}"
    
                if material_type == "lectures":
                    nodes = [doc]
                elif material_type == "tutorials":
                    nodes = tutorial_parser.get_nodes_from_documents([doc])
                elif material_type == "textbook":
                    if filename == "chapter_03.pdf":
                        high_chunk_textbook_parser = TokenTextSplitter(chunk_size=8192, chunk_overlap=100)
                        nodes = high_chunk_textbook_parser.get_nodes_from_documents([doc])
                    else:
                        nodes = textbook_parser.get_nodes_from_documents([doc])
                else:
                    nodes = tutorial_parser.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
                # if len(nodes) > 0 and filename == "chapter_03.pdf":
                #     sample_node = nodes[0]
                #     print(f"\n--- DEBUG: Verifying {filename} ---")
                #     # This shows what the LLM actually receives
                #     print("LLM VIEW:\n", sample_node.get_content(metadata_mode="llm"))
                #     print("-" * 30)
                #     # --------------------
    return all_nodes
            
def build_course_index(data_dir="./materials", calendar_file="master_calendar.csv"):

    calendar_path = os.path.join(data_dir, calendar_file)  
    calendar_df = pd.read_csv(calendar_path)
    calendar_df.columns = calendar_df.columns.str.strip()
    calendar_map = calendar_df.set_index('primary_file').to_dict('index')

    all_nodes = []

    material_types = ["lectures", "labs", "tutorials", "code", "textbook"]

    for material in material_types: 
        print (f"Ingesting {material} ...") 
        this_data_dir = os.path.join(data_dir, material)
        nodes = material_handler(this_data_dir, calendar_map, material)
        all_nodes.extend(nodes) 
        
    index = VectorStoreIndex.from_documents(all_nodes)
    index.storage_context.persist(persist_dir="./storage")
    return index



if len(sys.argv) < 2:
    print(f"No path to course materials provided")
    print(f"Usage: ")
    print(f"\t python ingest.py [course_material_folder]\n")
else: 
    path = Path(sys.argv[1])
    if path.exists() and path.is_dir():
        build_course_index(path)
    else:
        print(f"Course material folder not found. Please try again")

