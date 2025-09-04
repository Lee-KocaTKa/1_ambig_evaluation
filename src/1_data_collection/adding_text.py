import argparse  
import json 
import os 
import openai 
import getpass 
import faiss 
from pathlib import Path 
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore  

from const import (
    LLM_MODEL_CARD,
    EMBEDDING_MODEL_CARD,
    FILE_NAME,  
    AMBIGUITY_TYPE,
    SENTENCE_INDEX_TABLE,
    STARTING_PROMPT
)

# Text Generation & Text Embedding Models 
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])        ### set your OpenAI API key as an environment variable
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_CARD)   

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--trials', type=int, default=1000, help='API call count limits')
    parser.add_argument('--ambiguity-type', type=str, help='Type of Ambiguity') 
    parser.add_argument('--annotation-root', type=Path, default='../../data/vidaco_jsons') 
    parser.add_argument('--embedding-root', type=Path, default='../../data/embeddings')  
    parser.add_argument('--is-create-embeddings', action='store_true')

def create_existing_embeddings(texts: list[str], output_root: Path): 
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("dummy text"))) 
    
    vector_store = FAISS(
        embedding_function=embedding_model, 
        index=index, 
        docstore=InMemoryDocstore(), 
        index_to_docstore_id={},
    ) 
    
    vector_store.add_texts(texts) 
    
    vector_store.save_local(output_root) 
    

def create_prompts(ambiguity_type: str, examples: list[str]) -> list[openai.types.chat.ChatCompletionMessageParam]:
    starting_prompt = STARTING_PROMPT.safe_substitute(type=AMBIGUITY_TYPE[ambiguity_type])  
    
    

def main():
    args = parse_args() 
    ambiguity_type = args.ambiguity_type 
    print(f"Received Type to Augment : {ambiguity_type}")
    
    # Load existing examples 
    json_path = args.annotation_root / f"{FILE_NAME[ambiguity_type]}.json" 
    annotations = json.load(open(json_path, 'r')) 
    last_entry = annotations[-1]["SentenceID"] 
    current_index = int(last_entry.split("-")[1]) + 1   # e.g.) ellip-20-a -> 20 
    
    # FAISS vectorstore for duplicate check 
    sentence_data = [annotation['Sentence'] for annotation in annotations] 
    embedding_path = args.embedding_root / f"{FILE_NAME[ambiguity_type]}" 
    if not (embedding_path / "index.pkl").exists() or args.is_create_embeddings:  
        os.makedirs(embedding_path, exist_ok=True) 
        create_existing_embeddings(sentence_data, embedding_path)
    comparing_vector_store = FAISS.load_local(
        embedding_path, 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    ) 
    
    return 


if __name__ == "__main__":
    main()