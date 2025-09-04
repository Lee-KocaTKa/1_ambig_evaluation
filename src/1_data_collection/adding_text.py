import argparse  
import json 
import os 
import openai 
from pathlib import Path 
from langchain.embeddings import OpenAIEmbeddings 

from const import (
    LLM_MODEL_CARD,
    EMBEDDING_MODEL_CARD,
    FILE_NAME,  
    AMBIGUITY_TYPE,
    SENTENCE_INDEX_TABLE,
    STARTING_PROMPT
)

# Text Generation & Text Embedding Models 
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])        ### set your OpenAI API key as an environment variable
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_CARD)   

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--trials', type=int, default=1000, help='API call count limits')
    parser.add_argument('--ambiguity-type', type=str, help='Type of Ambiguity') 
    parser.add_argument('--annotation-root', type=Path, default='../../data/vidaco_jsons') 
    parser.add_argument('--embedding-root', type=Path, default='../../data/embeddings')  
    parser.add_argument('--is-create-embeddings', action='store_true')

def main():
    args = parse_args() 
    ambiguity_type = args.ambiguity_type 
    print(f"Received Type to Augment : {ambiguity_type}")
    
    # Load existing examples 
    json_path = args.annotation_root / f"{FILE_NAME[ambiguity_type]}.json" 
    annotations = json.load(open(json_path, 'r')) 
    last_entry = annotations[-1]["SentenceID"] 
    current_index = int(last_entry.split("-")[1]) + 1   # e.g.) ellip-20-a -> 20 
    
    
    
    return 


if __name__ == "__main__":
    main()