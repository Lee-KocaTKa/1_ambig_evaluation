import torch
import argparse 
import json 
import os 
import pickle 
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from pathlib import Path  
from tqdm import tqdm
from const import (
    IMAGE_PATH,
    JSON_PATH,
    TYPE_NAMES,
    MODEL_CARDS,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-choice", type=str, help="Model choice")
    parser.add_argument("--caption-root", type=Path, default=JSON_PATH) 
    parser.add_argument("--image-root", type=Path, default=IMAGE_PATH)  
    parser.add_argument("--image-style", type=bool, default=False, help="Whether to get results separated by image styles or not")
    #parser.add_argument("--random-chance", type=bool, default=False, help="Whether to use ambiguous caption random matching or not")
    return parser.parse_args()

def image_nonexistent(variants: list[dict], image_root: str, ambiguity_type:str) -> bool:
    for v in variants: 
        image_path = image_root / ambiguity_type / v["Image"] 
        try:
            image = Image.open(image_path) 
        except:
            print(f"Image not found: {ambiguity_type}/{v['Image']}")  
            return True   
    return False 

def main(): 
    args = argparse() 
    model_choice = args.model_choice
    caption_root = args.caption_root
    image_root = args.image_root
    #random_chance = args.random_chance 
    
    # Load model, tokeniser and processor 
    model_card = MODEL_CARDS[model_choice]
    processor = AutoProcessor.from_pretrained(model_card) 
    tokeniser = AutoTokenizer.from_pretrained(model_card) 
    model = AutoModel.from_pretrained(model_card, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda") 
    model.eval() 
    
    # Data Path 
    image_path = image_root 
    json_path = caption_root
    
    # Loop over ambiguity types
    for ambiguity_type in TYPE_NAMES:
        # Ambiguity Type Declaration
        print(f"Processing Ambiguity Type: {ambiguity_type}")
        
        # Load JSON file 
        json_file = json_path / f"{ambiguity_type}_test.json"  
        with open(json_file, 'r') as f:
            data = json.load(f) 
        
        result_container = [] 
        
        # Process each sample
        for i in tqdm(range(len(data))):
            result_box = {} # result per sample to go inside result_container 
            sample = data[i]
            variants = sample["Variants"]
            image_style = sample["Style"]  # JsonファイルにStyle情報があることを前提 
            
            result_box["GroupID"] = sample["GroupID"]
            result_box["Style"] = image_style  
            result_box["Ambiguous Sentence"] = sample["Sentence"] 
            
            if image_nonexistent(variants, image_path, ambiguity_type): 
                continue 
            
            # Feature Extraction
            #if random_chance: # Ambiguous caption as inputs
            #    input_captions = [sample["Sentence"] for _ in variants] 
            #else:  # Resolved captions as inputs
            input_captions = [v["Meaning"] for v in variants]
            ambig_captions = [sample["Sentence"] for _ in variants]  
            
            input_images = [Image.open(image_path / ambiguity_type / v["Image"]) for v in variants] 
            
            inputs = processor(
                text = input_captions,
                images = input_images,
                return_tensors="pt",
                padding=True
            ).to(model.device) 
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs) 
                logits_per_image = outputs.logits_per_image.cpu().numpy() 
                result_box["Logits"] = logits_per_image.tolist() 
    return 



if __name__ == "__main__":
    main()