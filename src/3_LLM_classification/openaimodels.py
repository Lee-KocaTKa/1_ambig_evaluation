import argparse 
import json 
import os 
from openai import OpenAI 
from pathlib import Path 
import base64 
from tqdm import tqdm 
from const import T2I_SYSTEM_PROMPT, T2I_USER_PROMPT, I2T_SYSTEM_PROMPT, I2T_USER_PROMPT, JSON_PATH, IMAGE_PATH 

def parse_args(): 
    """Parse command-line input arguments."""
    parser = argparse.ArgumentParser(description="Classification by Openai GPT models")
    parser.add_argument("--model-choice", type=str, required=True, help="Model choice")
    parser.add_argument("--caption-root", type=Path, default=JSON_PATH) 
    parser.add_argument("--image-root", type=Path, default=IMAGE_PATH)  
    parser.add_argument("--image-style", action="store_true", help="Whether to get results separated by image styles or not")
    #parser.add_argument("--random-chance", type=bool, default=False, help="Whether to use ambiguous caption random matching or not")
    #parser.add_argument("--vector-extraction", action="store_true", help="Whether to extract feature vectors or not")
    return parser.parse_args()

def check_images_exist(variants: List[Dict[str, Any]], image_root: Path, ambiguity_type:str) -> bool:
    """Check that all images in the variants exist."""
    for variant in variants: 
        image_path = image_root / ambiguity_type / variant["Image"] 
        if not image_path.exists(): 
            print(f"[Warning] Missing Image: {ambiguity_type}/{variant['Image']}")
            return False
    return True


def generate_model(args: argparse.Namespace) -> None: 
    """Main Generation loop across ambiguity types"""
    apikey = input("enter the key : ")
    client = OpenAI(api_key = apikey)
    
    json_path = args.caption_root 
    image_path = args.image_root 
    model_card = args.model_choice 
    
    
    output_path = "../../../data/ViLStrUB/LLM_classification_results"
    output_container = [] 
    
    image_list = os.listdir(image_path)
    image_list = [i for i in image_list if i != "copy.sh"]
    
    print(f"\n[INFO] Model {model_card}") 
    
    for i in range(len(image_list)): 
        ambig_type = image_list[i] 
        print(f"\n[INFO] Processing type: {ambig_type}") 
        with open(json_path + ambig_type + ".json", "r") as f: 
            json_data = json.load(f) 
        
        for j in range(len(json_data)): 
            sample = json_data[j] 
            

def main() -> None: 
    args = parse_args() 
    generate_model(args) 
    
    
if __name__=="__main__": 
    main()