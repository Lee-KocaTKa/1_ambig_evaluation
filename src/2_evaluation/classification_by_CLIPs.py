import torch
import argparse 
import json 
import os 
from PIL import Image
from transformers import AutoProcessor, AutoModel
from pathlib import Path  
from tqdm import tqdm
from const import (
    IMAGE_PATH,
    JSON_PATH,
    TYPE_NAMES,
    MODEL_CARDS,
)

def argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-choice", type=str, help="Model choice")
    parser.add_argument("--caption-root", type=Path, default=JSON_PATH) 
    parser.add_argument("--image-root", type=Path, default=IMAGE_PATH)  
    parser.add_argument("--image-style", type=bool, default=False, help="Whether to get results separated by image styles or not")
    return parser.parse_args()


def main(): 
    args = argparse() 
    model_choice = args.model_choice
    caption_root = args.caption_root
    image_root = args.image_root
    
    # Load model and processor 
    model_card = MODEL_CARDS[model_choice]
    processor = AutoProcessor.from_pretrained(model_card) 
    model = AutoModel.from_pretrained(model_card, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda") 
    model.eval() 
    
    # Data Loading 
    
    return 



if __name__ == "__main__":
    main()