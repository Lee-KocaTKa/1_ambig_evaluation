import argparse 
import json
import os 
import pickle 
from pathlib import Path
from typing import Any, Dict, List 

import torch 
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from const import IMAGE_PATH, JSON_PATH, MODEL_CARDS, TYPE_NAMES

def parse_args():
    """Parse command-line input arguments."""
    parser = argparse.ArgumentParser(description="Classification by CLIP models")
    parser.add_argument("--model-choice", type=str, requried=True, help="Model choice")
    parser.add_argument("--caption-root", type=Path, default=JSON_PATH) 
    parser.add_argument("--image-root", type=Path, default=IMAGE_PATH)  
    parser.add_argument("--image-style", action="store_true", help="Whether to get results separated by image styles or not")
    #parser.add_argument("--random-chance", type=bool, default=False, help="Whether to use ambiguous caption random matching or not")
    parser.add_argument("--vector-extraction", action="store_true", help="Whether to extract feature vectors or not")
    return parser.parse_args()


def check_images_exist(variants: List[Dict[str, Any]], image_root: Path, ambiguity_type:str) -> bool:
    """Check that all images in the variants exist."""
    for variant in variants: 
        image_path = image_root / ambiguity_type / variant["Image"] 
        if not image_path.exists(): 
            print(f"[Warning] Missing Image: {ambiguity_type}/{variant['Image']}")
            return False
    return True


def extract_features(model, processor, text_list, image_list=None, device="cuda") -> Dict[str, torch.Tensor]:
    """Extract text and/or image features using the model and processor."""
    with torch.no_grad():
        if image_list is not None: 
            inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True).to(device) 
        else: 
            inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device) 
        outputs = model(**inputs)
    return {
        "text_embeds": outputs.text_embeds.cpu(),
        "image_embeds": getattr(outputs, "image_embeds", None), 
        "logits_per_image": getattr(outputs, "logits_per_image", None)
    } 

    
def compute_accuracy(logits: torch.Tensor) -> int: 
    """Compute number of correctly matched pairs in a logits matrix"""
    return sum(row.argmax() == i for i, row in enumerate(logits))

 
def process_sample(
    model, processor, sample, ambiguity_type: str, image_root: Path, vector_extraction: bool 
) -> Dict[str, Any]: 
    """Process a single sample and return its result dictionary""" 
    variants = sample["Variants"]
    image_style = sample.get("Style", "")  # Style yet uploaded 
    group_id = sample["GroupID"] 
    ambiguous_sentence = sample["Sentence"] 
    
    if not check_images_exist(variants, image_root, ambiguity_type): 
        return {} 
    
    result = {
        "GroupID": group_id, 
        "Style": image_style, 
        "Ambiguous_Sentence": ambiguous_sentence 
    }
    
    input_captions = [v["Meaning"] for v in variants] 
    input_images = [Image.open(image_root / ambiguity_type / v["Image"]) for v in variants] 
    
    # Forward Pass 
    outputs = extract_features(model, processor, input_captions, input_images) 
    logits_img = outputs["logits_per_image"].cpu().numpy() 
    logits_txt = outputs["logits_per_text"].cpu().numpy() 
    
    result["Logits_per_Image"] = logits_img.tolist() 
    
    # Add features if requested 
    if vector_extraction:
        img_feats = outputs["image_embeds"].cpu().numpy() 
        txt_feats = outputs["text_embeds"].cpu().numpy() 
        for j, v in enumerate(variants): 
            result[f"Variant_{j}_Image_Feature"] = img_feats[j].tolist() 
            result[f"Variant_{j}_Text_Feature"] = txt_feats[j].tolist() 
            
        # Ambiguous sentence text features 
        ambig_out = extract_features(model, processor, [ambiguous_sentence]) 
        result["Ambiguous_Text_Feature"] = ambig_out["text_embeds"][0].tolist() 
        
    # Accuracy computation 
    i2t_correct = compute_accuracy(torch.tensor(logits_img)) 
    t2i_correct = compute_accuracy(torch.tensor(logits_txt)) 
    dual_correct = sum(
        (logits_img[i].argmax() == i) and (logits_txt[i].argmax() == i)
        for i in range(len(logits_img)) 
    )
    
    result["i2t_correct"] = i2t_correct
    result["t2i_correct"] = t2i_correct
    result["dual_correct"] = dual_correct
    result["num_variants"] = len(variants) 
    
    return result 


def evaluate_model(args: argparse.Namespace) -> None: 
    """Main Evaluation loop across ambiguity types."""
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model_card = MODEL_CARDS[args.model_choice]
    print(f"[INFO] Loading model: {args.model_choice} ({model_card})")
     
    processor = AutoProcessor.from_pretrained(model_card) 
    #tokeniser = AutoTokenizer.from_pretrained(model_card)
    model = AutoModel.from_pretrained(model_card, dtype=torch.bfloat16, attn_implementation="sdpa").to(device) 
    model.eval() 
    
    #results_all = [] 
    overall_totals = {"i2t": [0, 0], "t2i": [0, 0], "dual": [0, 0]} # [correct, total]
    
    for ambiguity_type in TYPE_NAMES: 
        print(f"\n[INFO] Processing type: {ambiguity_type}") 
        json_file = args.caption_root / f"{ambiguity_type}_test.json" 
        
        if not json_file.exists():
            print(f"[WARNING] Missing file: {json_file}") 
            continue 
        
        with open(json_file, "r") as f: 
            samples = json.load(f) 
        
        results = [] 
        per_type_totals = {"i2t": [0, 0], "t2i": [0, 0], "dual": [0, 0]}
        
        for sample in tqdm(samples, desc=f"{ambiguity_type}"): 
            result = process_sample(model, processor, sample, ambiguity_type, args.image_root, args.vector_extraction)
            if not result: 
                print(f"[WARNING] Noisy Sample: {sample["GroupID"]}")
                continue 
            results.append(result) 
            
            n = result["num_variants"] 
            per_type_totals["i2t"][0] += result["i2t_correct"] 
            per_type_totals["i2t"][1] += n 
            per_type_totals["t2i"][0] += result["t2i_correct"] 
            per_type_totals["t2i"][1] += n 
            per_type_totals["dual"][0] += result["dual_correct"] 
            per_type_totals["dual"][1] += n 
    
        # Save pertype pickle 
        output_dir = Path("../data/classification_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"results_{args.model_choice}_{ambiguity_type}.pkl" 
        with open(output_path, "wb") as f: 
            pickle.dump(results, f) 
            
        # Print pertype accuracy 
        keylist = ["i2t", "t2i", "dual"]
        
        print(f"\n[RESULTS: {ambiguity_type}]") 
        for key in keylist: 
            correct, total = per_type_totals[key] 
            if total > 0: 
                acc = correct / total 
                print(f"  {key}: {correct}/{total} = {acc:.4f}") 
                overall_totals[key][0] += correct
                overall_totals[key][1] += total 
                
    # Overall Results 
    print("\n[OVERALL RESULTS]")
    for key in keylist: 
        correct, total = overall_totals[key] 
        print(f"{key}: {correct}/{total} = {correct / total:.4f}") 
    
    
def main() -> None: 
    args = parse_args() 
    evaluate_model(args) 
    
     
if __name__ == "__main__": 
    main()     