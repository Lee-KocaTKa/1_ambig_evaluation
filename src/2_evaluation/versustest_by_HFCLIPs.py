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
    parser.add_argument("--model-choice", type=str, required=True, help="Model choice")
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
            outputs = model(**inputs) 
        else: 
            inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device) 
        #outputs = model(**inputs)
    return {
        "text_embeds": outputs.text_embeds.cpu(),
        "image_embeds": getattr(outputs, "image_embeds", None), 
        "logits_per_image": getattr(outputs, "logits_per_image", None),
        "logits_per_text": getattr(outputs, "logits_per_text", None) 
    } 

def extract_features_only_text(model, tokeniser, text_list, device="cuda"): #-> Dict[str, torch.Tensor]:
    """Extract text and/or image features using the model and processor."""
    with torch.no_grad():
        #if image_list is not None: 
        #    inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True).to(device) 
        #    outputs = model(**inputs) 
         
        inputs = tokeniser(text=text_list, return_tensors="pt", padding=True).to(device) 
        outputs = model.get_text_features(**inputs)
    
    return outputs 

    #    "text_embeds": outputs.text_embeds.cpu(),
    #    "image_embeds": getattr(outputs, "image_embeds", None), 
    #    "logits_per_image": getattr(outputs, "logits_per_image", None),
    #    "logits_per_text": getattr(outputs, "logits_per_text", None) 
    #} 

    
def compute_accuracy(logits: torch.Tensor) -> int: 
    """Compute number of correctly matched pairs in a logits matrix"""
    return sum(row.argmax() == i for i, row in enumerate(logits))

 
def process_sample(
    model, processor, tokeniser, sample, ambiguity_type: str, image_root: Path, vector_extraction: bool 
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
    logits_img = outputs["logits_per_image"].cpu()#.numpy() 
    logits_txt = outputs["logits_per_text"].cpu()#.numpy() 
    
    result["Logits_per_Image"] = logits_img.tolist() 
    
    # Add features if requested 
    if vector_extraction:
        img_feats = outputs["image_embeds"].cpu()#.numpy() 
        txt_feats = outputs["text_embeds"].cpu()#.numpy() 
        for j, v in enumerate(variants): 
            result[f"Variant_{j}_Image_Feature"] = img_feats[j].tolist() 
            result[f"Variant_{j}_Text_Feature"] = txt_feats[j].tolist() 
            
        # Ambiguous sentence text features 
        ambig_out = extract_features_only_text(model, tokeniser, [ambiguous_sentence]) 
        result["Ambiguous_Text_Feature"] = ambig_out.tolist() 
        
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
     
    processor = AutoProcessor.from_pretrained(model_card, use_fast=True)  
    tokeniser = AutoTokenizer.from_pretrained(model_card)
    model = AutoModel.from_pretrained(model_card, dtype=torch.bfloat16, attn_implementation="eager").to(device) 
    model.eval() 
    max_len = model.config.text_config.max_position_embeddings

    
    #results_all = [] 
    #overall_totals = {"i2t": [0, 0], "t2i": [0, 0], "dual": [0, 0]} # [correct, total]
    
    overall_origvs_total = 0 
    overall_origvs_correct = 0 
    overall_distract_total = 0
    overall_distract_correct = 0
    
    for ambiguity_type in TYPE_NAMES: 
        vectors = [] 
        print(f"\n[INFO] Processing type: {ambiguity_type}") 
        json_file = args.caption_root / f"{ambiguity_type}.json" 
        
        if not json_file.exists():
            print(f"[WARNING] Missing file: {json_file}") 
            continue 
        
        with open(json_file, "r") as f: 
            samples = json.load(f) 
        
        
        origvs_total = 0 
        origvs_correct = 0 
        distract_total = 0 
        distract_correct = 0 
        
        for sample in tqdm(samples, desc=f"{ambiguity_type}"): 
            
            ambiguous_caption = sample["Sentence"]
            variants = sample["Variants"] 
            disambiguated_captions = [v["Meaning"] for v in variants] 
            distraction_captions = []
            
            vector_container = {
                "GroupID" : sample["GroupID"],
                "Ambiguous_Caption": ambiguous_caption,
                "Ambiguous_Caption_Vector" : None,
                "Variants": [],
                "Style": sample["Style"]
            }
            
            for v in range(len(variants)):
                variant = variants[v] 
                description = variant["Description"]
                variant_container = {
                    "Image" : variant["Image"],
                    "Description_Vector" : [],
                    "Distraction_Vector" : []   
                }
                origvs_total += 1 
                distract_total += 1 
                
                candidate_captions = disambiguated_captions[:v] + disambiguated_captions[v+1:]
                
                input_image = [Image.open(args.image_root / ambiguity_type / variant["Image"])] 
                orig_input = [variant["Meaning"]] + [ambiguous_caption for _ in range(len(variants)-1)]
                distract_input = [variant["Meaning"]] + [s +  " " + description for s in candidate_captions]
                distraction_captions += distract_input[1:]
                
                with torch.no_grad(): 
                    orig_inputs = processor(
                        text=orig_input, 
                        images=input_image, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_len).to(device)
                    
                    distract_inputs = processor(
                        text=distract_input, 
                        images=input_image, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_len).to(device)
                     
                    ambig_embedding_input = tokeniser(
                        text=[ambiguous_caption], 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_len).to(device)
                    
                    description_embedding_input = tokeniser(
                        text=[description], 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_len).to(device)
                    
                    distraction_embedding_inputs = tokeniser(
                        text=distraction_captions, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=max_len).to(device)
                    
                    orig_outputs = model(**orig_inputs).logits_per_image[0]
                    distract_outputs = model(**distract_inputs).logits_per_image[0]
                    ambig_embedding = model.get_text_features(**ambig_embedding_input).cpu()
                    description_embedding = model.get_text_features(**description_embedding_input).cpu()
                    distraction_embeddings = model.get_text_features(**distraction_embedding_inputs).cpu()  
                    
                orig_pred = torch.argmax(orig_outputs).item() 
                distract_pred = torch.argmax(distract_outputs).item() 
                    
                if orig_pred == 0: 
                    origvs_correct += 1 
                if distract_pred == 0:
                    distract_correct += 1 
                
                variant_container["Description_Vector"] = description_embedding
                variant_container["Distraction_Vector"] = distraction_embeddings 
                
                vector_container["Variants"].append(variant_container)
            
            vector_container["Ambiguous_Caption_Vector"] = ambig_embedding 
            
        overall_origvs_total += origvs_total
        overall_origvs_correct += origvs_correct
        overall_distract_total += distract_total
        overall_distract_correct += distract_correct 
        
        
        print(f"{ambiguity_type}_origvs: {origvs_correct}/{origvs_total} = {origvs_correct / origvs_total:.4f}") 
        print(f"{ambiguity_type}_distract: {distract_correct}/{distract_total} = {distract_correct / distract_total:.4f}")             
                 
    
        # Save pertype pickle 
        output_dir = Path("../../../data/ViLStrUB/versus_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"results_{args.model_choice}_{ambiguity_type}.pkl" 
        
        with open(output_path, "wb") as f: 
            pickle.dump(vectors, f) 
            
        
                
    # Overall Results 
    print("\n[OVERALL RESULTS]")
    print(f"origvs: {overall_origvs_correct}/{overall_origvs_total} = {overall_origvs_correct / overall_origvs_total:.4f}") 
    print(f"distract: {overall_distract_correct}/{overall_distract_total} = {overall_distract_correct / overall_distract_total:.4f}")        
    
    
def main() -> None: 
    args = parse_args() 
    evaluate_model(args) 
    
     
if __name__ == "__main__": 
    main()     