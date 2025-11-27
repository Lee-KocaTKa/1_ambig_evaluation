import argparse 
import json 
import os 
from openai import OpenAI 
from pathlib import Path 
import base64 
from tqdm import tqdm 
from const import T2I_SYSTEM_PROMPT, T2I_USER_PROMPT, I2T_SYSTEM_PROMPT, I2T_USER_PROMPT, JSON_PATH, IMAGE_PATH 
from typing import Any, Dict, List

labels = {
    '1' : 'A',
    '2' : 'B',
    '3' : 'C'
}

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

def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def run_t2i(client, model_card, caption, image_paths):
    labels = "ABC"[:len(image_paths)]
    labels_str = ", ".join(labels)

    system_prompt = T2I_SYSTEM_PROMPT.format(
        num_images=len(image_paths),
        labels_str=labels_str
    )

    # Build label mapping ("Image A is the 1st image" ...)
    mapping_lines = [
        f"Image {labels[i]} is the {i+1}-th image."
        for i in range(len(image_paths))
    ]
    mapping_text = "\n".join(mapping_lines)

    user_prompt = T2I_USER_PROMPT.format(
        caption=caption,
        label_mapping=mapping_text
    )

    # Build message content
    content = [{"type": "text", "text": user_prompt}]

    # Append images in order
    for path in image_paths:
        img_b64 = encode(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    response = client.chat.completions.create(
        model=model_card,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        max_completion_tokens=50,
        temperature=0.0
    )
    answer = response.choices[0].message.content
    return answer


def run_i2t(client, model_card, image_path, captions):
    """
    image_path: str (path to the image)
    captions: list[str] (candidate captions)
    """

    # Number of captions => A, B, (C)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(captions)]
    labels_str = ", ".join(labels)

    # SYSTEM PROMPT
    system_prompt = I2T_SYSTEM_PROMPT.format(
        num_captions=len(captions),
        labels_str=labels_str
    )

    # Build caption block:
    #   Caption A: "xxx"
    #   Caption B: "yyy"
    #   Caption C: "zzz"
    caption_lines = [
        f'Caption {labels[i]}: "{captions[i]}"'
        for i in range(len(captions))
    ]
    caption_block = "\n".join(caption_lines)

    # USER PROMPT
    user_prompt = I2T_USER_PROMPT.format(
        caption_block=caption_block
    )

    # Build content: text + image
    content = [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode(image_path)}"}
        }
    ]

    response = client.chat.completions.create(
        model=model_card,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        max_completion_tokens=50
    )

    return response.choices[0].message.content


def generate_model(args: argparse.Namespace) -> None: 
    """Main Generation loop across ambiguity types"""
    apikey = input("enter the key : ")
    
    client = OpenAI(api_key = apikey)
    
    json_path = args.caption_root 
    image_path = args.image_root 
    model_card = args.model_choice 
    
    
    output_path = "../../../data/ViLStrUB/LLM_classification_results/"
    
    
    image_list = os.listdir(image_path)
    image_list = [i for i in image_list if i != "copy.sh"]
    
    print(f"\n[INFO] Model {model_card}") 
    
    for i in range(len(image_list)): 
        ambig_type = image_list[i] 
        t2i_container = []  # output per type
        i2t_container = []  
        print(f"\n[INFO] Processing type: {ambig_type}") 
        with open(str(json_path) + "/" + ambig_type + ".json", "r") as f: 
            json_data = json.load(f) 
        
        for j in tqdm(range(len(json_data))):
            #sample_container = {} 
            sample = json_data[j]
            style = sample["Style"] 
            #sample_container["GroupID"] = sample["GroupID"]
            #sample_container["Orig"] = sample["Sentence"]
            #sample_container["Style"] = sample["Style"]
            variants = sample["Variants"]
            #variants_num = len(variants)  
            #labels_str = "A, B, C" if variants_num==3 else "A, B" 
            captions = [v["Meaning"] for v in variants] 
            image_paths = [str(image_path) + "/" + ambig_type + "/" + v["Image"] for v in variants] 
            
            #T2I 
            for k in range(len(captions)): 
                caption = captions[k] 
                sample_container = {
                    "caption" : caption, 
                    "caption_number" : "ABCDEF"[k],
                    "style" : style 
                } 
                generated_result = run_t2i(
                    client=client,
                    model_card=model_card,
                    caption=caption,
                    image_paths=image_paths
                )
                sample_container["results"]=generated_result 
                t2i_container.append(sample_container) 
            
            #I2T
            for k in range(len(image_paths)): 
                image_way = image_paths[k] 
                sample_container= {
                    "image_path" : image_way,
                    "captions" : captions,
                    "image_number" : "ABCDEF"[k],
                    "style" : style   
                }
                generated_result = run_i2t(
                    client=client,
                    model_card=model_card,
                    image_path=image_way,
                    captions=captions 
                )
                sample_container["results"]=generated_result 
                i2t_container.append(sample_container) 
                
        t2i_file = output_path + "_" + model_card + "_" + ambig_type + "_t2i.json"
        i2t_file = output_path + "_" + model_card + "_" + ambig_type + "_i2t.json"
        
        with open(t2i_file, "w") as f: 
            json.dump(t2i_container, f, indent=4) 
        with open(i2t_file, "w") as f: 
            json.dump(i2t_container, f, indent=4) 
        
        print(f"file saved to {t2i_file}")
        print(f"file saved to {i2t_file}")
            
            

def main() -> None: 
    args = parse_args() 
    generate_model(args) 
    
    
if __name__=="__main__": 
    main()