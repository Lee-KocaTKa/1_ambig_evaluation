import json 
import os 
from openai import OpenAI 
import base64 


image_path = "../../../data/ViLStrUB/images/"
json_path = "../../../data/ViLStrUB/jsons/" 
output_path = "../../../data/ViLStrUB/jsons_UNIT2/"

def encode_image(path): 
    with open(path, 'rb') as f: 
        return base64.b64encode(f.read()).decode("utf-8") 


def extract_visual_objects(client, image_path, caption):
    # Encode image
    img_b64 = encode_image(image_path)

    # -------- System Prompt (Stable Agent Definition) --------
    system_prompt = """
You are a vision-language agent that outputs visual attributes
ONLY for objects explicitly mentioned in the caption.

Your behavior rules:
- Describe ONLY objects that appear both in the caption and the image.
- For each object, output at most TWO visual attributes.
- Attributes must be concise (e.g., "red", "wooden", "large", "striped shirt").
- Ignore objects not mentioned in the caption.
- Ignore actions, relationships, and scene-level descriptions.
- Keep it concise and factual.
Example style:
"the boy is wearing a striped shirt and the dog has brown fur"
"""

    # -------- User Prompt (Task Instance) --------
    user_prompt = f"""
Caption: "{caption}"
Now look at the image and output the object descriptions.
"""

    # --- API call ---
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                    }
                ]
            }
        ],
        temperature=0.0,
        max_completion_tokens=300
    )

    return response.choices[0].message.content


def main(): 
    apikey = input("enter the key : ")
    client = OpenAI(api_key=apikey)
    
    image_list = os.listdir(image_path) 
    for i in range(len(image_list)): 
        ambig_type = image_list[i] 
        if ambig_type in ["copy.sh", "adjscope", "anaph", "verbscope"]: 
            continue 
        print(f"Describing from type {ambig_type}") 
        with open(json_path + ambig_type + ".json", 'r') as f: 
            json_data = json.load(f) 
            
        for j in range(len(json_data)): 
            sample = json_data[j]
            caption = sample["Sentence"]
            variants = sample["Variants"] 
            for k in range(len(variants)): 
                variant = variants[k] 
                image_name = variant["Image"] 
                image_file_path = image_path + ambig_type + "/" + image_name 
                print(f"Describing image {image_name}") 
                description = extract_visual_objects(
                    client= client, 
                    image_path= image_file_path,
                    caption = caption
                )
                variant["Description"] = description 
        
        output_file_path = output_path + ambig_type + ".json" 
        with open(output_file_path, "w") as out: 
            json.dump(json_data, out, indent=4) 
        print(f"Saved descriptions to {output_file_path}") 
    
    return 




if __name__=="__main__": 
    main() 
    