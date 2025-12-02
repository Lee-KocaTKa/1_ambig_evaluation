import json 


t2ipath = "../../../data/ViLStrUB/LLM_classification_results/_gpt-4o_conj_t2i.json"
i2tpath = "../../../data/ViLStrUB/LLM_classification_results/_gpt-4o_conj_i2t.json"

with open(t2ipath, "r") as f:
    t2i_results = json.load(f)
with open(i2tpath, "r") as f:
    i2t_results = json.load(f)
    
print("4o")    

t2i_count = 0 
i2t_count = 0
t2i_correct = 0
i2t_correct = 0

dual_correct = 0
dual_count = 0

for i in range(len(t2i_results)):
    t2i_count += 1
    i2t_count += 1
    dual_count += 1
    
    i2t_seikai = 0 
    t2i_seikai = 0
    
    t2i = t2i_results[i]
    i2t = i2t_results[i]
    
    i2t_gold = i2t["image_number"]
    t2i_gold = t2i["caption_number"]
        
    t2i_answer = t2i["results"]
    i2t_answer = i2t["results"]
    
    if t2i_answer.startswith(t2i_gold):
        t2i_seikai = 1
        t2i_correct += t2i_seikai 
    if i2t_answer.startswith(i2t_gold):
        i2t_seikai = 1
        i2t_correct += i2t_seikai
    if t2i_seikai == 1 and i2t_seikai == 1:
        dual_correct += 1
        
print(f"T2I Accuracy: {t2i_correct / t2i_count:.4f} ({t2i_correct}/{t2i_count})")
print(f"I2T Accuracy: {i2t_correct / i2t_count:.4f} ({i2t_correct}/{i2t_count})")
print(f"Dual Accuracy: {dual_correct / dual_count:.4f} ({dual_correct}/{dual_count})")


    