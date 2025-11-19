### for the given json datasets, mark the image style according to the premarked lists 

import json 

ellip_C = [26,  19,  15,  9, 71, 63, 62, 61, 17, 2, 50, 32, 38, 21, 41, 10,  55, 22, 24, 11, 60, 34, 16, 29, 44, 47,46, 23, 30, 51, 53, 56, 7, 3, 36, 35, 14, 42,  31, 59,  33, 45, 43, 28, 58, 6,  57,  48,  5, 37, 8, 20, 49, 54, 40]
ellip_P = [98, 99, 100, 97, 96, 95, 94, 93, 92, 91, 89, 90, 88, 87, 85, 86, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 70, 69, 68, 67, 66, 65, 64, 39, 52, 27, 25, 18, 13, 4, 12, 1]

PP_C = [40, 24, 22, 6, 74,66, 59, 58, 9, 42, 8, 7, 4, 50, 49, 5, 48, 47, 46, 44, 45, 43, 41, 38, 39, 36, 35, 37, 34, 32, 30, 3, 31, 33,
29, 26, 27, 28, 23, 25, 21, 20, 19, 2, 18, 17, 16, 15, 14, 10, 11, 12, 13, 1]
PP_P = [102, 101, 103, 100, 99, 98, 97, 94, 95, 96, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 
 71, 72,73,70, 69, 68, 67, 65, 64, 63, 62, 61, 60, 57, 56, 55, 54, 53, 52, 51]

anaph_C = [83, 87, 51, 26, 46, 11,  81,79, 71, 70,
66, 68, 67, 63, 62, 60, 61, 59, 58, 57, 50, 53, 56, 33, 28, 27, 24, 25, 9, 7, 6, 
55, 8, 52, 54, 48, 49, 5, 47, 44, 45, 42, 43, 40, 4, 39, 41, 38, 37, 36, 35, 
32, 31, 34, 30, 3, 29, 23, 22, 20, 21, 19, 2, 18, 15, 16, 17, 14, 12, 13, 10, 1]
anaph_P = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 89, 90, 88, 86, 84, 80,  77, 76, 75
,74, 73, 72, 69, 64, 65, 78, 82, 85   ]

adj_C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 52, 69]
adj=P = [45, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

coord_C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72] 
coord_P = [54, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 , 90, 91, 92, 93, 94, 95, 96, 97, 98 , 99, 100] 

VP_C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,63, 64, 65, 66, 67, 68, 69, 70, 95, 143, 144, 145, 162, 163]
VP_P = [35, 36, 37, 38, 39, 40, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100 ]

vbs_C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 58, 60, 63, 69, 85, 86, 94]
vbs_P = [54, 55, 56, 57, 59, 61, 62, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100 ]


path = '../../../novy_vidaco/data/vidaco/testing_data/'
outputpath = '../../../data/ViLStrUB/jsons/'

with open(path + "adjscope_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in adj_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "adjscope.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 



with open(path + "anaph_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in anaph_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "anaph.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 
    
    
with open(path + "coordinate_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in coord_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "conj.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 
    
    
with open(path + "ellip_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in ellip_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "ellip.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 
    
    
    
with open(path + "syntax-pp_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in PP_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "pp.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 
    
    
with open(path + "syntax-vp_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in VP_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "vp.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 
    
    
with open(path + "verbscope_testing.json", "r", encoding="utf-8") as f: 
    data = json.load(f) 

for i in range(len(data)): 
    sample = data[i] 
    number = int(sample["GroupID"].split("-")[-1]) 
    if number in vbs_C: 
        sample["Style"] = "toon"
    else: 
        sample["Style"] = "foto" 

with open(outputpath + "verbscope.json", "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=4, ensure_ascii=False) 