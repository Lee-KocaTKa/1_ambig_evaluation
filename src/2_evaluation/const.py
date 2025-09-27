

IMAGE_PATH = '../../novy_vidaco/data/vidaco/images'
JSON_PATH = '../../novy_vidaco/data/vidaco/testing_data' 
TYPE_NAMES = ['syntax-vp', 'syntax-pp', 'anaph', 'ellip', 'adjscope', 'verbscope', 'coordinate']  
MODEL_CARDS = {
    "clip-base" : "openai/clip-vit-base-patch32",
    "clip-large" : "openai/clip-vit-large-patch14",
    "clip-336" : "openai/clip-vit-large-patch14-336",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" : "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
}