

IMAGE_PATH = '../../../novy_vidaco/data/vidaco/images'
JSON_PATH = '../../../data/ViLStrUB/jsons/' 
TYPE_NAMES = ['vp', 'pp', 'anaph', 'ellip', 'adjscope', 'verbscope', 'conj']  
MODEL_CARDS = {
    "vit-clip" : "openai/clip-vit-large-patch14-336",
    "rn-clip" : "JaehoHan/custom-clip-resnet50-image-encoder",
    "vit-openclip" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "convnext-openclip" : "laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg",
    "metaclip" : "facebook/metaclip-h14-fullcc2.5b",
    "metaclip2" : "facebook/metaclip-2-worldwide-l14",
    "siglip" : "google/siglip-so400m-patch14-384",
    "siglip2" : "google/siglip2-so400m-patch16-512",
    "fg-clip" : "qihoo360/fg-clip-large"
}