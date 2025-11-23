"""
const.py
Stores prompt templates for OpenAI multimodal classification:
- T2I (caption → image choice)
- I2T (image → caption choice)
- Direct match (optional)
All prompts support Python .format().
"""


# ============================================================
#  T2I MULTI-CHOICE CLASSIFICATION PROMPTS
# ============================================================

JSON_PATH = "../../../data/ViLStrUB/jsons_UNIT2/"
IMAGE_PATH = "../../../data/ViLStrUB/images/"

T2I_SYSTEM_PROMPT = """
You are a careful vision-language classifier.

Task:
- You will receive ONE caption and {num_images} images in a fixed order.
- The images correspond to labels {labels_str} in the same order.
- Your job is to choose which single image best matches the caption.

Rules:
- Respond in a forced choice format: choose ONLY one label from: {labels_str}.
- Then provide a short explanation.

Output format (must follow EXACTLY):
<LETTER>
Explanation: <your reasoning in one short sentence>
"""


T2I_USER_PROMPT = """
Caption:
"{caption}"

Now you will see the images shown in the same order as the labels.

{label_mapping}
"""

# Example label_mapping:
#   Image A is the 1st image.
#   Image B is the 2nd image.
#   Image C is the 3rd image.


# ============================================================
#  I2T MULTI-CHOICE CLASSIFICATION PROMPTS
# ============================================================

I2T_SYSTEM_PROMPT = """
You are a careful vision-language classifier.

Task:
- You will receive ONE image and {num_captions} candidate captions.
- The captions correspond to labels {labels_str}.

Your job is to choose which single caption best matches the image.

Rules:
- Respond in a forced choice format: choose ONLY one label from: {labels_str}.
- Then provide a short explanation.

Output format (must follow EXACTLY):
<LETTER>
Explanation: <your reasoning in one short sentence>
"""


I2T_USER_PROMPT = """
You will see the image first.
Then you will see the candidate captions.

{caption_block}
"""

# Example caption_block:
#   Caption A: "the boy pets the dog"
#   Caption B: "the boy feeds the dog"
#   Caption C: "the boy hugs the dog"


# ============================================================
#  DIRECT MATCHING PROMPTS (optional)
# ============================================================

DIRECT_MATCH_SYSTEM_PROMPT = """
You are a careful vision-language classifier.
You are checking whether an image matches a caption.

Task:
- Determine if the caption describes the image accurately.

Output format (must follow EXACTLY):
match
Explanation: <short reason>
OR:
not match
Explanation: <short reason>
"""

DIRECT_MATCH_USER_PROMPT = """
Caption:
"{caption}"

Image will be provided below.
"""


# ============================================================
#  Helper Format Examples (not used directly by code)
# ============================================================

LABEL_MAPPING_TEMPLATE = """
{lines}
"""
# where lines is something like:
#   "Image A is the 1st image.\nImage B is the 2nd image."

CAPTION_BLOCK_TEMPLATE = """
{lines}
"""
# where lines is:
#   'Caption A: "..." \nCaption B: "..."'

