from string import Template 

LLM_MODEL_CARD = "gpt-4o" 
EMBEDDING_MODEL_CARD = "text-embedding-ada-002"
IMAGE_GENERATION_MODEL_CARD = "dall-e-3"     

FILE_NAME = {
    "vp" : "syntax-vp",
    "pp" : "syntax-pp",
    "anaph" : "anaph",
    "ellip" : "ellip",
    "adjscope" : "adjscope",
    "verbscope" : "verbscope",
    "coordinate" : "coordinate"
}

AMBIGUITY_TYPE = {
    'vp' : "VP Attachment Ambiguity, occuring when it is unclear which part of a sentence a verb phrase is intended to modify",
    'pp' : "PP Attachment Ambiguity, occuring when it is unclear which part of a sentence a prepositional phrase is intended to modify",
    'anaph' : "Anaphoric Ambiguity, which occurs when it is unclear which antecedent a particular anaphor refers to within a given context",
    'ellip' : "Ellipsis Ambiguity, involving the omission of words or phrases that are understood from the context",
    'adjscope'  : "Adjective Scope Ambiguity, occuring when it is unclear how far the influence of an adjective extends within a sentece",
    'verbscope' : "Verb Scope Ambiguity, occuring when it is unclear how far the influence of a verb extends within a sentence",
    'coordinate' : "Coordiante Scope Ambiguity, occuring when it is unclear how far the influence of a coordinating conjunction such as AND/OR extends within a sentence",
    'misc' : "Miscellaneous Ambiguity, which we do not consider at this point"
}

SENTENCE_INDEX_TABLE = {
    0 : 'a',
    1 : 'b',
    2 : 'c',
    3 : 'd',
    4 : 'e',
    5 : 'f',
    6 : 'g',
    7 : 'h',
    8 : 'i',
    9 : 'j'
}

IMAGE_INDEX_TABLE = {
    0: "i",
    1: "ii",
    2: "iii",
    3: "iv",
    4: "v",
    5: "vi",
    6: "vii",
    7: "viii",
    8: "ix",
    9: "x",
}

STARTING_PROMPT = Template("""
Hi, I'm making a dataset by extending the following examples.
Output sentences in the following format:
    - An ambiguous sentence having 2 or 3 possible meanings: Avoid repeating common phrases and use a wide range of vocabulary and creative expression, a variety of synonyms and idioms.
    - Disambiguated sentences corresponded to ambiguous sentence: Do not say something else but just 2 or 3 sentences. These sentences are connected slash.
    - If I'm not satisfied, I will give you feedback. If I say good, then generate another round.
    - Create a text filled with detailed that allows one to easily visualize the scene.
The topic is ${type}. From now on, I will show you some of the examples:
"""
)
