#!/bin/bash 


set -e 


SCRIPT="classification_by_EVACLIP.py" 

MODELS=(
    #"vit-clip"
    #"rn-clip"
    #"vit-openclip"
    #"convnext-openclip"
    #"metaclip"
    #"metaclip2"
    #"siglip"
    #"siglip2"

)

VECFLAG="--vector-extraction" 


python "${SCRIPT}" ${VECFLAG}

echo "🎉 finished!"