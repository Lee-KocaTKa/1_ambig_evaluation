#!/bin/bash 


set -e 


SCRIPT="classification_by_CLIPs.py" 

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


for MODEL in "${MODELS[@]}"; do 
    echo "-----------------------------"
    echo " Running Model: ${MODEL}"
    echo "-----------------------------"

    python "${SCRIPT}" --model-choice "${MODEL}" ${VECFLAG} 

    echo "✅ Finished: ${MODEL}"
    echo
done 

echo "🎉 All models have been evaluated!"