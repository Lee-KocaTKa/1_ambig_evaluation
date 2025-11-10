#!/bin/bash 


set -e 


SCRIPT="classification_by_CLIPs.py" 

MODELS=(

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