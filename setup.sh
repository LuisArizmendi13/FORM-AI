#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "FORM-AI"; then
    echo "FORM-AI Conda environment already exists, switching..."
else
    echo "Creating FORM-AI Conda environment..."
    conda create -y -n "FORM-AI" python=3.10

fi

conda run -n FORM-AI pip install -r requirements.txt
conda run -n FORM-AI python -m pip install -e detectron2
