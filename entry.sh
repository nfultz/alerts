#! /bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"


python3.8 score.py

if [[ `date` = Mon* ]] ; then
  echo "rebuilding model" 
  python3.8 gen_features.py
  python3.8 train.py
fi