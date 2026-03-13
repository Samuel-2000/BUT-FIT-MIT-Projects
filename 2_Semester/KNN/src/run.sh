#!/bin/bash

workspace=/content/KNnice
program=$workspace/src/__main__.py
export PYTHONPATH=$workspace/src:$PYTHONPATH


python $program --train --adam --NNmodel 2 --dataset 3 --epochs 3
