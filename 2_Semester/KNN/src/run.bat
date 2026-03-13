@echo off
setlocal enabledelayedexpansion
set workspace=..
set program=%workspace%\src\__main__.py
set PYTHONPATH=%workspace%\src;%PYTHONPATH%

python %program% --train --adam --NNmodel 2 --dataset 3 --epochs 3