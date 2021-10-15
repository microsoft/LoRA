#!/bin/bash

cd eval
echo "installing evaluation dependencies"
echo "downloading e2e-metrics..."
git clone https://github.com/tuetschek/e2e-metrics e2e
pip install -r e2e/requirements.txt

echo "downloading GenerationEval for webnlg and dart..."
git clone https://github.com/WebNLG/GenerationEval.git
cd GenerationEval
./install_dependencies.sh
rm -r data/en
rm -r data/ru
cd ..
mv eval.py GenerationEval/

echo "script complete!"
