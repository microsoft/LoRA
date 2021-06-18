#!/bin/bash

echo "downloading e2e-metrics"
git clone https://github.com/tuetschek/e2e-metrics e2e

echo "downloading meteor..."
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz

echo "downloading bleurt checkpoint..."
git clone https://github.com/google-research/bleurt.git
mv bleurt/bleurt/test_checkpoint .
rm -rf bleurt
mkdir bleurt
mv test_checkpoint bleurt

echo "downloading webnlg references..."
git clone https://github.com/Yale-LILY/dart.git tmp
mv tmp/evaluation/webnlg-automatic-evaluation webnlg
rm -r tmp

echo "script complete"
