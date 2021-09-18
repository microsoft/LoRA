#!/bin/bash

echo "creating e2e datasets..."
path=data/e2e
echo "train..."
python src/format_converting_e2e.py $path/train.txt $path/train_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos
echo "test..."
python src/format_converting_e2e.py $path/test.txt $path/test_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_e2e.py $path/valid.txt $path/valid_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "creating webnlg datasets..."
path=data/webnlg_challenge_2017
echo "train..."
python src/format_converting_webnlg.py $path/train.json $path/train_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos

echo "test..."
python src/format_converting_webnlg.py $path/test.json $path/test_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_webnlg.py $path/dev.json $path/valid_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "creating dart datasets..."
path=data/dart
echo "train..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-train.json data/dart/train_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos

echo "test..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-test.json data/dart/test_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos

echo "valid..."
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-dev.json data/dart/valid_formatted.jsonl
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "script complete!"
