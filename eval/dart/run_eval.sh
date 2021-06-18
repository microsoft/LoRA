#! /bin/bash


export CURR_DIR=$PWD
export TMP_DIR=~/temp_eval

mkdir ${TMP_DIR}


OUTPUT_FILE=$1
EVAL_FILE=$2

echo $OUTPUT_FILE
echo "formatting OUTPUT_FILE..."
python tolower.py $OUTPUT_FILE

OUTPUT_FILE=${OUTPUT_FILE}.lt2


TEST_TARGETS_REF0=our_dart_ref/reference0.txt
TEST_TARGETS_REF1=our_dart_ref/reference1.txt
TEST_TARGETS_REF2=our_dart_ref/reference2.txt


# BLEU
echo "running BLEU..."
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} < ${OUTPUT_FILE} > ${TMP_DIR}/bleu.txt

python prepare_files.py ${OUTPUT_FILE} ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2}

# METEOR
cd ../meteor-1.5/
echo "running MET..."
java -Xmx2G -jar meteor-1.5.jar ${OUTPUT_FILE} ${TMP_DIR}/all-notdelex-refs-meteor.txt -l en -norm -r 8 > ${TMP_DIR}/meteor.txt
cd ../eval

# TER
cd ../tercom-0.7.25/
echo "running TER..."
java -jar tercom.7.25.jar -h ${TMP_DIR}/relexicalised_predictions-ter.txt -r ${TMP_DIR}/all-notdelex-refs-ter.txt > ${TMP_DIR}/ter.txt
cd ../eval


# # Uncomment out MoverScore, BERTScore and or BLEURT if the additional metrics are desired

# # MoverScore
# echo "running MOVER..."
# python moverscore.py ${TEST_TARGETS_REF0} ${OUTPUT_FILE} > ${TMP_DIR}/moverscore.txt

# # BERTScore
# echo "running BERTScore..."
# bert-score -r ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} -c ${OUTPUT_FILE} --lang en > ${TMP_DIR}/bertscore.txt

# # BLEURT
# echo "running BLEURT..."
# python -m bleurt.score -candidate_file=${OUTPUT_FILE} -reference_file=${TEST_TARGETS_REF0} -bleurt_checkpoint=../bleurt/bleurt/test_checkpoint -scores_file=${TMP_DIR}/bleurt.txt


# Write out eval metrics to file
python print_scores.py $EVAL_FILE

# Clean up tmp eval dir
rm -r ${TMP_DIR}
