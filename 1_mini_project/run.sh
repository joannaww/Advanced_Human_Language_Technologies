#! /bin/bash

BASEDIR=../../..

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py data/train/ > train.feat
python3 extract-features.py data/devel/ > devel.feat

# train CRF model
echo "Training CRF model..."
python3 train.py train.feat model.crf 
# run CRF model
echo "Running CRF model..."
python3 predict.py devel.feat model.crf > devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 util/evaluator.py NER data/devel devel-CRF.out > devel-CRF.stats

# train LR model
echo "Training LR model..."
python3 train.py train.feat model.lrg 
# run LR model
echo "Running CRF model..."
python3 predict.py devel.feat model.lrg > devel-LR.out
# evaluate LR results
echo "Evaluating LR results..."
python3 util/evaluator.py NER data/devel devel-LR.out > devel-LR.stats

# train RF model
echo "Training RF model..."
python3 train.py train.feat model.rf 
# run LR model
echo "Running RF model..."
python3 predict.py devel.feat model.rf > devel-RF.out
# evaluate LR results
echo "Evaluating RF results..."
python3 util/evaluator.py NER data/devel devel-RF.out > devel-RF.stats

