#!/bin/bash

# input arguments
DATA="${1-MUTAG}" 

# general settings
gm=IHGNN  # model
gpu_or_cpu=gpu
GPU=1  # select the GPU number
CONV_SIZE="32-64-32"
max_k=1  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=32  # batch size, set to 50 or 100 to accelerate training
dropout=True

# dataset-specific settings
case ${DATA} in
MUTAG)
  num_epochs=350
  learning_rate=0.0001
  ;;
BZR)
  num_epochs=350
  learning_rate=0.01
  ;;
BZR_MD)
  num_epochs=350
  learning_rate=0.01
  ;;
COX2)
  num_epochs=350
  learning_rate=0.01
  ;;
COX2_MD)
  num_epochs=350
  learning_rate=0.01
  ;;
DHFR)
  num_epochs=350
  learning_rate=0.0001
  ;;
DHFR_MD)
  num_epochs=350
  learning_rate=0.01
  ;;
ENZYMES)
  num_epochs=350
  learning_rate=0.01
  ;;
KKI)
  num_epochs=350
  learning_rate=0.01
  ;;
NCI1)
  num_epochs=350
  learning_rate=0.0001
  ;;
NCI109)
  num_epochs=350
  learning_rate=0.0001
  ;;
DD)
  num_epochs=350
  learning_rate=0.00001
  ;;
PTC)
  num_epochs=350
  learning_rate=0.0001
  ;;
PTC_FM)
  num_epochs=350
  learning_rate=0.0001
  ;;
PTC_FR)
  num_epochs=350
  learning_rate=0.01
  ;;
PTC_MM)
  num_epochs=350
  learning_rate=0.01
  ;;
PTC_MR)
  num_epochs=350
  learning_rate=0.01
  ;;
PROTEINS)
  num_epochs=350
  learning_rate=0.00001
  ;;
COLLAB)
  num_epochs=350
  learning_rate=0.0001
  #sortpooling_k=0.9
  ;;
IMDBBINARY)
  num_epochs=350
  learning_rate=0.01
  #sortpooling_k=0.9
  ;;
IMDBMULTI)
  num_epochs=350
  learning_rate=0.0001
  #sortpooling_k=0.9
  ;;
REDDIT-BINARY)
  num_epochs=350
  learning_rate=0.0001
  #sortpooling_k=0.9
  ;;
REDDIT-MULTI-5K)
  num_epochs=350
  learning_rate=0.0001
  #sortpooling_k=0.9
  ;;
*)
  num_epochs=500
  learning_rate=0.00001
  ;;
esac

echo "Running 10-fold cross validation"
start=`date +%s`
for i in $(seq 1 10)
do
  echo "fold: $i" 
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      -seed 1 \
      -data $DATA \
      -degree_as_tag 0 \
      -fold $i \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -max_k $max_k \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout
done
stop=`date +%s`
echo "End of cross-validation"
echo "The total running time is $[stop - start] seconds."
echo "The accuracy results for ${DATA} are as follows:"

python calculate_average_accuracy.py \
        --file_name MUTAG_acc_results.txt \
