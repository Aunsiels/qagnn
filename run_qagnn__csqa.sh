#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 48:00:00

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/lib/cuda-10.1/targets/x86_64-linux/lib"


dataset="csqa"
kb="all"
model='roberta-large'
shift
shift
args=$@


elr="1e-5"
dlr="1e-3"
bs=128
n_epochs=40

k=6 #num of gnn layers
gnndim=200
emb="tzw"
max_node=400

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "KB: $kb"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "Max nodes num: $max_node"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Training ######
for seed in 0; do
  CUDA_LAUNCH_BLOCKING=1 python3 -u qagnn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}_${kb}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}_${kb}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}_${kb}/graph/test.graph.adj.pk \
      --train_statements "./data/${dataset}_${kb}/statement/train.statement.jsonl" \
      --dev_statements "./data/${dataset}_${kb}/statement/dev.statement.jsonl" \
      --test_statements "./data/${dataset}_${kb}/statement/test.statement.jsonl" \
      --inhouse_train_qids "./data/${dataset}_${kb}/inhouse_split_qids.txt" \
      --kb ${kb} \
      --ent_emb_paths "./data/${kb}/tzw.ent.npy" \
      --num_relation 45 \
      --max_node_num ${max_node} \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > train_${kb}_${dataset}__emb_${emb}_max_nodes_${max_node}_enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done
