#!/bin/bash -l
#
#SBATCH --job-name="glstm"
#SBATCH --exclusive
#SBATCH --partition=tflow
#SBATCH --time=02:00:00
#SBATCH --output=lstm.%j.out
#SBATCH --error=lstm.%j.err

module load python/3.5.0
module load cudnn/8.0

save_dir="save_"$1
rnn_size=256
num_layers=2
learning_rate=0.001
num_epochs=100
data_dir='data/gio'
batch_size=50
seq_length=100
output_keep_prob=1.0
input_keep_prob=1.0
grad_clip=5.0
decay_rate=0.97

srun python3 train.py --save_dir=$save_dir --rnn_size=$rnn_size --learning_rate=$learning_rate --num_layers=$num_layers \
--output_keep_prob=$output_keep_prob --input_keep_prob=$input_keep_prob --grad_clip=$grad_clip --decay_rate=$decay_rate \
--num_epochs=$num_epochs --batch_size=$batch_size --seq_length=$seq_length --data_dir=$data_dir
