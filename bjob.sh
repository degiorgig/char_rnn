#!/bin/bash -l
#
#SBATCH --job-name="glstm"
#SBATCH --exclusive
#SBATCH --partition=tflow
#SBATCH --time=01:00:00
#SBATCH --output=lstm.%j.out
#SBATCH --error=lstm.%j.err

module load python/3.5.0
module load cudnn/8.0

date=`date '+%Y-%m-%d %H:%M:%S'`
save_dir="save_2"

srun python3 train.py --save_dir $save_dir --rnn_size 512 --learning_rate 0.001 --num_epochs 30 --data_dir 'data/gio'
