#!/bin/bash
# one cytoX_attention model for both unseen cell types (CD14 Mono + NK), trained on the split from split_hcd.py
#   single run :  sbatch run_cytox.sbatch
#   ablation   :  sbatch --array=0-26 run_cytox.sbatch   (3 encoders x 3 shifts x 3 seeds)
#SBATCH --job-name=cytox-attn
#SBATCH --account=bjhm-delta-gpu      # 38 GPU-h left; bjor-delta-gpu has ~1800
#SBATCH --partition=gpuA40x4          # or gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out

# ---- edit these two paths ----
REPO=/projects/bjhm/$USER/cytoX          # clone of the attentionVAE branch
DATA=/projects/bjhm/$USER/data/split     # train.h5ad / test.h5ad from split_hcd.py

module load pytorch-conda/2.8
source /projects/bjhm/$USER/envs/cytox/bin/activate

ENCODERS=(mlp attention hybrid)
SHIFTS=(mean cosine learned)
i=${SLURM_ARRAY_TASK_ID:-0}
ENC=${ENCODERS[$(( i / 9 % 3 ))]}
SHIFT=${SHIFTS[$(( i / 3 % 3 ))]}
SEED=$(( i % 3 ))
# no array --> the main model: hybrid encoder + learned shift attention, seed 0
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then ENC=hybrid; SHIFT=learned; SEED=0; fi

# one outdir per run --> parallel array tasks never write the same results.csv
OUT=$REPO/runs/${ENC}_${SHIFT}_s${SEED}
echo "encoder=$ENC shift=$SHIFT seed=$SEED -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd $REPO
python cytoX_attention/train.py \
  --train $DATA/train.h5ad --test $DATA/test.h5ad \
  --holdout "CD14 Mono" NK \
  --encoder $ENC --shift $SHIFT --seed $SEED \
  --epochs 100 --patience 25 --batch-size 32 \
  --outdir $OUT
