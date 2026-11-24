#!/bin/bash
# best-guess configs (no tuning run yet) + references, 3 seeds where it matters
#   sbatch --array=0-6 delta/run_next.sbatch
#
#   0-2  hybrid encoder + cosine shift   (expected best: no learned shift params, tau picked by LOO)
#   3-5  hybrid encoder + learned shift  (fixed version: bounded scores, best-LOO checkpoint, lr 1e-3)
#   6    mlp encoder    + mean shift     (scGen re-implementation, reference)
#SBATCH --job-name=cytox-next
#SBATCH --account=bjhm-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out

REPO=/projects/bjhm/tliu16/cytoX
DATA=/projects/bjhm/data/split

module load pytorch-conda/2.8
source /projects/bjhm/tliu16/envs/cytox/bin/activate

CONFIGS=(
  "hybrid cosine 0"  "hybrid cosine 1"  "hybrid cosine 2"
  "hybrid learned 0" "hybrid learned 1" "hybrid learned 2"
  "mlp mean 0"
)
read ENC SHIFT SEED <<< "${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT=$REPO/runs/next/${ENC}_${SHIFT}_s${SEED}
echo "encoder=$ENC shift=$SHIFT seed=$SEED -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd $REPO
python cytoX_attention/train.py \
  --train $DATA/train.h5ad --test $DATA/test.h5ad \
  --holdout "CD14 Mono" NK \
  --encoder $ENC --shift $SHIFT --seed $SEED \
  --epochs 200 --patience 25 --batch-size 32 \
  --shift-lr 1e-3 --shift-epochs 200 \
  --outdir $OUT
