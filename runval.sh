#!/usr/bin/env zsh
#SBATCH --job-name=ddm2
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-07:00:00
#SBATCH --output="ddm2stage1-%j.txt"
#SBATCH -G 2
cd $SLURM_SUBMIT_DIR
module load anaconda/full
bootstrap_conda
conda activate ddm2
python3 stage1Val.py -p val -gpu 0,1