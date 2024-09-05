#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J MobileNetV2_dnl
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=20000
#SBATCH -p yyanggpu1
#SBATCH -q yyanggpu1

#SBATCH --gres=gpu:1

#SBATCH -t 0-100:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

eval "$(conda shell.bash hook)"
conda activate torchreid
module purge
module load cuda/9.1.85


cd /home/ywan1053/reid-strong-baseline-master/
python3 tools/compute_gradcam.py --config_file='grad_cam/mobilenetv2_53_DSP_VCH_DNL_SGD/config.yml'