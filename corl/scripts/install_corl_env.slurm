#!/bin/bash
#SBATCH --job-name=install_corl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/install_corl_%j.log

module load cuda/12.6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate corl

cd /projects/u6gd/umar/codes/ULM-R1

# Reinstall torch with CUDA (aarch64 + H100)
conda install pytorch torchvision pytorch-cuda=12.6 -c pytorch -c nvidia -y

# Verify CUDA is visible
python -c "import torch; print('torch version:', torch.__version__); print('cuda version:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

# Build flash-attn for H100 (sm_90)
TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation

echo "Done."
