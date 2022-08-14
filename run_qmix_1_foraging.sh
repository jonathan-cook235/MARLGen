#$ -S /bin/bash
#$ -N episodic-control-array
#$ -cwd
#$ -o $HOME/result-arr-aug14-3.out
#$ -e $HOME/result-arr-aug14-3.err
#$ -l h_rt=12:00:00
#$ -l gpu=true
#$ -R y
#$ -l tmem=32G
#$ -t 1-5


echo "We're Starting!"
hostname
date

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"


PATH_TO_CONDA='/share/apps/miniconda3'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${PATH_TO_CONDA}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${PATH_TO_CONDA}/etc/profile.d/conda.sh" ]; then
        . "${PATH_TO_CONDA}/etc/profile.d/conda.sh"
    else
        export PATH="${PATH_TO_CONDA}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate the correct gpu environment
conda activate MARLGen

# Get to the correct directory
cd $HOME/

# setup wandb
wandb login 8f0ba26a350a67397b8e8abdc7865b1feb4a0b46

# set vulkan environment variables
# export VULKAN_SDK='/vulkan/1.3.216.0/x86_64'
# export PATH="${VULKAN_SDK}/bin:$PATH"
# export LD_LIBRARY_PATH="${VULKAN_SDK}/lib:$LD_LIBRARY_PATH"
# export VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"

export VULKAN_SDK=~/vulkan/1.3.216.0/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d

echo "Running experiment..."
# run experiment with these arguments
python Dissertation/epymarl/src/main_qmix_1_foraging.py 

date
echo "We're Done!"