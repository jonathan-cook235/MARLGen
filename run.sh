d#$ -S /bin/bash
#$ -N episodic-control-array
#$ -cwd
#$ -o $HOME/result-arr.out
#$ -e $HOME/result-arr.err
#$ -l h_rt=72:00:00
#$ -l gpu=true
#$ -R y
#$ -l tmem=30G
#$ -t 1-15


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
VULKAN_SDK=~/share/apps/vulkan-1.3.216.0/x86_64
export PATH=$PATH:$VULKAN_SDK/bin
export LD_LIBRARY_PATH=$VULKAN_SDK/lib

# set vulkan environment variable
# VULKAN_SDK="/share/apps/vulkan-1.3.216.0/"

# run experiment with these arguments
python Dissertation/epymarl/src/main.py 

date
echo "We're Done!"