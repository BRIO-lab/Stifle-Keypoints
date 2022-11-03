#!/bin/bash
#SBATCH --job-name=hum_seg_network_etl
#SBATCH --mail-user=nicholasverdugo@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output ./slurm/logs/nick_hum_etl.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

export PATH=/blue/banks/ajensen123/JTML/envs/jtml-env/bin:$PATH

# Run etl.py to create test, train, and verify data set csv. Only need to do this once, unless you want to create new 
# datasets to train off of. For the use of different config files, you can run etl.py once and point the new config
# files to the created csvs to show consistency in data results. 

# etl and train need a config file passed in as an argument.

python scripts/___etl.py hum_config
echo "etl.py complete! running train.py..."
python scripts/__train.py hum_config
