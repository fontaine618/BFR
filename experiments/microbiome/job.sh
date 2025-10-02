#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bfr_microbiome
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=8:00:00
#SBATCH --array=0-29
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20g
#SBATCH --output=/storage/home/spf5519/work/BFR/experiments/microbiome/logs/microbiome_%A_%a.out
# The application(s) to execute along with its input arguments and options:
module load python/3.11.2
source /storage/home/spf5519/work/BFR/bfr/bin/activate
python -m run.py $SLURM_ARRAY_TASK_ID