#!/bin/bash -l

#Specify the GPU partition
#SBATCH -t 72:00:00
#SBATCH -p gpudacdtbig
#Specify the number of GPUs to be used
#SBATCH --gres=gpu:1
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL
# Define an output file - will contain error messages too
#SBATCH -o train_f100_w100_log.out
# Define job name
#SBATCH -J yahooF100W100
# Insert your own username to get e-mail notifications
#SBATCH --mail-user=e.pitsikalis@liverpool.ac.uk
#SBATCH --mail-type=ALL

conda activate weirules

echo =========================================================
echo SLURM job: submitted date = `date`
date_start=`date +%s`
echo =========================================================
echo Job output begins
echo -----------------
echo $hostname
# $SLURM_NTASKS is defined automatically as the number of processes in the
# parallel environment.
python -u train_frcnn_weirules.py 
echo
echo ---------------
echo Job output ends
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished
date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
