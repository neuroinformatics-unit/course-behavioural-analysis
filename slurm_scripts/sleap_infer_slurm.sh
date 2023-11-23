#!/bin/bash

#SBATCH -J slp_infer # job name
#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 16 # number of cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH --gres gpu:a4500:1 # request 1 GPU (of specific kind)
#SBATCH -o slurm.%x.%N.%j.out # write STDOUT
#SBATCH -e slurm.%x.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@domain.com

# Load the SLEAP module
module load SLEAP

# Data directory
DATA_DIR=/ceph/scratch/neuroinformatics-dropoff/behav-analysis-course/mouse-EPM
# Path to the video files on which to run inference
VIDEO_1=$DATA_DIR/rawdata/sub-01_id-M708149/ses-01_date-20200317/behav/sub-01_ses-01_task-EPM_time-165049_video.mp4

# SLEAP project directory
SLP_DIR=$DATA_DIR/derivatives/behav/software-SLEAP_project

# Make folder to store predictions, if it doesn't already exist
mkdir -p $SLP_DIR/predictions

# Run the inference command for the first video
sleap-track $VIDEO_1 \
    -m "$SLP_DIR/models/231122_133449.centroid/training_config.json" \
    -m "$SLP_DIR/models/231122_133449.centered_instance/training_config.json" \
    -o "$SLP_DIR/predictions/video-1.predictions.slp" \
    --gpu auto \

# Export the predictions to HDF5 format
sleap-convert \
    -o "$SLP_DIR/predictions/video-1.predictions.analysis.h5" \
    --format analysis \
    "$SLP_DIR/predictions/video-1.predictions.slp"

