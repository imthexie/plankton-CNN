#!/bin/bash # set the partition where the job will run
#SBATCH --partition=short
# set the number of nodes
#SBATCH --nodes=1
# set the number of GPU cards to use per node
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
# set name of job #SBATCH --job-name=vggfine
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --sxie@stanford.edu
# run the application
srun -n$SLURM_NTASKS ./build/tools/caffe train -solver ~/plankton-CNN/cs231n/VGG_solver.prototxt -weights snapshots/VGG_ILSVRC_16_layers.caffemodel
