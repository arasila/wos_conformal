#! /bin/bash

#$ -q large.q
#$ -cwd
#$ -V
#$ -l "h_rt=24:00:00"

./vy $1 $2
