#! /bin/bash

#$ -q large.q
#$ -cwd
#$ -V
#$ -l "h_rt=00:02:00"
#$ -o o
#$ -e e

./a.out
