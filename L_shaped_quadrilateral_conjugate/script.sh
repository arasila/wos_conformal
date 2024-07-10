#!/bin/bash
set -e

# Compile the program
make


# Number of jobs to run
end=$1

# Get the name of the current directory
current_dir=$(basename "$PWD")

# This removes the file with the name based on the current directory
rm "${current_dir}_data.txt"
rm output/*

# Submit the jobs
for (( i=0; i<=end; i++ ))
do
    qsub run.sh $1 $i
    sleep 3s
done

# Wait for all jobs to finish
while [ $(ls | grep "point" | wc -l) -ne $((end+1)) ]
do
    echo "Waiting for all jobs to finish... $(ls | grep "point" | wc -l)"
    sleep 5s
done

# Move output files to the output directory
mkdir -p output
mv point* run.sh.{o,e}* output

# Concatenate all point files into data.txt with the directory name
cd output
cat point* > "${current_dir}_data.txt"

# Move the final points.txt to the parent directory
mv "${current_dir}_data.txt" ../

