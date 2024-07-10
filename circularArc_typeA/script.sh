#!/bin/bash
set -e

# Compile the program
make
# Number of jobs to run
end=$1

# Get the name of the current directory
current_dir=$(basename "$PWD")

# Capture start time
start_time=$(date +%s)

rm -f points*
rm -f run.sh.{o,e}*
rm -f core.*

# Remove the file with the name based on the current directory if it exists
if [ -f "${current_dir}_data.txt" ]; then
    rm "${current_dir}_data.txt"
fi

# Check if the output directory exists and remove files if they exist
if [ -d "output" ]; then
    if [ "$(ls -A output)" ]; then
        rm output/*
    else
        echo "Output directory is empty."
    fi
else
    mkdir -p output
fi

# Submit the jobs
for (( i=0; i<end; i++ ))
do
    qsub run.sh $1 $i
    sleep 3s
done

# Wait for all jobs to finish
while [ $(ls | grep "point" | wc -l) -ne $((end)) ]
do
    echo "Waiting for all jobs to finish... $(ls | grep "point" | wc -l)"
    sleep 60s
done

# Move output files to the output directory
mkdir -p output
mv point* run.sh.{o,e}* output

# Concatenate all point files into data.txt with the directory name
cd output
cat point* > "${current_dir}_data.txt"

# Move the final points.txt to the parent directory
mv "${current_dir}_data.txt" ../

# Capture end time
end_time=$(date +%s)

# Calculate elapsed time
elapsed_time=$((end_time - start_time))

# Convert elapsed time to hours, minutes, and seconds
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

# Format the elapsed time
formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

# Print and store the elapsed time
echo "Total time used: $formatted_time (hh:mm:ss)"
echo "Total time used: $formatted_time (hh:mm:ss)" > "${current_dir}_timeinfo.txt"

# Move the time file to the parent directory
mv "${current_dir}_timeinfo.txt" ../

