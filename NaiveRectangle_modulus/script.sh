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

# Remove the file with the name based on the current directory if it exists
if [ -f "${current_dir}_modulus.txt" ]; then
    rm "${current_dir}_modulus.txt"
fi

if [ -f "$fm2h_0.txt" ]; then
	rm fp*.txt fm*.txt cfp*.txt cfm*.txt 
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
for (( i=0; i<=end; i++ ))
do
    qsub run1.sh $1 $i
    sleep 3s
    qsub run2.sh $1 $i
    sleep 3s
done

# Wait for ux jobs to finish
while [[ $(ls | grep -c "^fp2h_") -ne $((end+1)) || $(ls | grep -c "^cfp2h_") -ne $((end+1)) ]]
do
    echo "Waiting for calculation to finish... "
    sleep 60s
done

# Move output files to the output directory
mkdir -p output
mv cfm2h_* cfmh_* cfph_* cfp2h_* fm2h_* fmh_* fph_* fp2h_* run1.sh.{o,e}* run2.sh.{o,e}* ./output/

cd output
cat fm2h_* > "fm2h.txt"
cat fmh_* > "fmh.txt"
cat fph_* > "fph.txt"
cat fp2h_* > "fp2h.txt"
cat cfm2h_* > "cfm2h.txt"
cat cfmh_* > "cfmh.txt"
cat cfph_* > "cfph.txt"
cat cfp2h_* > "cfp2h.txt"

mv cfm2h.txt cfmh.txt cfph.txt cfp2h.txt fm2h.txt fmh.txt fph.txt fp2h.txt ../

cd ..
./modulus

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

