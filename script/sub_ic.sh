#!/bin/sh
mtx_dir="$(cd "$(dirname "$0")"/.. && pwd)/data"     # data dir 
full_time="$1"

input_file="../matrixList.txt"  # testing matrix list 

output_file="../output/log/IC0/IC_BAAS_${full_time}.log"
# clear output file 
> "$output_file"
# the begin time 
echo $full_time

nthread=2   # thread count 
turns=100   # testing turns
export OMP_NUM_THREADS=$nthread
export OMP_PROC_BIND=close
export OMP_PLACES=cores
exec 3< "${input_file}"

while IFS= read -r matrix <&3; do
    # the .mtx file path 
    matrix_path="$mtx_dir/$matrix.mtx"
    echo $matrix
    if [ ! -f "$matrix_path" ]; then
        echo "Warning: Matrix file $matrix_path does not exist, skipping..." >> "$output_file"
        continue
    fi
    ../build/test/IC_Level_Merge $matrix_path $nthread "$full_time" 100 >> "$output_file"
done

echo $(date +"%y-%m-%d %H-%M-%S")
exec 3<&-

