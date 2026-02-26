

full_time=$(date "+%Y-%m-%d-%H-%M-%S")
nohup ./sub_ilu.sh "${full_time}" > "../output/out/ILU0/ILU0_BAAS_${full_time}.out" 2>&1 &