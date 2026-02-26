
full_time=$(date "+%Y-%m-%d-%H-%M-%S")
nohup ./sub_trsv.sh "${full_time}" > "../output/out/TRSV/TRSV_BAAS_${full_time}.out" 2>&1 &
