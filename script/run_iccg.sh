
full_time=$(date "+%Y-%m-%d-%H-%M-%S")
nohup ./sub_iccg.sh "${full_time}" > "../output/out/ICCG/ICCG_BAAS_${full_time}.out" 2>&1 &