
full_time=$(date "+%Y-%m-%d-%H-%M-%S")
nohup ./sub_ic.sh "${full_time}" > "../output/out/IC0/IC0_BAAS_${full_time}.out" 2>&1 &