cd ../
pwd
source ~/.bashrc
conda activate ssl


declare -A alg_dist_dict
alg_dist_dict=([pimodel]="10001" [pseudolabel]="10002" [meanteacher]="10003" [uda]="10004" [mixmatch]="10005" [remixmatch]="10006" [fixmatch]="10007")
alg=$1
num_class=$2
weight_decay=$3
dist_port=${alg_dist_dict[${alg}]}
exp_name=${alg}_cifar${num_class}


#for size in 4000 250 40; do
for size in 250 4000; do
	python ${alg}.py --world-size 1 --rank 0 --amp False --multiprocessing-distributed True --num_labels ${size} --save_name ${exp_name}@${size}_${weight_decay} --weight_decay ${weight_decay} --dataset cifar${num_class} --num_classes ${num_class} --widen_factor 2 --overwrite True --dist-url 'tcp://127.0.0.1:'${dist_port}
done

