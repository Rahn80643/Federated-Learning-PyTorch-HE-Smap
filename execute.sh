#! /bin/bash


ring_dim=8192
scale_bit=52 # 52
# model="ConvNet"

dataset="cifar100" #"tiny_imagenet" #"cifar100"

model="lenet_DLG" # "efficientnetb0" #"lenet_DLG" # "mobilenet_v3" #"lenet_DLG"
load_smap_dir="" 

s_ratio=0.5
s_method="magnitude"
echo "Running with: --he_lib TenSeal_CKKS_without_flatten --ring_dim $ring_dim --scale_bit $scale_bit --model $model --s_ratio $s_ratio"
python -u main.py --he_lib TenSeal_CKKS_without_flatten --ring_dim "$ring_dim" --scale_bit "$scale_bit" --model $model --dataset $dataset --s_ratio "$s_ratio" --s_method "$s_method" --load_map false --load_smap_dir $load_smap_dir 2>&1 | tee TenSeal_"$model"_"$dataset"_"$ring_dim"_"$scale_bit"_"$s_ratio"_20250129_mag.log