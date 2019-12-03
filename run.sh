model_path="./models/BaSNet"
output_path="./outputs/BaSNet"
log_path="./logs/BaSNet"
seed=-1

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}