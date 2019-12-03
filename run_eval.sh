model_path="./models/BaSNet_eval"
output_path="./outputs/BaSNet_eval"
log_path="./logs/BaSNet_eval"
model_file='./BaSNet_model_best.pkl'

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}
