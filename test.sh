export CUDA_VISIBLE_DEVICES=0

model_name=MDFM_AdaKAN

seq_len=96
e_layers=4
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=16
train_epochs=15
patience=5
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/\
  --data_path ETTh1.csv \
  --model_id ${model_name}_ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 32 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window