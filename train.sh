CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --master_addr 127.0.0.1 --master_port 12200 \
    --nproc_per_node 4 --nnodes 1 --node_rank 0 \
    train.py \
    --data_path data/rise/place \
    --aug --aug_jitter --no_project \
    --num_action 16 --voxel_size 0.005 \
    --obs_feature_dim 512 --hidden_dim 512 \
    --nheads 8 --num_encoder_layers 4 --num_decoder_layers 7 \
    --dim_feedforward 2048 --dropout 0.1 \
    --ckpt_dir logs/place \
    --batch_size 240 --num_epochs 1000 --save_epochs 200 --num_workers 24 \
    --seed 233  
