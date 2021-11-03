# run mlp blending to get final embeddings
cd ./code2
CUDA_VISIBLE_DEVICES=0 python mlp_blending_5cv_distill.py

