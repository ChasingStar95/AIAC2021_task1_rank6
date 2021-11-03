# Download pretrained weights
# :<<!
cd ./data/bert-base-chinese
wget https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-chinese/resolve/main/config.json
wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt

cd ../../
# !

cd ./code2

# transfer data (Take a long time about 10h)
# python transfer_data.py

# make features
python title_w2v_feat_new.py
python make_seg_feat.py

# train models (largest model takes about 40h on single P40)
CUDA_VISIBLE_DEVICES=0 python final_enhancedrcnn1001.py > log0.txt &
CUDA_VISIBLE_DEVICES=1 python final_esim1011_5.py > log1.txt &
CUDA_VISIBLE_DEVICES=2 python final_gru_lxmert1001.py > log2.txt &
CUDA_VISIBLE_DEVICES=3 python final_inception_lxmert1001.py > log3.txt &
CUDA_VISIBLE_DEVICES=4 python final_lstm_nextvlad_siamese1001.py > log4.txt &
CUDA_VISIBLE_DEVICES=5 python final_lstm1001.py > log5.txt &
CUDA_VISIBLE_DEVICES=6 python final_lxmert1001.py > log6.txt &
CUDA_VISIBLE_DEVICES=7 python final_light_lxmert1001.py > log7.txt &

