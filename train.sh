# python ~/LWCUP/factor_pool/FactorEval.py

# CUDA_VISIBLE_DEVICES=0 python ~/LWCUP/model/train.py

CUDA_VISIBLE_DEVICES=0 python ~/LWCUP/model/train_DLOB.py 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/train_DLOB.log &