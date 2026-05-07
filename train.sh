# python ~/LWCUP/factor_pool/FactorEval.py

CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_5 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label5.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_10 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label10.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_20 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label20.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_40 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label40.log &
CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_60 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label60.log &

# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_5 --sft 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label5_sft.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_10 --sft 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label10_sft.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_20 --sft 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label20_sft.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_40 --sft 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label40_sft.log &
# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_60 --sft 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label60_sft.log &


# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train.py --task_label label_5 --sft True 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/lgbm_logs/train_lgbm_label5_sft.log &

# CUDA_VISIBLE_DEVICES=1 python ~/LWCUP/model/train_DLOB.py 2>&1 | tee /home1/zhuzhoufan/LWCUP/results/train_DLOB.log &