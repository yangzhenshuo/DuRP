python3 ./scripts/test_speed.py \
    --val_dir ./datas/SRPG/test \
    --weights ./checkpoints/model_latest.pth \
    --result_dir ./results/SRPG/test \
    --arch DuRP \
    --stage test \
    --gpu_device '0' \
    --mode fp16 \
    --save