python3 ./train/train.py --description "full stage" --batch_size 1 --gpu_device '0' \
    --train_ps 512 --val_ps 512 --loss full_loss --arch DuRP --stage full --env _0 \
    --nepoch 400 --T_max 32 --lr 1e-5 --weight_decay 0.0001 --eta_min 1e-6 \
    --PRNet_weights  \
    --IRNet_weights  \
    --save_dir ./experiment/ 
