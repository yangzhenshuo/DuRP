python3 ./train/train.py --description image_reconstruct --batch_size 4 --gpu_device '0' \
    --train_ps 512 --val_ps 512 --loss img_loss --arch IRNet --stage img --env _0\
    --nepoch 400  --lr 0.0001 --weight_decay 0.0001 \
    --save_dir ./experiment/