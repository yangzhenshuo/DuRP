python3 ./train/train.py --description polarization_reconstruct --batch_size 1 --gpu_device '0' \
    --train_ps 512 --val_ps 512 --loss pol_loss --arch PRNet --stage pol --env _0 \
    --nepoch 400  --lr 0.0001 --weight_decay 0.0001 --eta_min 1e-5 \
    --save_dir ./experiment/