CUDA_VISIBLE_DEVICES=0 nohup python -u feasibility_gt_label.py  --log_dir log/feasibility_gt_label_cifar10/ --is_test_dataset False --clip_backbone ViT-B/16 --dataset cifar10 --num_centroids 1  --cosine_sim True &> nohups/feasibility_gt_label_cifar10.txt &


