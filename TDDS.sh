
nohup python -m train_origin --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --dynamics --save_path ./checkpoint/all-dataset-new > out/cifar10_all_data_new.log 2>&1 &

nohup python -m train_dyn --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --lbd 1 --dynamics --save_path ./checkpoint/dyn_alldataset_lbd_1 > out/cifar10_tdds_dyn_lbd=1.log 2>&1 &


nohup python -m train_origin+loss_copy --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --lbd -1 --dynamics --save_path ./checkpoint/cifar10_origin+loss_-1_twop > out/cifar10_origin+loss_lbd=-1_twop.log 2>&1 &

nohup python -m train_origin+loss_copy --gpu 0 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --lbd -1 --dynamics --save_path ./checkpoint/cifar100_origin+loss_-1_twop > out/cifar100_origin+loss_lbd=-1_twop.log 2>&1 &




nohup python -m train_origin+loss --gpu 6 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --lbd -0.1 --dynamics --save_path ./checkpoint/cifar100_origin+loss_-0.1 > out/cifar100_origin+loss_lbd=-0.1.log 2>&1 &





nohup python -m train_now --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --dynamics --save_path ./checkpoint/all-dataset_cifar10 > out/cifar10_all_data.log 2>&1 &


nohup python -m train_now --gpu 2 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --dynamics --save_path ./checkpoint/all-dataset_cifar100 > out/cifar100_all_data.log 2>&1 &



nohup python -m random_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --save_path ./checkpoint/random_0.1 > out/cifar10_random_0.1_2.log 2>&1 &

nohup python -m random_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/random_0.3 > out/cifar10_random_0.3.log 2>&1 &

nohup python -m random_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.5 --dynamics --save_path ./checkpoint/random_0.5 > out/cifar10_random_0.5.log 2>&1 &

nohup python -m random_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 1.0 --dynamics --save_path ./checkpoint/random_1.0 > out/cifar10_random_1.0_2.log 2>&1 &

nohup python -m importance_evaluation --dynamics_path ./checkpoint/cifar10_origin+loss_-1_twop/npy/ --mask_path ./checkpoint/generated_mask_cifar10_origin+loss_-1_twop/ \
    --trajectory_len 30 --window_size 10 --decay 0.9 > out/importance_eval_cifar10_origin+loss_-1_twop.log 2>&1 &



nohup python -m importance_evaluation --dynamics_path ./checkpoint/cifar100_origin+loss_-0.1/npy/ --mask_path ./checkpoint/generated_mask_cifar100_origin+loss_-0.1/ \
    --trajectory_len 30 --window_size 10 --decay 0.9 > out/importance_eval_cifar100_origin+loss_-0.1.log 2>&1 &



nohup python -m importance_evaluation --dynamics_path ./checkpoint/cifar10_origin+loss_-1/npy/ --mask_path ./checkpoint/generated_mask_origin+loss_-1_copy/ \
    --trajectory_len 30 --window_size 10 --decay 0.9 > out/importance_eval_origin+loss_-1_copy.log 2>&1 &



nohup python -m fullg_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.7 --lbd -0.75 --save_path ./checkpoint/fullg > out/fullg_0.7_-0.75_2.log 2>&1 &

nohup python -m fullg_train --data_path ./data --dataset cifar10 --arch resnet18 --epochs 2 --learning_rate 0.1  \
   --batch-size 100 --rate 0.1 --save_path ./checkpoint/fullg > out/fullg.log 2>&1 &


nohup python -m fullg_train_copy --gpu 3 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --lbd 0.5 --save_path ./checkpoint/fullg > out/fullg_0.1_0.5.log 2>&1 &
nohup python -m fullg_train_copy --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --lbd -1 --save_path ./checkpoint/fullg > out/fullg_0.1_0.5.log 2>&1 &


nohup python -m fullg_train_cos --gpu 6 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.7 --lbd 0 --save_path ./checkpoint/fullg > out_cos/fullg_0.7_0_onecy.log 2>&1 &


nohup python -m fullg_dyn_plot --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --lbd -0.5 --save_path ./checkpoint/fullg > out/fullg_dyn_0.1_-0.5_copy.log 2>&1 &

nohup python -m fullg_dyn --gpu 5 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --lbd -0.5 --save_path ./checkpoint/fullg > out/fullg_dyn_0.3_-0.5_copy.log 2>&1 &

nohup python -m fullg_dyn_pc --gpu 2 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --lbd -0.5 --save_path ./checkpoint/fullg > out/fullg_dyn_pc_0.1_-0.5_copy2.log 2>&1 &

nohup python -m fullg_dyn_cntpc --gpu 4 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.7 --lbd -0.8 --save_path ./checkpoint/fullg > out/fullg_dyn_cntpc_0.7_-0.8.log 2>&1 &



nohup python -m fullg_dyn_ed --gpu 6 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --lbd -0.5 --save_path ./checkpoint/fullg > out/fullg_dyn_ed_0.3_-0.5.log 2>&1 &

nohup python -m fullg_dyn_ran --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --random_rate 0.1 --rate 0.1 --lbd -0.5 --save_path ./checkpoint/fullg > out/fullg_dyn_ran_0.1_0.1_-0.5.log 2>&1 &



nohup python -m allg_dyn_ed --gpu 5 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --save_path ./checkpoint/fullg > out/allg_dyn_ed_0.1.log 2>&1 &

nohup python -m allg_dyn_pc --gpu 4 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/fullg > out/allg_dyn_pc_0.3.log 2>&1 &

nohup python -m allg_big_dyn_pc --gpu 6 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/fullg > out/allg_big_dyn_pc_0.3.log 2>&1 &

nohup python -m allg_pc --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/fullg > out/allg_pc_0.3.log 2>&1 &


nohup python -m allce_dyn --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --save_path ./checkpoint/fullg > out/allce_dyn_0.1.log 2>&1 &



nohup python -m allce_dyn_ed --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/fullg > out/allce_dyn_ed_0.3.log 2>&1 &



nohup python -m fullg_new_dyn --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --lbd -0.5 --up_lbd 0.9 --save_path ./checkpoint/fullg > out/EMA_fullg_new_dyn_0.1_-0.5_0.9.log 2>&1 &

nohup python -m fullg_new_dyn --gpu 3 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --lbd -0.5 --up_lbd 0.9 --save_path ./checkpoint/fullg > out/EMA_cifar100_fullg_new_dyn_0.3_-0.5_0.9.log 2>&1 &


nohup python -m allce --gpu 0 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --save_path ./checkpoint/test > out/cifar100_allce_0.1.log 2>&1 &

nohup python -m allce_pc --gpu 4 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/test > out/cifar10_allce_pc_0.3.log 2>&1 &

nohup python -m allce_dyn_pc --gpu 5 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.3 --save_path ./checkpoint/test > out/cifar10_allce_dyn_pc_0.3.log 2>&1 &


nohup python -m allg --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 100 --rate 0.1 --save_path ./checkpoint/test > out/allg_0.1.log 2>&1 &


nohup python -m train_subset_gpu --gpu 1 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 128 --save_path ./checkpoint/cifar10_origin+loss_-1_prune_copy --subset_rate 0.9 --mask_path ./checkpoint/generated_mask_origin+loss_-1_copy/data_mask_win10_ep30.npy --score_path ./checkpoint/generated_mask_origin+loss_-1_copy/score_win10_ep30.npy > out/cifar10_origin+loss_-1_0.9_copy.log 2>&1 &




nohup python -m train_subset_gpu --gpu 0 --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 128 --save_path ./checkpoint/cifar10_origin+loss_-1_prune --subset_rate 0.9 --mask_path ./checkpoint/generated_mask_cifar10_origin+loss_-1_twop/data_mask_win10_ep30.npy --score_path ./checkpoint/generated_mask_cifar10_origin+loss_-1_twop/score_win10_ep30.npy > out/cifar10_origin+loss_twop_-1_0.9.log 2>&1 &


nohup python -m train_subset_gpu --gpu 0 --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1 \
    --batch-size 128 --save_path ./checkpoint/cifar100_origin+loss_0.1_prune --subset_rate 0.9 --mask_path ./checkpoint/generated_mask_cifar100_origin+loss_0.1/data_mask_win10_ep30.npy --score_path ./checkpoint/generated_mask_cifar100_origin+loss_0.1/score_win10_ep30.npy > out/cifar100_origin+loss_0.1_0.9.log 2>&1 &




nohup python -m train_subset --data_path ./data --dataset cifar100 --arch resnet18 --epochs 200 --learning_rate 0.1  \
    --batch-size 32 --save_path ./checkpoint/pruned-dataset-cifar100-origin --subset_rate 0.9 --mask_path ./checkpoint/generated_mask_origin+loss_0/data_mask_win10_ep30.npy --score_path ./checkpoint/generated_mask_origin+loss_0/score_win10_ep30.npy > out/train_subset_100_0.1.log 2>&1 &
