import os

root_data_dir = '../'
dataset = '../../Dataset/amazon_2w'
behaviors = 'amazon_2w_users.tsv'
images = 'amazon_2w_items.tsv'
lmdb_data = 'amazon_images.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'vit'
freeze_paras_before = 0

mode = 'train'
item_tower = 'modal'

epoch = 100
load_ckpt_name = 'None'
pretrained_recsys_model = 'epoch-59.pt'  # vit-sasrec:epoch-59.pt

adapter_type = "lora"
l2_weight_list = [0]
drop_rate_list = [0.1]
batch_size_list = [8]
lr_list = [1e-3]
embedding_dim_list = [64]
fine_tune_lr_list = [1e-3]
adapter_cv_lr_list = [1e-4]  # 4e4
adapter_sasrec_lr_list = [1e-4]  # 4e4
adding_adapter_to_list = ['all']
# None or all
fine_tune_to_list = ['None']
# None or TRUE
finetune_layernorm = "None"
cv_adapter_down_size = 64
# None or True
is_serial = "True"

for adapter_sasrec_lr in adapter_sasrec_lr_list:
    for adapter_cv_lr in adapter_cv_lr_list:
        for adding_adapter_to in adding_adapter_to_list:
            for fine_tune_to in fine_tune_to_list:
                for l2_weight in l2_weight_list:
                    for batch_size in batch_size_list:
                        for drop_rate in drop_rate_list:
                            for lr in lr_list:
                                for embedding_dim in embedding_dim_list:
                                    for fine_tune_lr in fine_tune_lr_list:
                                        label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                                            item_tower, batch_size, embedding_dim, lr,
                                            drop_rate, l2_weight, fine_tune_lr)
                                        run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
                                                 python  -m torch.distributed.launch --nproc_per_node 4 --master_port 1265\
                                                 ../run_adapter.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
                                                 --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                                 --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                                 --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {} --pretrained_recsys_model {}\
                                                --adapter_sasrec_lr {} --adapter_cv_lr {} --adding_adapter_to {} --fine_tune_to {}\
                                                 --finetune_layernorm {} --is_serial {} --cv_adapter_down_size {} --adapter_type {}\
                                                 ".format(
                                            root_data_dir, dataset, behaviors, images, lmdb_data,
                                            mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                            l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                            CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr,
                                            pretrained_recsys_model, adapter_sasrec_lr, adapter_cv_lr, adding_adapter_to
                                            , fine_tune_to, finetune_layernorm, is_serial, cv_adapter_down_size,
                                            adapter_type)
                                        os.system(run_py)
