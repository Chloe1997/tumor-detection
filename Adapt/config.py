import os
# set yout work root path
work_root_path = os.path.join('/work', os.listdir('/work')[0])

def read_file_list(filename):
    file_list = []
    try:
        with open(filename, 'r') as f:
            for n in f:
                file_list.append(n.strip())
        return file_list
    except:
        print('[ERROR] Read file not found' + filename)

# +
train_source_list_path = '../train_source_list.txt'
# train_target_list_path = '../train_target_list.txt'
# val_list_path = '../KVGH_val.txt'

train_target_list_path = '../3D_list.txt'
val_list_path = '../val_3D_list.txt'
# -

config = {
    "data_pkl_path": f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID',
    "3D_data_pkl_path": f'{work_root_path}/3D_DATA/PROCESSED_DATA/CASE_UUID',
    "domain":[False, True], # source, target is 3D or not
    "project_name":'DCG-PTM',
    "dis_type": 'l2', # cosine or l2
    "fet_dim": 2,
    "run_name":"NCKU2KVGH_0829",
    "train_source_list": read_file_list(train_source_list_path),
    "train_target_list": read_file_list(train_target_list_path),
    "val_list": read_file_list(val_list_path),
    "seed":2022,
    "num_class": 2,
    "stride_size":128,
    "patch_size":256,
    "train_batch_size" : 16,
    "val_batch_size" : 16,
    "lr": 0.001,
    "te_lr": 0.001,
    "max_iterator_num": 300001,
    'evaluation_steps':500,
    "print_freq" :500,
    'consistency_rampup':5000,
    'confidence_thresh':0.8,
    'center_freq':100,
    'mas':100,
    'pc_weight':0.1,
    'center_alpha':0.05,
    "log_path": "../../LOG/logs/",
    "checkpoint_path":"best_weight.txt",
    "is_train": True,
    "pretrain_path": "../../LOGS/NCKU_center2_29/best_model.pt",
    "log_string_dir": "../LOGS/NCKU23D/",
}




