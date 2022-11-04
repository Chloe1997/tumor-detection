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

test_list_path = '../test_list.txt'

# -
config = {
    "wsi_root_path": f'{work_root_path}/DATA/RAW_DATA/RAW_IMAGE/',
    "mask_path": f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID/',
    "data_pkl_path": f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID/',
    "patch_size": 256,
    "stride_size":128,
    "test_batch_size" : 64,
    "test_list": read_file_list(test_list_path),
    "num_class": 2,
    "inference_model": 0 , # 0 -> source model, 1-> teahcer model, 2-> student model
    "model_checkpoint_path": "",
    "preprocess_save_path":f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID',
    "log_string_dir": "../LOGS/NCKU2KVGH/",
    "best_weights":"best.ckpt",
    "txt_filename":"../RESUTS/NCKUNCKU2KVGH/"
}


