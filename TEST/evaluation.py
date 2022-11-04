import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
import pyvips
from config_test import config
from Mask2json import mask_2_ASAP_Json, json_to_xml
import cv2

def init_para(case_name):
    slide_path = config['wsi_root_path'] + f'/{case_name}.tif'
    slide = pyvips.Image.new_from_file(slide_path, page=0)
    img_shape = [int(slide.height / config['stride_size']),
                 int(slide.width / config['stride_size'])]
    global show_img_argMax
    show_img_argMax = np.zeros(shape=img_shape)

tile_size = config['stride_size']

def show_testing_img(predictions, x, y):
    # print(label,batch_name,predictions)
    pred = torch.argmax(predictions, 1)
    pred = pred.cpu().numpy()
    global show_img_argMax
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x_t = (x/tile_size).astype(int)
    y_t = (y/tile_size).astype(int)
    show_img_argMax[y_t, x_t] = pred


def get_metric(loss, predictions, labels):
    pre = np.argmax(predictions, 1)
    one_hot_pre = np.zeros((pre.shape[0], 2))
    one_hot_pre[np.arange(pre.shape[0]), pre] = 1

    # --TP--
    label_pos_index = labels[:, 1] == 1
    pre_pos_index = one_hot_pre[:, 1] == 1

    # --TN--
    label_neg_index = labels[:, 0] == 1
    pre_neg_index = one_hot_pre[:, 0] == 1

    P = label_pos_index.sum()
    N = label_neg_index.sum()

    TP = (label_pos_index * pre_pos_index).sum()
    TN = (label_neg_index * pre_neg_index).sum()
    FP = N - TN
    FN = P - TP

    return float(loss), float(P), float(N), float(TP), float(TN), float(FP), float(FN)


def save_metric(loss, P, N, TP, TN, FP, FN):
    accuracy = (TP + TN) / (P + N)

    if P == 0:
        sensitivity = 0.0
    else:
        sensitivity = TP / P

    if N == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = FP / N

    if P + (TP + FP) == 0:
        F1_score = 0
    else:
        F1_score = (2 * TP) / (P + (TP + FP))

    print('\tBatch Loss = %.2f\t Accuracy = %.2f' % (loss, accuracy))
    print('\tSensitivity = %.2f\t FPR = %.2f' % (sensitivity, false_positive_rate))
    print('\tF1 score = %.2f' % (F1_score))


# +
import PIL.Image
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def numpy2vips(a):
    height, width = a.shape
    linear = a.reshape(width * height * 1)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, 1,
                                      dtype_to_format[str(a.dtype)])
    return vi

def save_tif(img_argMax,case_path,save_mask_path):
    slide_path = config['wsi_root_path'] + f'/{case_path}.tif'
    slide = pyvips.Image.new_from_file(slide_path, page=0)
    img_shape = (int(slide.height),
                 int(slide.width))
    img_argMax = cv2.resize(img_argMax, img_shape, interpolation=cv2.INTER_AREA)
    vips_img = numpy2vips(img_argMax)
    vips_img.tiffsave(save_mask_path, compression='deflate', tile=True,
                      bigtiff=True, pyramid=True, miniswhite=False, squash=False)


def save_evaluation(case_path):
    global show_img_argMax
    output_path = config['mask_path'] + case_path + '/'
    o_file = output_path + 'score.txt'


    # GT
    gt_path =  config["mask_path"]+f"{case_path}/{case_path}_mask.tiff"
    img_GT = pyvips.Image.openslideload(gt_path, level=7)
    w, h = img_GT.height, img_GT.width
    region = pyvips.Region.new(img_GT)
    patch = region.fetch(0, 0, h, w)
    show_img_GT = np.ndarray(buffer=patch, dtype=np.uint8, shape=[w, h, img_GT.bands])
    show_img_GT = cv2.cvtColor(show_img_GT, cv2.COLOR_RGBA2GRAY)
    im1 = PIL.Image.fromarray(show_img_GT)
    im1.save(output_path + "img_GT.tiff")


    show_img_argMax *= 255.0 / show_img_argMax.max()
    show_img_argMax = np.uint8(show_img_argMax)
    im1 = PIL.Image.fromarray(show_img_argMax)
    im1.save(output_path + "img_argMax.tiff")

    ##  label_predict
    tn = np.where((show_img_argMax == 0) & (show_img_GT == 0))
    fp = np.where((show_img_argMax == 255) & (show_img_GT == 0))

    tp = np.where((show_img_argMax == 255) & (show_img_GT == 255))
    fn = np.where((show_img_argMax == 0) & (show_img_GT == 255))

    normal_label = len(tn[0]) + len(fp[0])
    tumor_label = len(tp[0]) + len(fn[0])

    normal_predict = len(tn[0]) + len(fn[0])
    tumor_predict = len(tp[0]) + len(fp[0])

    n = normal_label + tumor_label
    p0 = float((len(tn[0]) + len(tp[0])) / n)
    pc = float((normal_label * normal_predict + tumor_label * tumor_predict) / (n * n))
    k = (p0 - pc) / (1 - pc)

    TP_score = len(tp[0])
    TN_score = len(tn[0])
    FP_score = len(fp[0])
    FN_score = len(fn[0])

    IOU_score = float(TP_score / (TP_score + FP_score + FN_score))
    Sen_score = float(TP_score / (TP_score + FN_score))
    F1_score = float(2 * TP_score / (2 * TP_score + FP_score + FN_score))
    kappa_score = k
    delimiter_ = ','

    score_list = []
    score_list.append(case_path)
    score_list.append(TP_score)
    score_list.append(TN_score)
    score_list.append(FP_score)
    score_list.append(FN_score)
    score_list.append(IOU_score)
    score_list.append(Sen_score)
    score_list.append(F1_score)
    score_list.append(kappa_score)

    f_o = open(o_file, "a+")
    for score in score_list:
        f_o.write(str(score) + delimiter_)
    now = datetime.now()
    dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
    f_o.write(dt_str)
    f_o.write("\n")
    f_o.close()
     

    print('IOU Score: ' + str(float(len(tp[0]) / (len(tp[0]) + len(fp[0]) + len(fn[0])))))
    print('Sensitivity: ' + str(float(len(tp[0]) / (len(tp[0]) + len(fn[0])))))
    
    # write to summary txt file
    last_mode = True
    eval_score_path = config['txt_filename']+'.txt'
    f_o = open(eval_score_path,"a+")
    for score in score_list:
        f_o.write(str(score) + delimiter_)
    now = datetime.now()
    dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
    f_o.write(dt_str)
    f_o.write("\n")
    f_o.close()

# -
save_result = False
if save_result:
    # Write mask (thumbnail)
    save_mask_path = config["mask_path"] + f"/{case_path}/{case_path}_result.tiff"
    save_tif(show_img_argMax,case_path,save_mask_path)
    # Write json.
    mask_2_ASAP_Json(show_img_argMax, output_path + case_path +'_result.json')
    json_to_xml(output_path + case_path +'_result.json', output_path + case_path +'_result.xml')
    # save resulting mask
    os.system(f"python Gen_mask.py {case_path}")




