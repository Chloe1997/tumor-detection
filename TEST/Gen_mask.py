import xml.etree.ElementTree as ET
import cv2
import numpy as np
from time import perf_counter
from functools import wraps
import os
import sys
sys.path.append(r'../')
from config import config

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

def timer(function):
    @wraps(function)
    def new_function(*args, **kwargs):
        start_time = perf_counter()
        result = function(*args)
        elapsed = perf_counter() - start_time
        print('Function "{name}" took {time} seconds to complete.'.format(
            name=function.__name__, time=elapsed))
        return result

    return new_function


def numpy2vips(a):
    height, width = a.shape
    linear = a.reshape(width * height * 1)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, 1,
                                      dtype_to_format[str(a.dtype)])
    return vi


def color_group(mask, xml_root, group_name, color=255):
    coordinates = xml_root.findall(
        ".//Annotation[@PartOfGroup='{}']/Coordinates".format(group_name))

    for regions in coordinates:
        points = []
        for region in regions:
            x = float(region.attrib['X'])
            y = float(region.attrib['Y'])
            points.append([x, y])
        if len(points):
            pts = np.asarray([points], dtype=np.int32)
            cv2.fillPoly(mask, pts, color=color)

    return mask


@timer
def genMask(slide_path, xml_path, save_mask_path):
    slide = pyvips.Image.new_from_file(slide_path)
    mask = np.zeros((slide.height, slide.width), dtype=np.uint8)

    xml_root = ET.parse(xml_path)
    mask = color_group(mask, xml_root, 'tumor', 255)
    mask = color_group(mask, xml_root, 'normal', 0)
    vips_img = numpy2vips(mask)
    vips_img.tiffsave(save_mask_path, compression='deflate',tile=True,
                      bigtiff=True, pyramid=True, miniswhite=False, squash=False)


if __name__ == '__main__':
    import pyvips
    # python gen_mask.py name.mrxs
    slide_name = sys.argv[1]
    uuid = slide_name.split('.')[0]
    slide_path = config["wsi_root_path"] + f"/{slide_name}.tif"
    xml_path = config["preprocess_save_path"] + f"/{uuid}/{uuid}_result.xml"
    save_mask_path = config["preprocess_save_path"] + f"/{uuid}/{uuid}_result.tiff"
    print('[INFO] start converting '+uuid+' annotation into mask ...')
    genMask(slide_path, xml_path, save_mask_path)
    print('[INFO] convert '+uuid+' done')
