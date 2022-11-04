import PIL.Image
import cv2
import pyvips as vi
import numpy as np
from datetime import datetime
import json
import xml.etree.cElementTree as ET
import sys


# region
def create_polygon_annotation_xml(parent, name, group, contour):
    if group == 'tumor':
        color = "#FF0000"
    elif group == 'normal':
        color = "#008000"
    elif group == 'tumor-verify':
        color = "#FFA500"
    else:
        color = "#0000FF"


    Annotation = ET.SubElement(parent, "Annotation", Name=name, Type="Polygon", PartOfGroup=group, Color=color)
    Coordinates = ET.SubElement(Annotation, "Coordinates")
    #     print(contour)
    for i in range(len(contour)):
        ET.SubElement(Coordinates, "Coordinate", Order=str(i), X=str(contour[i]['x']), Y=str(contour[i]['y']))


## Optimizer xml
def prettyXml(element, indent, newline, level=0):
    # determine whether the sub-elements exist
    if element:
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)

    temp = list(element)

    for subelement in temp:
        # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
            # 对子元素进行递归操作
        prettyXml(subelement, indent, newline, level=level + 1)


def json_to_xml(json_file, xml_file):
    ## open json file
    input_file = open(json_file)
    file = json.load(input_file)
    contours_ = file['annotation']

    root = ET.Element("ASAP_Annotations")
    Annotations = ET.SubElement(root, "Annotations")
    for i in range(len(contours_)):
        create_polygon_annotation_xml(Annotations,
                                      name=str(i + 1),
                                      group=str(contours_[i]['partOfGroup']),
                                      contour=contours_[i]['coordinates'])

    AnnotationGroups = ET.SubElement(root, "AnnotationGroups")

    Group = ET.SubElement(AnnotationGroups, "Group", Name="tumor", PartOfGroup="None", Color="#FF0000")
    ET.SubElement(Group, "Attributes")
    Group = ET.SubElement(AnnotationGroups, "Group", Name="normal", PartOfGroup="None", Color="#008000")
    ET.SubElement(Group, "Attributes")

    # Group = ET.SubElement(AnnotationGroups, "Group", Name="Result", PartOfGroup="None", Color="#0000FF")
    ET.SubElement(Group, "Attributes")
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)

    ## Optimizer xml
    tree = ET.parse(xml_file)
    root = tree.getroot()
    prettyXml(root, '\t', '\n')

    ## show the xml after optimization
    # ElementTree.dump(root)

    tree.write(xml_file, encoding='utf-8', xml_declaration=True)


# endregion

# region
def get_Contours(gray_image):
    ret, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    return contours, hierarchy


def get_contours_depth_list(hierarchy):
    depth = -1
    last_node_parent = -2
    parent_stack = list()
    contours_depth_list = list()
    if hierarchy is None:
        return contours_depth_list
    for i in range(hierarchy.shape[1]):
        if last_node_parent < hierarchy[0][i][3]:
            parent_stack.append(hierarchy[0][i][3])
            last_node_parent = hierarchy[0][i][3]
            depth += 1
        elif last_node_parent == hierarchy[0][i][3]:
            pass
        else:
            while True:
                if hierarchy[0][i][3] == parent_stack.pop():
                    parent_stack.append(hierarchy[0][i][3])
                    last_node_parent = hierarchy[0][i][3]
                    break
                depth -= 1
        contours_depth_list.append(depth)
    return contours_depth_list


def caculate_contour_area(contour, result_mask, detected_patch_area):
    img = np.zeros(shape=result_mask.shape)
    img = cv2.fillPoly(img, [contour], color=1)

    area = int(img.sum()) * detected_patch_area * detected_patch_area
    return area


def create_polygon_annotation(contour, result_mask):
    contour = (contour + 1) * 128
    Coordinates = []
    for i in range(contour.shape[0]):
        Coordinates.append({'x': str(contour[i][0][0]), 'y': str(contour[i][0][1])})
    return Coordinates


def write_polygon_annotation(folder_path, result_contours, result_contours_depth_list, result_mask):
    result = dict()
    Annotations = []
    for i in range(result_contours.__len__()):
        color = '#' + ("0x%06X" % (4294967295 - 200 * result_contours_depth_list[i]))[2:]
        contour_area = str(caculate_contour_area(result_contours[i], result_mask, 128))
        Annotations.append({'name': str(result_contours_depth_list[i]), 'type': "Polygon", 'partOfGroup': 'Result',
                            'area': contour_area, color: color,
                            'coordinates': create_polygon_annotation(result_contours[i], result_mask)})
    result['annotation'] = Annotations
    with open(folder_path, 'w') as f:
        json.dump(result, f)


def mask_2_ASAP_Json(mask, folder_path):
    '''
    Convert mask result to ASAP xml.

    :param enable_evaluation: Enable calculation evaluation.
    :param enable_tumor_candidates: Using TumorCandidates algo. to refine result.
    :return:
    '''

    # Find polygons from result image of classifier.
    result_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
    result_mask = np.where(mask == 255, 255, result_mask)
    result_contours, result_hierarchy = get_Contours(result_mask)
    result_contours_depth_list = get_contours_depth_list(result_hierarchy)

    # Write polygon result annotation
    write_polygon_annotation(folder_path, result_contours, result_contours_depth_list, result_mask)



# endregion


