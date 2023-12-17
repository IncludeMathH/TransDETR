import os.path as osp
import os
import numpy as np
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
import json

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return np.array(new_box)


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
            
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
#     print(v)
#     print(anchor)
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_rotate(box):
    """
    Represents the minimum circumscribed rectangle as the top-left coordinate + length and width + rotation angle.
    The returned rectangle coincides with the input rectangle after theta angle is rotated.

    Input:
        box <np.array: (8, )>: x1,y2...,x3,y3, denotes the four point coordinates of the min area rectangle of an object
    Output:
        rect <np.array: (4, )>: topLeft_x, topLeft_y, width, height. 
        theta <float>: The returned rectangle coincides with the input rectangle after theta angle is rotated.
    """
    theta = find_min_rect_angle(box)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min, x_max-x_min, y_max-y_min]), theta
    
def getBboxesAndLabels_icd13(height, width, annotations):
    """
    Input:
        height <int>: the height of the frame in video
        width <int>: the width of the frame in video
        annotations: the anno of a frame
    Output:
        bboxes_box <np.array, (n, 4)>: (topLeft_x, topLeft_y, w, h), the rotated bbox of objects in a frame. 
        IDs <np.array, (n, )>: ID of the target that appears in a frame. n denotes the number of objects.
        rotates <np.array, (n, )>: The rotate angle of rotated bbox of the objects.
        words <List[str], (n, )>: The transcription of the objects.
        bboxes <np.array, (n, 4)>: (topLeft_x, topLeft_y, w, h). Horizontal bounding boxes.
    """
    bboxes = []
    IDs = []
    rotates = []
    bboxes_box = []
    words = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        # annotation is the anno of an object in the frame
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points_rotate = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points_rotate = cv2.boxPoints(points_rotate).reshape((-1))
        rotate_box, rotate = get_rotate(points_rotate)      # Another representation of minAreaRect
        
        x, y, w, h = cv2.boundingRect(points.reshape((4, 2)))      # topLeft_x, topLeft_y, width, height
        box = np.array([x, y, w, h])        
        
        quality = annotation.attrib["Quality"]
        Transcription = annotation.attrib["Transcription"]
        if quality == "LOW":
            Transcription = "###"   
        elif "?" in Transcription or "#" in Transcription:
            Transcription = "###"   
            
        words.append(Transcription)    
        bboxes_box.append(rotate_box)
        IDs.append(annotation.attrib["ID"])
        rotates.append(rotate)
        bboxes.append(box)

    if bboxes:
        bboxes_box = np.array(bboxes_box, dtype=np.float32)
        bboxes = np.array(bboxes, dtype=np.float32)
        # filter the coordinates that overlap the image boundaries.
        bboxes_box[:, 0::2] = np.clip(bboxes_box[:, 0::2], 0, width - 1)
        bboxes_box[:, 1::2] = np.clip(bboxes_box[:, 1::2], 0, height - 1)
        IDs = np.array(IDs, dtype=np.int64)
        rotates = np.array(rotates, dtype=np.float32)
    else:
        bboxes_box = np.zeros((0, 4), dtype=np.float32)
        bboxes = np.zeros((0, 4), dtype=np.float32)
        # polygon_point = np.zeros((0, 8), dtype=np.int)
        IDs = np.array([], dtype=np.int64)
        rotates = np.array([], dtype=np.float32)
        words = []

    return bboxes_box, IDs, rotates, words, bboxes

def parse_xml(annotation_path, image_path):
    """
    Input:
        annotation_path <str>: The dir of annotation of an video.
        image_path <str>: The dir of an frame of the video.
    Output:
        bboxess <List[np.array], (num_objects, 4)>: The rotated minAreaRects in a video, which are horizonal. 
        IDss <List[np.array], (num_objects, )>: The assigned tracked id of objects.
        rotatess <List[np.array], (num_objects, )>: The rotated angle of minAreaRects.
        wordss <List[List[str]], List[str]>: The transcription of the objects.
        orignial_bboxess <List[np.array], (num_objects, 4)>: The horizonal bboxes of objects.
    """
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    bboxess, IDss, rotatess, wordss, orignial_bboxess = [], [] , [], [], []
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

            
    for _, child in enumerate(root):
        bboxes, IDs, rotates, words, orignial_bboxes = \
            getBboxesAndLabels_icd13(height, width, child)
        bboxess.append(bboxes) 
        IDss.append(IDs)
        rotatess.append(rotates)
        wordss.append(words)
        orignial_bboxess.append(orignial_bboxes)
    return bboxess, IDss, rotatess, wordss, orignial_bboxess

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def gen_data_path(path, data_path_str="./datasets/data_path/RoadText3k.train"):
    """
    regard labels as the standard
    """
    label_path = os.path.join(path, "labels")
    lines = []
    for video_name in os.listdir(label_path):
        frame_path = os.path.join(label_path, video_name)
        print(video_name)
        for i in range(1, len(os.listdir(frame_path))+1):
            frame_real_path = "RoadText3k/images/" + video_name + "/{}.jpg".format(i) + "\n"
            lines.append(frame_real_path)
    write_lines(data_path_str, lines)  

def main(): 
    # path of ground truth of ICDAR2015 video
    from_label_root = "./Data/RoadText3k/roadtext-annotation-fixed.json"
    with open(from_label_root, 'r') as load_f:
        anno_dict = json.load(load_f)

    # path of video frames 
    video_root = './Data/RoadText3k/images'

    # path to generate the annotation
    label_root = './Data/RoadText3k/labels'
    mkdirs(label_root)

    tid_curr = 0
    for video_id, video_anno in tqdm(anno_dict.items()):
        image_path_frame = osp.join(video_root, video_id)
        seq_label_root = osp.join(label_root, video_id)
        mkdirs(seq_label_root)

        ID_list = {}      # 每个视频的track_id都从1开始分配
        for frame_id, frame_anno in video_anno.items():
            lines = []
            # frame_id starts from 1
            label_fpath = osp.join(seq_label_root, '{}.txt'.format(frame_id))
            frame_path_one = osp.join(image_path_frame, "{}.jpg".format(frame_id))
            if not os.path.exists(frame_path_one):
                # if this image doesn't exit
                continue
            
            img = cv2.imread(frame_path_one)
            seq_height, seq_width = img.shape[:2]

            frame_anno = frame_anno['labels']
            if not frame_anno:
                # TODO: there are many mistakes in roadtext anno file, there is some work to add.
                with open(label_fpath, 'w') as f:
                    pass
                    continue
            
            for object_anno in frame_anno:
                track_id = object_anno['id']
                if track_id not in ID_list:
                    tid_curr += 1
                    ID_list[track_id] = tid_curr
                    real_id = tid_curr                  # begin from 1
                else:
                    real_id = ID_list[track_id]

                x1, x2, y1, y2 = object_anno['box2d']['x1'], object_anno['box2d']['x2'], object_anno['box2d']['y1'], object_anno['box2d']['y2']
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                if object_anno['category'] in {'Illegible', 'Non_English_Legible'}:
                    word = '###'
                else:
                    word = object_anno['ocr']

                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f} {}\n'.format(
                real_id, x / seq_width, y / seq_height, (x2 - x1) / seq_width, (y2 - y1) / seq_height, 0, x1, y1, x2, y2, word)
                lines.append(label_str)
                
            write_lines(label_fpath, lines)  
    # ===================================  
    # to generate data_path .txt
    gen_data_path(path="./Data/RoadText3k")

if __name__ == '__main__':
    main()
