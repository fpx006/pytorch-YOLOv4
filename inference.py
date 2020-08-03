import sys, csv
sys.path.append("/fuel")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from models import *
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect


def get_label_string(img, boxes, ev_label, class_names):
    width = img.shape[1]
    height = img.shape[0]
    max_area = 0
    max_area_id = -1
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        area = (x2-x1)*(y2-y1)
        if area > max_area:
            max_area_id = i
            max_area = area

    if ev_label == '1':
        result_type = 2
    else:
        result_type = 5

    if max_area_id < 0:
        return '',result_type

    box = boxes[max_area_id]
    x1 = np.clip(int((box[0] - box[2] / 2.0) * width), 0, width-1)
    y1 = np.clip(int((box[1] - box[3] / 2.0) * height), 0, height-1)
    x2 = np.clip(int((box[0] + box[2] / 2.0) * width), 0, width-1)
    y2 = np.clip(int((box[1] + box[3] / 2.0) * height), 0, height-1)
    cls_conf = box[5]
    cls_id = box[6]
    cls_name = class_names[cls_id]

    if ev_label == '1':
        if cls_id == 1:
            result_type = 0
        else:
            result_type = 1
    else:
        if cls_id == 0:
            result_type = 3
        else:
            result_type = 4


    if ev_label == '1' and (cls_name == 'car' or cls_name == 'bus' or cls_name == 'truck'):
        return str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+',1', result_type
    else:
        return str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+',0', result_type

def inference_yolov4(is_local=False):

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = sys.argv[4]
        width = int(sys.argv[5])
        namesfile = int(sys.argv[6])
    else:
        print('use default value ')
        #print('  python models.py num_classes weightfile imgfile namefile')
        n_classes = 2
        #WORK_FOLDER = os.path.dirname(os.path.abspath(__file__))
        if is_local:
            WORK_FOLDER = '/fuel/fueling/perception/emergency_detection'
        else:
            WORK_FOLDER = '/mnt/bos/modules/perception/emergency_detection'

        #weightfile = os.path.join(WORK_FOLDER, 'pretrained_model/yolov4.pth')
        weightfile = os.path.join(WORK_FOLDER, 'checkpoints/Yolov4_epoch600.pth')
        #weightfile = 'pretrain_model/yolov4.pth'
        #imgfile = os.path.join(WORK_FOLDER, 'data/dog.jpg')
        height = 320
        width = 320

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    if is_local:
        use_cuda = False
    else:
        use_cuda = True

    if use_cuda:
        model.cuda()

    if namesfile == None:
        if n_classes == 20:
            namesfile = os.path.join(WORK_FOLDER, 'data/voc.names')
        elif n_classes == 80:
            namesfile = os.path.join(WORK_FOLDER, 'data/coco.names')
        elif n_classes == 2:
            namesfile = os.path.join(WORK_FOLDER, 'data/emergency_vehicle.names')
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)

    #image_files = glob.glob(os.path.join(WORK_FOLDER, 'data/emergency_vehicle/images/*.jpg'))
    csv_file = os.path.join(WORK_FOLDER, 'data/emergency_vehicle/val.csv')
    with open(csv_file) as f:
        gt_log = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    output_file = os.path.join(WORK_FOLDER, 'data/emergency_vehicle/tmp.txt')
    f = open(output_file, "w")

    statistics_array = [0,0,0,0,0,0]
    row_id = 0
    for row in gt_log:
        img_file, ev_label = row
        img_path = os.path.join(WORK_FOLDER, 'data/emergency_vehicle/images/', img_file)
        print('processing ', img_path, '...')
        img = cv2.imread(img_path)

        # Inference input size is 416*416 does not mean training size is the same
        # Training size could be 608*608 or even other sizes
        # Optional inference sizes:
        #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        #for i in range(2):  # This 'for' loop is for speed check
                            # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        #plot_boxes_cv2(img, boxes[0], img_path.replace('images', 'predictions'), class_names)

        label_string, result_type = get_label_string(img, boxes[0], ev_label, class_names)
        row = img_file + ' ' + label_string + '\n'
        if label_string != '':
            f.write(row)
        row_id += 1
        print(row_id, '/', len(gt_log),': ', row)

        statistics_array[result_type] += 1

        print('************************************************************************')
        
    f.close()
    tot_ev = statistics_array[0]+statistics_array[1]+statistics_array[2]
    if tot_ev == 0:
        tot_ev = 1
    print('total emergency vehicle ', tot_ev)
    print('number of correct: ', statistics_array[0], '  wrong: ', statistics_array[1], '  unknown: ', statistics_array[2])
    print('ratio of correct: ', statistics_array[0]/tot_ev, '  wrong: ', statistics_array[1]/tot_ev, '  unknown: ', statistics_array[2]/tot_ev)

    tot_gv = statistics_array[3]+statistics_array[4]+statistics_array[5]
    if tot_gv == 0:
        tot_gv = 1
    print('total general vehicle ', tot_gv)
    print('number of correct: ', statistics_array[3], '  wrong: ', statistics_array[4], '  unknown: ', statistics_array[5])
    print('ratio of correct: ', statistics_array[3]/tot_gv, '  wrong: ', statistics_array[4]/tot_gv, '  unknown: ', statistics_array[5]/tot_gv)

if __name__ == "__main__":
    inference_yolov4(is_local=True)