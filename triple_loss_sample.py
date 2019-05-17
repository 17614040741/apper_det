#coding=utf-8

from face_shape_classifier_api import FSC
import os
import csv
from beard import beard
from eyeglasses import eyeglass_Detector

face_model_path='./face_shape_classifier_api/model/retrained_graph_.pb'

def load_triple(image_dir_path='./data/carve'):
    res = {}
    for anchor in os.listdir(image_dir_path):
        if anchor == '.DS_Store': continue
        # for item in os.listdir(image_dir_path + '/' +anchor):
        anchor_path = os.path.join(image_dir_path, anchor)
        tmp = {}
        for item in os.listdir(anchor_path):
            if item == '.DS_Store': continue
            item_path = os.path.join(anchor_path, item)
            tmp[item] = []
            for sample in os.listdir(item_path):
                if sample == '.DS_Store': continue
                sample_path = os.path.join(item_path, sample)
                for sample_file in os.listdir(sample_path):
                    if sample_file == '.DS_Store': continue
                    tmp[item].append(os.path.join(sample_path,sample_file))
                # tmp = {item, [pic for pic in sample]}
            res[anchor] = tmp
    return res


def apper_detect_img(image_path=''):
    """新增接口，用于按图片输出 encoding 结果"""
    file_name = os.path.basename(image_path)
    if file_name == '.DS_Store':pass
    f=FSC.face_shape_classifier(face_model_path)
    face_out = f.classify_image(image_path)
    one_hot_face = [0]*5
    if face_out is not None:
        one_hot_face[face_out] = 1
    threshold = 0.18 #判断是否佩戴眼镜的阈值
    glass_out = eyeglass_Detector.get_eyeglassRes(image_path,threshold)
    if glass_out is None: glass_out = 1000
    beard_out = beard.beard_detect_image(image_path)
    f.close_sess()
    # return file_name,face_out,glass_out,beard_out
    return file_name,one_hot_face,glass_out,beard_out

def dummy():
    header = ['anchor','p/n','file','face_0','face_1','face_2','face_3','face_4','glass','beard']
    triple = load_triple()
    with open('tri_loss_other_features.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for anchor in triple.keys():
            for sample_label in triple.get(anchor).keys():
                for path in triple.get(anchor).get(sample_label):
                    file,faces,glasses,beards = apper_detect_img(path)
                    line = [anchor,sample_label,file,faces,glasses,beards]
                    print(line)
                    writer.writerow(line)

    # anchor,p_or_n,
    # file,faces,glasses,beards = apper_detect_img()


if __name__ == '__main__':
    # triple = load_triple()
    # print(triple)
    dummy()