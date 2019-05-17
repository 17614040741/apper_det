'''
整合脸型、眼睛、胡须输出
'''
import os
from face_shape_classifier_api import FSC
from eyeglasses import eyeglass_Detector
import sys
from sys import path
path.append('./beard')
from beard import beard
import csv
import numpy as np
from keras.utils import to_categorical

face_model_path='./face_shape_classifier_api/model/retrained_graph_.pb'
# image_dir_path='./data/test-all_output3/'
output_file = './result-3.csv'

def apper_detect_folder(image_dir_path='./data/test-all_output3/'):
    """按文件夹处理图片，给出众数作为结果"""
    f=FSC.face_shape_classifier(face_model_path)
    face_out=[]
    for face_dir in os.listdir(image_dir_path):
        face_out.append((face_dir,f.classify_dir(image_dir_path+face_dir)))
    glass_out = eyeglass_Detector.eye_detect(image_dir_path)
    beard_out = beard.beard_detect(image_dir_path)
    f.close_sess()
    return face_out,glass_out,beard_out

def apper_detect_folder_main():
    faces,glasses,beards = apper_detect_folder()
    headers = ['folder','face_0','face_1','face_2','face_3','face_4','glass','beard']
    # 排序保证文件夹输出顺序一致
    faces = sorted(faces)
    faces = np.array(faces)
    faces = faces[:,1]
   
    one_hot_faces = [[0]*5 if x is None else to_categorical(x,5) for x in faces]
    #one_hot_faces = [[None]*5 if x is None else to_categorical(x,5) for x in faces]
    one_hot_faces = np.array(one_hot_faces)
    face_0 = one_hot_faces[:,0]
    face_1 = one_hot_faces[:,1]
    face_2 = one_hot_faces[:,2]
    face_3 = one_hot_faces[:,3]
    face_4 = one_hot_faces[:,4]
    glasses = sorted(glasses)
    glasses = np.array(glasses)
    folder = glasses[:,0]
    beards = sorted(beards)
    beards = np.array(beards)
    glasses = glasses[:,1]
    beards = beards[:,1]
    res = np.vstack((folder,face_0,face_1,face_2,face_3,face_4,glasses,beards))
    res = res.T
    with open(output_file,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(res)

