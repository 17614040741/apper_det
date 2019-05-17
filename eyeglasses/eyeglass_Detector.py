# -*- coding: utf-8 -*-
"""
https://github.com/wtx666666/realtime-glasses-detection

Created on Thu Aug 16 22:20:37 2018
@author: James Wu
modified on 2019-4-12 by Gao

*Requirements：
python 3.6
numpy 1.14.0
opencv-python 3.4.0
dlib 19.7.0
"""

import os

import cv2
import dlib
import numpy as np


#==============================================================================
#   1.landmarks格式转换函数 
#       输入：dlib格式的landmarks
#       输出：numpy格式的landmarks
#==============================================================================   
def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

#==============================================================================
#   2.绘制回归线 & 找瞳孔函数
#       输入：图片 & numpy格式的landmarks
#       输出：左瞳孔坐标 & 右瞳孔坐标
#==============================================================================   
def get_centers(img, landmarks):
    # 线性回归
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) #画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

#==============================================================================
#   3.人脸对齐函数
#       输入：图片 & 左瞳孔坐标 & 右瞳孔坐标
#       输出：对齐后的人脸图片
#============================================================================== 
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

#==============================================================================
#   4.是否戴眼镜判别函数
#       输入：对齐后的人脸图片，阈值
#       输出：判别值(1/0)
#============================================================================== 
def judge_eyeglass(img, threshold):
    img = cv2.GaussianBlur(img, (11,11), 0) #高斯模糊

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #y方向sobel边缘检测
    sobel_y = cv2.convertScaleAbs(sobel_y) #转换回uint8类型
    #cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y #边缘强度矩阵
    
    #Otsu二值化
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #计算特征长度
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 1/2)
    w = np.int32(d * 2/7)
    h = np.int32(d * 3/4)
    
    roi = thresh[y:y+h, x:x+w] #提取ROI
    measure = sum(sum(roi/255)) / float(np.shape(roi)[0] * np.shape(roi)[1])#计算评价值
    
    #cv2.imshow('roi',roi)
    # print("判决分数："+str(measure))
    
    #根据评价值和阈值的关系确定判别值
    if measure > threshold:#阈值可调，经测试在0.11左右
        judge = 1
    else:
        judge = 0
    return judge

#==============================================================================
#   **************************调用函数***********************************
#==============================================================================
def get_eyeglassRes(img_path, threshold):
    predictor_path = "././eyeglasses/data/shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()#人脸检测器detector
    predictor = dlib.shape_predictor(predictor_path)#人脸关键点检测器predictor
    img = cv2.imread(img_path)
    # 读取视频帧
    # _, img = cap.read()

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    rects = detector(gray, 1)

    # 对每个检测到的人脸进行操作
    for i, rect in enumerate(rects):

        # 检测并标注landmarks
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)

        # 线性回归
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

        # 人脸对齐
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

        # 判断是否戴眼镜
        judge = judge_eyeglass(aligned_face, threshold)
        return judge

def eye_detect(image_dir_path):
    glass_out=[]
    threshold = 0.18 #判断是否佩戴眼镜的阈值
    for face_dir in os.listdir(image_dir_path):
        # 对文件夹下面的所有图片进行眼镜判别，求众数作为输出
        if str(face_dir).endswith('.DS_Store'):continue
        labels = []
        for image in os.listdir(image_dir_path+face_dir):
            judge = get_eyeglassRes(image_dir_path+face_dir+'/'+image, threshold)
            if judge == None:
                continue
            labels.append(judge)
        if labels==None or len(labels)==0:
            labels.append(1000)
        counts = np.bincount(labels)
        mode = np.argmax(counts)
        glass_out.append((face_dir,mode))
    return glass_out


if __name__ == '__main__':
    # img_dic_path = './eyeglasses/1/'  #输入图片目录地址
    # threshold = 0.18 #判断是否佩戴眼镜的阈值
    # for image in os.listdir(img_dic_path):
    #     print(img_dic_path+image)
    #     #****************************************************************************
    #     # 输入：图片地址，阈值
    #     # 输出：判别值(1/0)
    #     judge = get_eyeglassRes(img_dic_path+image, threshold)
    #     #****************************************************************************
    #     print("是否佩戴眼镜：" + str(judge))
    image_dir_path='./face_shape_classifier_api/demo_output/'
    glass_out = eye_detect(image_dir_path)
    print(glass_out)
