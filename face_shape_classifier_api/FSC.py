# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:59:07 2019

@author: Dell
"""
import tensorflow as tf
import os
import numpy as np

class face_shape_classifier:
    def __init__(self, model_path):
        '''
        初始化
        parma:model_path:模型文件路径
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
    
    
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
  
        
    def classify_image(self,image_path,return_format='label'):
        '''
        对单个头像图片进行分类
        param:image_path 图片输入路径
        param:return_format 返回的格式，'label'返回识别的标签，'Probability'返回各预测结果下的概率
        return: INT or NP.ARRAY
        oblong:0/round:1/ heart:2/square:3/oval:4
        '''
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()        
        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        predictions = self.sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        if return_format=='label':
            return top_k[0]
        elif return_format=='Probability':
            return predictions[0]
    
    
    def classify_dir(self,image_dir_path):
        '''
        对文件夹下面的所有图片进行脸型识别，以判定众数作为输出结果
        param:image_dir_path 图片文件夹路径
        return int
        '''
        labels=[]
        try:
            if os.listdir(image_dir_path):
                for image in os.listdir(image_dir_path):
                    try:
                        out= self.classify_image(image_dir_path+'/'+image)
                        labels.append(out)
                        counts = np.bincount(labels)
                        return np.argmax(counts)
                    except:
                        print('error image format !')
            else:
                pass
        except:
            pass
    def close_sess(self):
        '''
        关闭tf.sess节点
        '''
        self.sess.close()


#调用示例
# model_path='D:/workfile/inception-face-shape-classifier-master/retrained_graph_.pb'

# model_path='./face_shape_classifier_api/model/retrained_graph_.pb'
# f=face_shape_classifier(model_path)
#
# image_dir_path='./face_shape_classifier_api/demo_output/'
# out=[]
# for face_dir in os.listdir(image_dir_path):
#     out.append((face_dir,f.classify_dir(image_dir_path+face_dir)))
#
#
# f.close_sess()

