import numpy as np
import cv2
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array




face_cascade = cv2.CascadeClassifier('./beard/haarcascade_frontalface_default.xml')
model = load_model('./beard/beard detection.h5')

def beard_detect(img_dir_path):
    i = 0
    beard_out = []
    for face_dir in os.listdir(img_dir_path):
         # 对文件夹下面的所有图片进行胡须判别，并将众数输出
        labels = []
        if str(face_dir).endswith('.DS_Store'):continue
        for img in os.listdir(img_dir_path+face_dir):
            # Capture frame-by-frame
            img_path = img_dir_path+face_dir+'/' + img
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            faces = face_cascade.detectMultiScale(frame, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = frame[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (64, 64))
                roi_beard = roi_gray[35:90, 7:55]
                roi_beard = cv2.resize(roi_beard, (28, 28))
                roi_beard_array = img_to_array(roi_beard)
                roi_beard_array = roi_beard_array / 255
                roi_beard_array = np.expand_dims(roi_beard_array, 0)
                prediction = model.predict(roi_beard_array)
                if prediction[0][0] < 0.5:
                    # answer = img +' '+ 'Beard'
                    answer = 1
                    labels.append(answer)
                else:
                    # answer = img +' ' +'Non Beard'
                    answer = 0
                    labels.append(answer)
        if labels==None or len(labels)==0:
            labels.append(1000)
        counts = np.bincount(labels)
        mode = np.argmax(counts)
        beard_out.append((face_dir,mode))
    return beard_out

def beard_detect_image(img_path):
    # Capture frame-by-frame
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    faces = face_cascade.detectMultiScale(frame, 1.2, 5)
    answer = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_beard = roi_gray[35:90, 7:55]
        roi_beard = cv2.resize(roi_beard, (28, 28))
        roi_beard_array = img_to_array(roi_beard)
        roi_beard_array = roi_beard_array / 255
        roi_beard_array = np.expand_dims(roi_beard_array, 0)
        prediction = model.predict(roi_beard_array)
        if prediction[0][0] < 0.5:
            answer = 1
    return answer

if __name__=='__main__':
    image_dir_path='./face_shape_classifier_api/demo_output/'
    beard_out = beard_detect(image_dir_path)
    print(beard_out)