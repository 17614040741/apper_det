import cv2
import glob
images = glob.glob('C:/Users/Samarth/Desktop/downloads/beard men faces/*.png')
for i in range(len(images)):
    image = cv2.imread(images[i], 0)
    image = cv2.resize(image, (64, 64))
    roi = image[35:90, 7:55]

    cv2.imwrite('C:/Users/Samarth/Desktop/Beard/train/Beard/beard men/' + str(i) + '_beard_men.png', roi)
image = cv2.imread(images[1],0)
image = cv2.resize(image,(64,64))
roi =  image[35:90,7:55]

cv2.imwrite('C:/Users/Samarth/Desktop/Beard/train/Non Beard/''10'+'.png',roi)
cv2.imshow('ii',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()