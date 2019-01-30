import os
import random
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random
from keras import preprocessing

width = 96
height = 96

class_names = ['HELLO', 'MY','FRIEND']

camera = cv2.VideoCapture(0)
camera_height = 500
raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []

while(True):
    
    _, frame = camera.read()
    
    
    frame = cv2.flip(frame, 1)

    
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) 
    frame = cv2.resize(frame, (res, camera_height))

   
    cv2.rectangle(frame, (400, 125), (600, 400), (0, 255, 0), 2)

    
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("1"):
        
        raw_frames_type_1.append(frame)
        print('1 key pressed - saved TYPE_1 frame')
    elif key & 0xFF == ord("2"):
        
        raw_frames_type_2.append(frame)
        print('2 key pressed - Saved TYPE_2 frame')
    elif key & 0xFF == ord("3"):
        
        raw_frames_type_3.append(frame)
        print('3 key pressed - Saved TYPE_3 frame')
camera.release()
cv2.destroyAllWindows()


save_width = 399
save_height = 399

for i in range(1, 5):
    name = './data/images_type_{}'.format(i)
    os.makedirs(name, exist_ok=True)
    
for i, frame in enumerate(raw_frames_type_1):
    roi = frame[125+2:400-2, 400+2:600-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('./data/images_type_1/{}.png'.format(i), cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))

for i, frame in enumerate(raw_frames_type_2):
    roi = frame[125+2:400-2, 400+2:600-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('./data/images_type_2/{}.png'.format(i), cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))

for i, frame in enumerate(raw_frames_type_3):
    roi = frame[125+2:400-2, 400+2:600-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('./data/images_type_3/{}.png'.format(i), cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))



def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

images_type_1 = load_images('./data/images_type_1')
images_type_2 = load_images('./data/images_type_2')
images_type_3 = load_images('./data/images_type_3')

X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)

print(X_type_1.shape)
print(X_type_2.shape)
print(X_type_3.shape)


plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(images_type_1))
    plt.imshow(image)
    plt.title('{} image'.format(class_names[0]))
    plt.axis('off')


plt.show()

X = np.concatenate((X_type_1, X_type_2,X_type_3), axis=0)

X = X / 255.
X.shape

from keras.utils import to_categorical

y_type_1 = [0 for item in enumerate(X_type_1)]
y_type_2 = [1 for item in enumerate(X_type_2)]
y_type_3 = [2 for item in enumerate(X_type_3)]

y = np.concatenate((y_type_1, y_type_2,y_type_3), axis=0)

y = to_categorical(y, num_classes=len(class_names))

print(y.shape)

