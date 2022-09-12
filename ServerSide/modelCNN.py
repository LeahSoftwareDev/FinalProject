import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K
K.set_image_dim_ordering('th')

# def loadMyModel():
#     train_path='data/train'
#     test_path='data/test'
#     valid_path='data/validate'
#
#     classes = ['articAccent','coda','dynamicForte', 'dynamicFortePiano', 'dynamicMezzo', 'dynamicMF', 'dynamicPiano', 'fClef', 'fermataAbove', 'Gclefs', 'keyboardPedalPed','keyboardPedalUp','keyFlat','keyNatural','keySharp','ornamentMordent','ornamentTrill','repeatDot','rest8th','restHalf','restQuarter','segno','timeSig3','timeSig4','timeSig8','timeSigCommon','tuplet3']
#
#     batch_size = 32
#
#     dataGen=ImageDataGenerator()
#
#     train_batches = dataGen.flow_from_directory(
#         directory=train_path,
#         classes=classes,
#         class_mode='categorical',
#         target_size=(28, 28),
#         batch_size=batch_size,
#         shuffle=True)
#
#     valid_batches = dataGen.flow_from_directory(
#         directory=valid_path,
#         classes=classes,
#         class_mode='categorical',
#         target_size=(28, 28),
#         batch_size=batch_size,
#         shuffle=True)
#
#     test_batches = dataGen.flow_from_directory(
#         directory=test_path,
#         classes=classes,
#         class_mode='categorical',
#         target_size=(28, 28),
#         batch_size=batch_size,
#         shuffle=False)
#
#     xTrain, yTrain=train_batches.next()
#     xTest, yTest=valid_batches.next()
#     xTrain = xTrain/255
#     xTest = xTest/255
#
#     numClasses = yTest.shape[1]
#     return xTrain, yTrain, xTest, yTest, numClasses
#
# def baselineMyModel(numClasses):
#   # create model
#   model = Sequential()
#
#   model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#
#   model.add(Dropout(0.2))
#
#   model.add(Flatten())
#
#   model.add(Dense(128, activation='relu'))
#   model.add(Dropout(0.2))
#
#   model.add(Dense(numClasses, activation='softmax'))
#
#   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#   model.summary()
#
#   return model
#
# def fitMyModel(model, xTrain, yTrain, xTest, yTest):
#     model.fit(xTrain, yTrain, validation_data=(xTest, yTest),epochs=30,batch_size=32,verbose=2)
#     model.save('modelCNN.h5')
#     print("Saved model CNN to disk")
#
# (xTrain, yTrain, xTest, yTest, numClasses)=loadMyModel()
# model=baselineMyModel(numClasses)
# fitMyModel(model, xTrain, yTrain, xTest, yTest)
#
# # בדיקת תוצאות
# im = r'../images/dataset/size28/test/ל/10.png'
# from skimage import transform
# from PIL import Image
# from keras.models import load_model
# import numpy as np
# MyModel = load_model(r"C:\Users\User\Documents\תיכנות\שיעורים\פרויקטים\ravKavPython\model\myModel_28.h5")
#
# npImage = Image.open(im)
# npImage = np.array(npImage).astype('float32')
# npImage = transform.resize(npImage,(28,28,1))
# npImage = np.expand_dims(npImage, axis=0)
# pred = MyModel.predict(npImage)
# print(pred)
# local=np.argmax(pred, axis=1)
# print(local)
# classes = ['articAccent', 'coda', 'dynamicForte', 'dynamicFortePiano', 'dynamicMezzo', 'dynamicMF', 'dynamicPiano',
#            'fClef', 'fermataAbove', 'Gclefs', 'keyboardPedalPed', 'keyboardPedalUp', 'keyFlat', 'keyNatural',
#            'keySharp', 'ornamentMordent', 'ornamentTrill', 'repeatDot', 'rest8th', 'restHalf', 'restQuarter', 'segno',
#            'timeSig3', 'timeSig4', 'timeSig8', 'timeSigCommon', 'tuplet3']
#

def loadMyModel():
    dataGen=ImageDataGenerator()
    train_batches = dataGen.flow_from_directory(
        directory=pathImageTrain,
        class_mode='categorical',
        target_size=(28, 28),
        batch_size=32,
        # 35964
        color_mode='grayscale')
    valid_batches = dataGen.flow_from_directory(
        directory=pathImageValidate,
        class_mode='categorical',
        target_size=(28, 28),
        batch_size=32,
        # 8991
        color_mode='grayscale')
    xTrain, yTrain=train_batches.next()
    xTest, yTest=valid_batches.next()
    xTrain = xTrain/255
    xTest = xTest/255

    numClasses = yTest.shape[1]
    return xTrain, yTrain, xTest, yTest, numClasses

def baselineMyModel(numClasses):
  # create model
  model = Sequential()

  model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Dropout(0.2))

  model.add(Flatten())

  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(numClasses, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


  model.summary()

  return model

def fitMyModel(model, xTrain, yTrain, xTest, yTest):
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest),epochs=30,batch_size=32,verbose=2)
    model.save('myModel_28.h5')

(xTrain, yTrain, xTest, yTest, numClasses)=loadMyModel()
model=baselineMyModel(numClasses)
fitMyModel(model, xTrain, yTrain, xTest, yTest)

# בדיקת תוצאות
im = r'../images/dataset/size28/test/ל/10.png'

from skimage import transform
from PIL import Image
from keras.models import load_model

MyModel = load_model(r"C:\Users\User\Documents\תיכנות\שיעורים\פרויקטים\ravKavPython\model\myModel_28.h5")

npImage = Image.open(im)
npImage = np.array(npImage).astype('float32')
npImage = transform.resize(npImage,(28,28,1))
npImage = np.expand_dims(npImage, axis=0)
pred = MyModel.predict(npImage)
print(pred)
local=np.argmax(pred, axis=1)
print(local)
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף', 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת']

# print(classes[local[0]])
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import tensorflow as tf
#
# label_code = {'fClef':0, 'Gclefs':1, 'keyFlat':2,'keyNatural':3,'keySharp':4,'ornamentMordent':5,'ornamentTrill':6,'repeatDot':7,'rest8th':8,
#               'restHalf':9,'restQuarter':410,'timeSig3':11,'timeSig4':12,'timeSig8':13,'timeSigCommon':14,'other':15, 'Whole':16,'Quarter':17,'Half':18,'double':19, 'single':20}
#
# label_decode = ['fClef', 'Gclefs', 'keyFlat','keyNatural','keySharp','ornamentMordent','ornamentTrill','repeatDot','rest8th','restHalf','restQuarter','timeSig3','timeSig4','timeSig8','timeSigCommon','other', 'Whole','Quarter','Half','double', 'single']
#
# df = pd.DataFrame(columns = ['path', 'label'])
#
# for dirname, _, filenames in os.walk('new data'):
#     for filename in filenames:
#
#         path = os.path.join(dirname, filename)
#         name = dirname.split('\\')[-1]
#         label = label_code[name]
#         df = df.append({'path' : path, 'label' : label}, ignore_index = True)
#
# print(df.head())
#
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2, random_state = 77)
#
# train.head()
# test.head()
#
# from tensorflow.keras.preprocessing.image import load_img
#
# class dataloader(tf.keras.utils.Sequence):
#
#     def __init__(self, batch_size, img_width, img_height, data):
#         self.bs = batch_size
#         self.h = img_height
#         self.w = img_width
#         self.path = data['path'].values
#         self.label = data['label'].values
#
#     def __len__(self):
#         return len(self.path) // self.bs
#
#     def __getitem__(self, idx):
#         i = idx * self.bs
#         batch_paths = self.path[i: i + self.bs]
#         batch_labels = self.label[i: i + self.bs]
#
#         X = np.zeros((self.bs, self.h, self.w, 3), dtype="float32")
#         y = np.zeros((self.bs, 5), dtype="int32")
#
#         for j in range(self.bs):
#             img = load_img(batch_paths[j], color_mode="rgb", target_size=(self.h, self.w))  # color_mode = "grayscale"
#             img = np.array(img, dtype='float32')
#             img = 1 - img / 127.5
#             X[j] = img
#             y[j, batch_labels[j]] = 1
#         return X, y
#
# train_gen = dataloader(5, 224, 224, train)
#
# test_gen = dataloader(5, 224, 224, test)
# batch1 = test_gen[3]
#
# images = batch1[0]
#
# labels = batch1[1]
#
# print("images in batch = ", images.shape)
# print("labels in batch = ", labels.shape)
#
# for i, j in zip(images, labels):
#     plt.subplots()
#     plt.imshow(i)
#     plt.title(label_decode[np.argmax(j)])
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
#
# model = Sequential()
# model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(Conv2D(4, (4, 4), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(4, (3, 3), activation='relu'))
# model.add(Conv2D(4, (3, 3), activation='relu'))
# model.add(Conv2D(3, (3, 3), activation='relu'))
# model.add(Conv2D(3, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(5,activation='softmax'))
#
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
#
# model.fit(train_gen, epochs=20, validation_data=test_gen, verbose=1)
# model.save("modelCNN_notes.h5")
# print("Saved model to disk")
#
# path = 'Datasets/Quarter/q114.jpg'
#
# img = load_img(path , color_mode = "rgb", target_size=(224, 224)) # (h ,w) color_mode = "grayscale"
# img = np.array(img, dtype = 'float32')
# img = 1-img/127.5
#
# img = img.reshape(1,224,224,3)
#
# y1 = model.predict(img)
# print(y1)