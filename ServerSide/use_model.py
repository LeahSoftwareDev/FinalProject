from skimage import transform
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os


np_image = Image.open('pict/tmp1.jpg')
np_image = np.array(np_image).astype('float32')
np_image = transform.resize(np_image, (40, 40, 3))
np_image = np.expand_dims(np_image, axis=0)
batch_size = 32

train_path = 'splited data/train'
test_path = 'splited data/test'
valid_path = 'splited data/validate'

classes = ['fClef', 'Gclefs', 'keyFlat', 'keyNatural', 'keySharp', 'ornamentMordent', 'ornamentTrill', 'repeatDot',
           'rest8th', 'restHalf', 'restQuarter', 'timeSig3', 'timeSig4', 'timeSig8', 'timeSigCommon', 'other', 'Whole',
           'Quarter', 'Half', 'double', 'single']

train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                                         classes=classes,
                                                         class_mode='categorical',
                                                         target_size=(40, 40),
                                                         batch_size=batch_size,
                                                         shuffle=True)

valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                                         classes=classes,
                                                         class_mode='categorical',
                                                         target_size=(40, 40),
                                                         batch_size=batch_size,
                                                         shuffle=True)

test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                                        classes=classes,
                                                        class_mode='categorical',
                                                        target_size=(40, 40),
                                                        batch_size=batch_size,
                                                        shuffle=False)
step_size_test = test_batches.n

new_model = load_model("model_VGG16.h5")
predict = new_model.predict(np_image)
labels = (train_batches.class_indices)
test_batches.reset()
predictions = new_model.predict_generator(test_batches, steps=step_size_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_batches.classes, y_pred))
print('Classification Report')
target_names = ['fClef', 'Gclefs', 'keyFlat','keyNatural','keySharp','ornamentMordent','ornamentTrill','repeatDot','rest8th','restHalf','restQuarter','timeSig3','timeSig4','timeSig8','timeSigCommon','other', 'Whole','Quarter','Half','double', 'single']

print(classification_report(test_batches.classes, y_pred, target_names=target_names))
incorrect = (y_pred != test_batches.classes)

image_paths_test = [os.path.join(test_path, filename) for filename in test_batches.filenames]

misclassified_image_paths = np.array(image_paths_test)[incorrect]

misclassified_image_classes = y_pred[incorrect]
