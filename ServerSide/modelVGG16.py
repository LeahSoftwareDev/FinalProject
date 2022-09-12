from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop

#TODO run modal again on the new datasets

batch_size = 32

train_path='splited data/train'
test_path='splited data/test'
valid_path='splited data/validate'

classes = ['fClef', 'Gclefs', 'keyFlat','keyNatural','keySharp','ornamentMordent','ornamentTrill','repeatDot','rest8th','restHalf','restQuarter','timeSig3','timeSig4','timeSig8','timeSigCommon','other', 'Whole','Quarter','Half','double', 'single']



train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(40,40),
                                            batch_size=batch_size,
                                            shuffle=True)

valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(40,40),
                                            batch_size=batch_size,
                                            shuffle=True)

test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                               classes=classes,
                                               class_mode='categorical',
                                               target_size=(40,40),
                                               batch_size=batch_size,
                                               shuffle=False)


vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=(40,40,3))
type(vgg16_model)
model = Sequential()

for layer in vgg16_model.layers:
  model.add(layer)

type(model)
model.summary()

conv_model = Sequential()

for layer in vgg16_model.layers[:-8]:
  conv_model.add(layer)

conv_model.summary()

for layer in conv_model.layers:
  layer.trainable = False

conv_model.summary()

transfer_layer = model.get_layer('block5_pool')

# define the conv_model inputs and outputs
conv_model = Model(inputs=conv_model.input,
                   outputs=transfer_layer.output)


num_classes = 21

# start a new Keras Sequential model.
new_model = Sequential()

# add the convolutional layers of the VGG16 model
new_model.add(conv_model)

# flatten the output of the VGG16 model because it is from a
# convolutional layer
new_model.add(Flatten())

# add a dense (fully-connected) layer.
# # this is for combining features that the VGG16 model has
# # recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# add a dropout layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test set
new_model.add(Dropout(0.5))

# add the final layer for the actual classification
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'

globals()['_model']=new_model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'],run_eagerly=True)

from tensorflow.python.keras.callbacks import EarlyStopping
# early stopping stops the learning process in the case that the process doesn't progress for 2 epochs
# or in the case of over fitting to the training data over the validation data
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=2,
                   verbose=1,
                   mode='auto')

step_size_train=train_batches.n//train_batches.batch_size
step_size_valid=valid_batches.n//valid_batches.batch_size
new_model.summary()
history = new_model.fit_generator(train_batches,
                        epochs=3,
                        steps_per_epoch=step_size_train,
                        validation_data=valid_batches,
                        validation_steps=step_size_valid,
                        callbacks = [es],
                        verbose=1)

step_size_test=test_batches.n//test_batches.batch_size

result = new_model.evaluate(test_batches, steps=step_size_test)

print("Test set classification accuracy: {0:.2%}".format(result[1]))

saved_model = new_model.save("model_VGG16.h5")
print("Saved model to disk")


