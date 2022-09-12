from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

def test_model(imgUrl):
    # print("check_model")
    # saved_model()
    saved_model = load_model("model_VGG16.h5")
    img = image.load_img(imgUrl, target_size=(40,40))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    classes = ['fClef', 'Gclefs', 'keyFlat','keyNatural','keySharp','ornamentMordent','ornamentTrill','repeatDot','rest8th','restHalf','restQuarter','timeSig3','timeSig4','timeSig8','timeSigCommon','other', 'Whole','Quarter','Half','double', 'single']

    output = saved_model.predict(img)
    # print(output)
    # print(np.argmax(output))
    print(classes[np.argmax(output)])
    return np.argmax(output)

# test_model('pict/h1.jpg')
