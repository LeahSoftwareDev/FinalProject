import cv2
import os
from PIL import Image
from sklearn.model_selection import train_test_split

#חיתוך מתוך התמונה הכללית
def create_dataset(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0), fx=0.4, fy=0.4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #להפוך את התמונה לשחור לבן
    _,thresh = cv2.threshold(gray,50,300,cv2.THRESH_BINARY_INV) # threshold - קצוות
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    dilated = cv2.dilate(thresh,kernel,iterations = 18) # dilate
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # get contours - חילוץ קווי מתאר
    #הפרמטרים: תמונת מקור,מצב אחזור קווי המתאר,שיטת קירוב קווי המתאר
    #מחזירה: קווי המתאר וההיררכיה
    # cv2.imshow('contours',dilated)
    # cv2.waitKey(0)
    idx =0
    c=0
    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        idx += 1
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        # discard areas that are too large
        # x -= 2
        # y -= 3
        # h+=1
        w+=1
        if h>150 or w>150:
            print('x=' + str(x), 'y=' + str(y), 'w=' + str(w), 'h=' + str(h))
            continue
        # discard areas that are too small
        if h<20 or w<20:
            #print('x=' + str(x), 'y=' + str(y), 'w=' + str(w), 'h=' + str(h))
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print('x='+str(x),'y='+str(y),'w='+str(w),'h='+str(h))
        roi = img[y:y + h, x:x + w]
        c+=1
        # cv2.imshow('img' + str(idx) + '.jpg', roi)
        # cv2.imwrite(r'C:\Users\324863349\Documents\data\1/' + str(c) + '.jpg', roi)

    print(c)
    #output
    cv2.imshow('out',img)
    cv2.waitKey(0)
    #cv2.imwrite('Data_output/_25.jpg', img)
    # #cv2.imwrite('Data_output/notes_with_rectangles.jpg',img)

# create_dataset(r"M:\what-child-is-this-for-piano-intermediate.png")
# create_dataset('Piano_sheets/Wonderwall_LI.jpg')

#עיבוי תמונה
# Importing necessary functions
# from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

def createFolder(directory):
  '''
  function to create folders
  :param directory:
  :return:
  '''
  try:
    if not os.path.exists(directory):
      os.makedirs(directory)
  except OSError:
    print('Error: Creating directory. ' + directory)

def create_model_dirs():
  # create the dataset folder
  createFolder('splited data')
  #create train test validate folders
  createFolder('splited data/train')
  createFolder('splited data/test')
  createFolder('splited data/validate')
  #train
  createFolder('splited data/train/fClef')
  createFolder('splited data/train/Gclefs')
  createFolder('splited data/train/keyFlat')
  createFolder('splited data/train/keyNatural')
  createFolder('splited data/train/keySharp')
  createFolder('splited data/train/ornamentMordent')
  createFolder('splited data/train/ornamentTrill')
  createFolder('splited data/train/repeatDot')
  createFolder('splited data/train/rest8th')
  createFolder('splited data/train/restHalf')
  createFolder('splited data/train/restQuarter')
  createFolder('splited data/train/timeSig3')
  createFolder('splited data/train/timeSig4')
  createFolder('splited data/train/timeSig8')
  createFolder('splited data/train/timeSigCommon')
  createFolder('splited data/train/other')
  createFolder('splited data/train/Whole')
  createFolder('splited data/train/Quarter')
  createFolder('splited data/train/Half')
  createFolder('splited data/train/double')
  createFolder('splited data/train/single')
  #test
  createFolder('splited data/test/fClef')
  createFolder('splited data/test/Gclefs')
  createFolder('splited data/test/keyFlat')
  createFolder('splited data/test/keyNatural')
  createFolder('splited data/test/keySharp')
  createFolder('splited data/test/ornamentMordent')
  createFolder('splited data/test/ornamentTrill')
  createFolder('splited data/test/repeatDot')
  createFolder('splited data/test/rest8th')
  createFolder('splited data/test/restHalf')
  createFolder('splited data/test/restQuarter')
  createFolder('splited data/test/timeSig3')
  createFolder('splited data/test/timeSig4')
  createFolder('splited data/test/timeSig8')
  createFolder('splited data/test/timeSigCommon')
  createFolder('splited data/test/other')
  createFolder('splited data/test/Whole')
  createFolder('splited data/test/Quarter')
  createFolder('splited data/test/Half')
  createFolder('splited data/test/double')
  createFolder('splited data/test/single')
  #validate
  createFolder('splited data/validate/fClef')
  createFolder('splited data/validate/Gclefs')
  createFolder('splited data/validate/keyFlat')
  createFolder('splited data/validate/keyNatural')
  createFolder('splited data/validate/keySharp')
  createFolder('splited data/validate/ornamentMordent')
  createFolder('splited data/validate/ornamentTrill')
  createFolder('splited data/validate/repeatDot')
  createFolder('splited data/validate/rest8th')
  createFolder('splited data/validate/restHalf')
  createFolder('splited data/validate/restQuarter')
  createFolder('splited data/validate/timeSig3')
  createFolder('splited data/validate/timeSig4')
  createFolder('splited data/validate/timeSig8')
  createFolder('splited data/validate/timeSigCommon')
  createFolder('splited data/validate/other')
  createFolder('splited data/validate/Whole')
  createFolder('splited data/validate/Quarter')
  createFolder('splited data/validate/Half')
  createFolder('splited data/validate/double')
  createFolder('splited data/validate/single')

def split_to_train_test():
  X = []   # מערך שיכיל את כל נתיבי התמונות
  base_dir = 'new data'
  for i in os.listdir(base_dir):
    print(i)
    for j in os.listdir(fr'{base_dir}/{i}'):
    # עובר על כל התיקיה וממלא את המערך בנתיבי כל התמונות
      X.append(j)

    # devide the images to train_validate and  test
    dirTrainValidate, dirTest = train_test_split(X, test_size=0.2, random_state=1)
    #save the test picture in the test folder
    for item in dirTest:
      ori = fr'{base_dir}/{i}/{item}'
      dest= fr'splited data/test/{i}/{item}'
      shutil.copy(ori, dest)
    # devide the  train_validate images to train and validate
    dirTrain, dirValidate = train_test_split(dirTrainValidate, test_size=0.125, random_state=1)

    for item in dirTrain:
      ori = fr'{base_dir}/{i}/{item}'
      dest= fr'splited data/train/{i}/{item}'
      shutil.copy(ori, dest)


    for item in dirValidate:
      ori = fr'{base_dir}/{i}/{item}'
      dest= fr'splited data/validate/{i}/{item}'
      shutil.copy(ori, dest)
      X = []

# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
def image_augmantion(path,p):
    datagen = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        brightness_range=(0.5, 1.5))

    # Loading a sample image
    img = load_img(path)
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1,) + x.shape)
    # Generating and saving 5 augmented samples
    # using the above defined parameters.
    i = 1
    for batch in datagen.flow(x, batch_size=1,save_to_dir=p,save_prefix=str(i), save_format='jpg'):
        i += 1
        if i > 6:
            break

# path1=r'C:/Users/324863349/Documents/data/1'
# for dir in os.listdir(path1):
#     print(dir)
#     path=path1+os.path.splitext(dir)[0]
#     i=0
#     for img in os.listdir(path):
#         path_ori = path + '/' + img
#         image_augmantion(path_ori,i,path)
#         i += 1

# for image in os.listdir('C:/Users/324863349/Documents/data/1'):
#     if image.endswith('.jpg'):
#         path='C:/Users/324863349/Documents/data/1/{}'.format(image)
#         newpath='C:/Users/324863349/Documents/data/1/1/'
#         image_augmantion(path,newpath)

# הפונקציה הבאה מעבדת את התמונות לגודל של 224 על 224 עם רקע לבן,
# ומעבירה את התמונות המעובדות לתיקייה 'resized':

def resize_img(path, newpath):
    '''שינוי גודל התמונה'''
    # print('height ' + str(height), 'width ' + str(width))
    # img = cv2.resize(img, (0, 0), fx=0.9, fy=0.9)  #
    # הגדלת גודל התמונה
    img = cv2.imread(path)
    img=cv2.resize(img, (81, 81))
    cv2.imwrite(newpath, img)

def resize_with_white_background(ori_path,dest_path):
    '''
    function to resize the data images

    :param ori_path:
    :param dest_path:
    :return: none
    '''
    img = Image.open(ori_path)

    # resize and keep the aspect ratio
    img.thumbnail((40, 40), Image.ANTIALIAS)

    # add the white background
    img_w, img_h = img.size
    background = Image.new('RGB', (40, 40), (255, 255, 255))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.save(dest_path)

# resize_with_white_background('pict/h.jpg','pict/h1.jpg')
# resize_with_white_background(r"Data_output\Gclefs\G_claf170.jpg",r"C:\Users\324863349\Desktop\img.jpg")
# for dir in os.listdir(r"D:/פרויקט-קוד/project code/Datasets/datasets/datasets"):
#     print(dir)
#     for image in os.listdir(r"D:/פרויקט-קוד/project code/Datasets/datasets/datasets/{}".format(dir)):
#         if image.endswith('.jpg'):
#             img = cv2.imread(r"D:/פרויקט-קוד/project code/Datasets/datasets/datasets/{}/{}".format(dir,image))
#             resize_with_white_background(r"D:/פרויקט-קוד/project code/Datasets/datasets/datasets/{}/{}".format(dir,image), r"D:/פרויקט-קוד/project code/Datasets/datasets/datasets1/{}/{}".format(dir,image))

#
#
# for dir in os.listdir('splited data/train'):
#     print(dir)
#     if dir == 'Whole' or dir == 'Half' or dir == 'Quarter':
#         for image in os.listdir('splited data/train/{}'.format(dir)):
#             if image.endswith('.jpg'):
#                 path='splited data/train/{}/{}'.format(dir,image)
#                 newpath='splited data/train/{}/{}'.format(dir,image)
#                 resize_img(path,newpath)

# for image in os.listdir(r'C:\Users\324863349\Documents\data/1'):
#     if image.endswith('.jpg'):
#         path=r'C:\Users\324863349\Documents\data/1/{}'.format(image)
#         newpath=r'C:\Users\324863349\Documents\data/1/{}'.format(image)
#         resize_with_white_background(path,newpath)