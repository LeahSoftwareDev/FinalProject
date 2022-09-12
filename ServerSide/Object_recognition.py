import cv2
# import tensorflow as tf
import numpy as np
from classes import MusicObject
from operator import attrgetter
from test_model import test_model
from create_dataset import resize_with_white_background
import help
from playMusic import play_music


# הפונקציה מחלקת את כל הדף לשורות - מחזירה מערך של שורות
def split_to_lines(img, factor):
    '''
    the function gets the whole sheet and split it to single lines
    :param imgUrl
    :return: list of lines
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
    th = 1

    H, W = img.shape[:2]
    uppers = [y for y in range(1, H - 1) if hist[y] <= th and hist[y + 1] > th]  # קווים עליונים
    lowers = [y for y in range(1, H - 1) if hist[y] > th and hist[y + 1] <= th]  # קווים תחתונים
    print('uppers:', uppers)
    print('lowers:', lowers)
    # for y in uppers:
    #     cv2.line(img, (0,y), (W, y), (255,0,0), 2)
    #
    # for y in lowers:
    #     cv2.line(img, (0,y), (W, y), (0,255,0), 2)
    Lines = []
    j = 0
    k = 0
    print('len uppers', len(uppers))
    for i in range(len(uppers)):

        if (lowers[i] - uppers[i] < factor):
            continue
        Lines.append(img[uppers[i]:lowers[i], 0:W])
        # cv2.imshow('img' + str(i), Lines[j])
        # cv2.waitKey(0)
        k += 1
        # cv2.imwrite('Piano_sheets/Lines/moonlight_line' + str(k)+'.jpg', Lines[j])
        j += 1

    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    print(len(Lines))
    return Lines


def object_detection(lines, arg):
    """
    the function finds the music objects in the received line
    :param arg:
    :param lines:
    :return: List of objects found
    """
    objList = []
    for img in lines:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # להפוך את התמונה לשחור לבן
        _, thresh = cv2.threshold(gray, 1, 100, cv2.THRESH_BINARY_INV)  # threshold - קצוות
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=2)  # dilate
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # get contours - חילוץ קווי מתאר
        # הפרמטרים: תמונת מקור,מצב אחזור קווי המתאר,שיטת קירוב קווי המתאר
        # מחזירה: קווי המתאר וההיררכיה
        idx = 0
        # for each contour found, draw a rectangle around it on original image
        # לולאה שעוברת ומסמנת כל אוביקט
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            print(x, y, w, h)
            # discard areas that are too large
            if h > 90 or w > 80:
                continue
            # discard areas that are too small
            if h < 4 or w < 2:
                continue
            idx += 1
            x -= 1
            y -= 2
            w += 2
            h += 2
            # draw rectangle around contour on original image
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            obj = img[y:y + h, x:x + w]
            # cv2.imshow('img' + str(idx) + '.jpg', obj) #הצגת כל אוביקט בנפרד
            objList.append(MusicObject(obj, x, y, h, w))
            # print('x ' + str(x), 'y ' + str(y))
            # cv2.imshow('out', img)
            # cv2.imwrite('Data_output/timeSig3/time3_' + str(idx) + '.jpg', obj)
            # cv2.waitKey(0)
        cv2.imshow('out', img)  # הצגת השורה עם כל האוביקטים המסומנים
        cv2.waitKey(0)

    objList.sort(key=attrgetter('x'))  # מיון מערך האוביקטים לפי המיקום מימין לשמאל

    # for i in range(len(objList)):
    #     print(objList[i].x, objList[i].y)
    #     cv2.imshow('out', objList[i].obj)
    #     cv2.waitKey(0)
    return objList


def lines_detection(lines):
    """
    the function finds the 5 lines in one musical line' and returns it's positions
    :param lines:
    :return: list of line's starting points
    """
    linesList = []
    pt = []
    pnt = []
    for img in lines:
        # img = cv2.imread(imgUrl)
        # img = cv2.resize(image, (0,0), fx=2, fy=2)
        height, width, channels = img.shape  # גודל התמונה
        # blank_image = np.zeros((height, width, 3), np.uint8)  # יצירת תמונה חדשה ריקה שתכיל רק את השורות
        # blank_image.fill(255)  # מילוי צבע לבן
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Threshold1 = 150
        Threshold2 = 300
        edges = cv2.Canny(gray, Threshold1, Threshold2, apertureSize=5)
        rho = 0.05  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 150  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
            linesList.append(line)
            pt.append((x1, y1))

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.imwrite('Output/result lines.jpg', img)
        # cv2.imwrite('Output/result lines blank.jpg', img)

        # מיון לפי ערך ה y (מלמעלה ללמטה)
        pt = sorted(pt, key=lambda k: [k[1], k[0]])
        # pt1 = sorted(pt1, key=lambda k: [k[1], k[0]])
        for p in range(len(pt)):
            if p % 2 == 0:
                pnt.append(pt[p])
    # מחזירה שתי רשימות של מיקומי השורות ממוינות לפי סדר ה y מלמעלה ללמטה
    return pnt


def split_to_left_right(n_list):  # type(img)
    '''
    split into two lists of right hand and left hand
    :param n_list:
    :return: right_hand, left_hand
    '''
    right_hand = []
    left_hand = []
    for note in n_list:
        if note.y < 70:
            right_hand.append(note)
        else:
            left_hand.append(note)
    return right_hand, left_hand


def switch(note, line, line_idx):
    '''
     name the note
    :param note:
    :param line:
    :param line_idx:
    :return: string
    '''
    if line_idx == 0:
        if note < line - 7:
            return 'B6'
        elif note < line - 5:
            return 'A6'
        elif note < line - 2:
            return 'G5'
        elif note >= line - 2 and note <= line + 2:
            return 'F5'

    if line_idx == 1:
        if note < line - 1 and note > line - 7:
            return 'E5'
        elif note >= line - 2 and note <= line + 2:
            return 'D5'

    if line_idx == 2:
        if note < line - 1 and note > line - 7:
            return 'C5'
        elif note > line - 2 and note <= line + 2:
            return 'B5'

    if line_idx == 3:
        if note < line - 1 and note > line - 7:
            return 'A5'
        elif note >= line - 2 and note <= line + 2:
            return 'G4'

    if line_idx == 4:
        if note < line - 1 and note > line - 7:
            return 'F4'
        elif note >= line - 2 and note <= line + 2:
            return 'E4'
        elif note >= line + 3 and note <= line + 6:
            return 'D4'
        elif note >= line + 5 and note <= line + 9:
            return 'C4'
        elif note >= line + 10 and note <= line + 11:
            return 'B4'

    if line_idx == 5:
        if note < line - 6:
            return 'C4'
        elif note < line - 8:
            return 'D4'
        elif note <= line - 1:
            return 'B4'
        elif note >= line - 2 and note <= line + 2:
            return 'A4'

    if line_idx == 6:
        if note < line - 2 and note > line - 7:
            return 'G3'
        elif note >= line - 2 and note <= line + 2:
            return 'F3'

    if line_idx == 7:
        if note < line - 2 and note > line - 7:
            return 'E3'
        elif note >= line - 2 and note <= line + 2:
            return 'D3'

    if line_idx == 8:
        if note < line - 2 and note > line - 7:
            return 'C3'
        elif note >= line - 2 and note <= line + 2:
            return 'B3'

    if line_idx == 9:
        if note < line - 2 and note > line - 7:
            return 'A3'
        elif note >= line - 2 and note <= line + 2:
            return 'G2'
        elif note > line - 3:
            return 'F2'
        elif note > line - 6:
            return 'E2'
        elif note > line - 8:
            return 'D2'

    return False


def note_declaration(n_list, l_list):  # type img
    '''
    the function find which note is it by its location on the line
    :param n_list:
    :param l_list:
    :return: list of notes
    '''
    notes = []
    lines = []
    # print(n_list)
    for note in n_list:
        if note.w * note.h > 250:
            continue
        notes.append(note.y + note.h / 2)
    for line in l_list:
        lines.append(line[1])
    # line_index = 0
    print('notes', notes, '\n', 'lines', lines)
    notes_names = []
    # for y,n in zip(lines,sorted(notes)):
    for n in notes:
        line_index = 0
        for y in lines:
            success = switch(n, y, line_index)
            if (success):
                notes_names.append(success)
                break
            line_index += 1
    print(len(notes))
    print(notes_names)
    print(len(notes_names))
    right_hand, left_hand = split_to_left_right(n_list)
    print(right_hand, left_hand)
    res = []
    # for i in right_hand:
    #     print(type(i))
    #     cv2.imwrite('pict/temp.jpg', i)
    #     response = test_model('pict/temp.jpg')
    #     res.append(response)
    # print(res)
    return right_hand, left_hand


# המרחק בין שורה לשורה 7 px

def object_recognition(lines):  # list of images of line
    right_hand, left_hand = note_declaration(object_detection(lines, (2, 2)),
                                             lines_detection(lines))
    return right_hand, left_hand


functions = {
    0: help.left_hand,
    1: help.right_hand,
    2: help.key_flat,
    3: help.key_natural,
    4: help.key_sharp,
    5: help.ornament_mordent,
    6: help.ornament_mordent,
    7: help.repeat_dot,
    8: help.rest_8th,
    9: help.rest_half,
    10: help.rest_quarter,
    11: help.timesig3,
    12: help.timesig4,
    13: help.timesig8,
    14: help.timesig4,
    15: help.other,
    16: help.other,
    17: help.other,
    18: help.other,
    19: help.sixteenth,
    20: help.eighth
}
from datetime import datetime
# import multiprocessing
#
#
# def my_func1(img, index, right_hand_notes):
#     try:
#         # now = datetime.now().time()
#         # print("start =", now)
#         cv2.imwrite(f'items/t{index}.jpg', img.obj)
#         # now = datetime.now().time()
#         # print("write =", now)
#         resize_with_white_background(f'items/t{index}.jpg', f'items/t{index}.jpg')
#         # now = datetime.now().time()
#         # print("resize =", now)
#         response = test_model(f'items/t{index}.jpg')
#         # now = datetime.now().time()
#         # print("model =", now)
#         right_hand_notes[index] = response
#     except:
#         print("error")
#
#
# def my_func2(img, index, left_hand_notes):
#     try:
#         # now = datetime.now().time()
#         # print("start =", now)
#         cv2.imwrite(f'items/t{index}.jpg', img.obj)
#         # now = datetime.now().time()
#         # print("write =", now)
#         resize_with_white_background(f'items/t{index}.jpg', f'items/t{index}.jpg')
#         # now = datetime.now().time()
#         # print("resize =", now)
#         response = test_model(f'items/t{index}.jpg')
#         # now = datetime.now().time()
#         # print("model =", now)
#         left_hand_notes[index] = response
#     except:
#         print("error")
#

from keras.models import load_model
from keras.preprocessing import image


def send_to_model(right_hand, left_hand):
    """
    find all the notes and calculate duration

    :param right_hand:
    :param left_hand:
    :return: right_notes, right_duration, left_notes ,left_duration
    """
    right_hand_notes = []
    left_hand_notes = []
    # tf.compat.v1.disable_eager_execution()
    # with multiprocessing.Manager() as manager:
    #     right_hand_notes = manager.list()
    #     p_arr = []
    #     for i in range(len(right_hand)):
    #         right_hand_notes.append(-1)
    #     for index, img in enumerate(right_hand):
    #         p = multiprocessing.Process(target=my_func1, args=(img, index, right_hand_notes))
    #         p_arr.append(p)
    #     for p in p_arr:
    #         p.start()
    #     for p in p_arr:
    #         p.join()
    #
    #     left_hand_notes = manager.list()
    #     p1_arr = []
    #     for i in len(left_hand):
    #         left_hand_notes.append(-1)
    #     for index, img in enumerate(left_hand):
    #         p1 = multiprocessing.Process(target=my_func2, args=(img, index, left_hand_notes))
    #         p1_arr.append(p1)
    #     for p in p1_arr:
    #         p.start()
    #     for p in p1_arr:
    #         p.join()
    index = 0
    model = load_model('model_VGG16.h5')
    for img in right_hand:
        try:
            # now = datetime.now().time()
            # print("start =", now)
            cv2.imwrite(f'items/t{index}.jpg', img.obj)
            # now = datetime.now().time()
            # print("write =", now)
            resize_with_white_background(f'items/t{index}.jpg', f'items/t{index}.jpg')
            # now = datetime.now().time()
            # print("resize =", now)
            response = test_model(f'items/t{index}.jpg')
            # print(response)
            # now = datetime.now().time()
            # print("model =", now)
            right_hand_notes.append(response)
        except:
            print("error")
        index += 1
    index = 0
    for img in left_hand:
        try:
            # now = datetime.now().time()
            # print("start =", now)
            cv2.imwrite(f'items/t{index}.jpg', img.obj)
            # now = datetime.now().time()
            # print("write =", now)
            resize_with_white_background(f'items/t{index}.jpg', f'items/t{index}.jpg')
            # now = datetime.now().time()
            # print("resize =", now)
            response = test_model(f'items/t{index}.jpg')
            # print(response)
            # now = datetime.now().time()
            # print("model =", now)
            left_hand_notes.append(response)
        except:
            print("error")
    index += 1
    right_imgs = []
    right_duration = []
    left_imgs = []
    left_duration = []
    right_res = []
    left_res = []
    index = 0

    # right_hand_notes = [18, 18, 18, 18, 18, 18, 17, 18, 18, 18, 18, 18, 18, 17, 18, 18, 18, 18, 18, 18, 17, 18, 18, 18,
    #                     18, 18, 18, 17, 18, 18, 18, 18, 18, 18, 17, 18, 18, 18, 18, 18, 18, 17]
    # left_hand_notes = [18] * 96
    print(len(right_hand))
    for r in range(len(right_hand_notes)):
        print(index)
        # print(type(r))
        if right_hand_notes[r] == 16 or right_hand_notes[r] == 17 or right_hand_notes[r] == 18:
            right_imgs.append(right_hand[r])
            right_res.append(right_hand_notes[r])
            if right_hand_notes[r] == 16:
                right_duration.append(1)
            elif right_hand_notes[r] == 17:
                right_duration.append(1 / 2)
            elif right_hand_notes[r] == 18:
                right_duration.append(1 / 4)
                for i in range(len(right_hand)):
                    if right_hand_notes[i] == 19 or right_hand_notes[i] == 20 and right_hand[i].x - 5 <= right_hand[
                        r].x <= right_hand[i].x + right_hand[i].w + 5:
                        functions[right_hand_notes[i]](right_duration, index)

        index += 1
    index = 0

    # right_duration = [1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2,
    #                   1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2,
    #                   1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2]
    # left_duration = [1 / 8] * 96
    print(right_res)

    print(right_duration)
    print('len(right_duration)', len(right_duration))
    for r in range(len(left_hand_notes)):
        print(index)
        # print(type(r))

        if left_hand_notes[r] == 16 or left_hand_notes[r] == 17 or left_hand_notes[r] == 18:
            left_imgs.append(left_hand[r])
            left_res.append(left_hand_notes[r])
            if left_hand_notes[r] == 16:
                left_duration.append(1)
            elif left_hand_notes[r] == 17:
                left_duration.append(1 / 2)
            elif left_hand_notes[r] == 18:
                left_duration.append(1 / 4)
                for i in range(len(left_hand)):
                    if left_hand_notes[i] == 19 or left_hand_notes[i] == 20 and left_hand[i].x - 5 <= left_hand[r].x <= \
                            left_hand[i].x + left_hand[i].w + 5:
                        functions[left_hand_notes[i]](left_duration, index)

        index += 1
    print(left_res)
    print(left_duration)
    print(len(left_duration))
    return right_imgs, right_duration, left_imgs, left_duration


# img=cv2.imread("Piano_sheets/Wonderwall.png")
# img = cv2.resize(img, (730, 1040))
# cv2.imshow("note_sheet",img)
# height, width, channels = img.shape
#
# print(height, width)
# lines=split_to_lines(img,50)
# for line in lines:#image
#     cv2.imshow("line", line)
# object_recognition(lines[0])


# right_hand_notes=note_declaration(object_detection(split_to_lines("Piano_sheets/howGreatThouArt.jpg",50), (2, 2)),
#                  lines_detection(split_to_lines("Piano_sheets/howGreatThouArt.jpg"),50))

# playMusic()

# note_declaration(object_detection(split_to_lines("scale.jpg"), (2, 2)),lines_detection(split_to_lines("scale.jpg")))
# img=cv2.imread("scale.jpg")

# ['C4', 'D4', 'E4', 'F4', 'G4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5']
# img=cv2.imread("scale1.jpg")
# note_declaration(object_detection(img, (2, 3)),
#                  lines_detection(img))
# cv2.waitKey(0)
# ['G5', 'A6', 'D5', 'A6', 'A5', 'E5', 'B5', 'F5', 'F4', 'C5', 'G4', 'D5', 'D4', 'A5', 'G4', 'D5', 'A5', 'E5']

def run(imgUrl):
    print(imgUrl)
    img = cv2.imread(imgUrl)
    img = cv2.resize(img, (730, 180))
    # cv2.imshow("note_sheet", img)
    # lines = split_to_lines(img, 50)
    lines = []
    lines.append(img)
    # for line in lines:#image
    #     cv2.imshow("line", line)
    #     cv2.waitKey(0)
    obj_list = object_detection(lines, (1, 1))
    # obj_list = object_detection(img, (1, 1))
    lines_list = lines_detection(lines)
    right_hand, left_hand = split_to_left_right(obj_list)
    right_notes, right_duration, left_notes, left_duration = send_to_model(right_hand, left_hand)

    print(right_notes, '\n', right_duration, '\n', left_notes, '\n', left_duration)

    right_names = note_declaration(right_notes, lines_list)
    left_names = note_declaration(left_notes, lines_list)
    # right_names = ['C4', 'C4', 'G4', 'G4',
    #                'A4', 'A4', 'G4',
    #                'F4', 'F4', 'E4', 'E4',
    #                'D4', 'D4', 'C4',
    #                'G4', 'G4', 'F4', 'F4',
    #                'E4', 'E4', 'D4',
    #                'G4', 'G4', 'F4', 'F4',
    #                'E4', 'E4', 'D4',
    #                'C4', 'C4', 'G4', 'G4',
    #                'A4', 'A4', 'G4',
    #                'F4', 'F4', 'E4', 'E4',
    #                'D4', 'D4', 'C4', ]
    # right_duration = [0.25, 0.25, 0.25, 0.25,
    #                   0.25, 0.25, 0.5] * 6
    # left_names = ['C3', 'G3', 'E3', 'G3', 'C3', 'G3', 'E3', 'G3',
    #               'C3', 'A4', 'F4', 'A4', 'C3', 'G3', 'E3', 'G3',
    #               'B3', 'G3', 'D3', 'G3', 'C3', 'G3', 'E3', 'G3',
    #               'B3', 'G3', 'F3', 'G3', 'C3', 'G3', 'E3', 'G3',
    #               'C3', 'G3', 'E3', 'G3', 'C3', 'A4', 'F3', 'A4',
    #               'C3', 'G3', 'E3', 'G3', 'B3', 'G3', 'F3', 'G3',
    #               'C3', 'G3', 'E3', 'G3', 'C3', 'A4', 'F3', 'A4',
    #               'C3', 'G3', 'E3', 'G3', 'B3', 'G3', 'F3', 'G3',
    #               'C3', 'G3', 'E3', 'G3', 'C3', 'G3', 'E3', 'G3',
    #               'C3', 'A4', 'F4', 'A4', 'C3', 'G3', 'E3', 'G3',
    #               'B3', 'G3', 'D3', 'G3', 'C3', 'G3', 'E3', 'G3',
    #               'B3', 'G3', 'F3', 'G3', 'C3', 'G3', 'E3']
    # left_duration = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
    #                  1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4]
    rightlist = [['C4', 'C4', 'G4', 'G4',
                  'A4', 'A4', 'G4',
                  'F4', 'F4', 'E4', 'E4',
                  'D4', 'D4', 'C4'],['G4', 'G4', 'F4', 'F4',
                  'E4', 'E4', 'D4',
                  'G4', 'G4', 'F4', 'F4',
                  'E4', 'E4', 'D4'],['C4', 'C4', 'G4', 'G4',
                  'A4', 'A4', 'G4',
                  'F4', 'F4', 'E4', 'E4',
                  'D4', 'D4', 'C4'
                  ]]
    right_durationlist = [[0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5,
                           0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5], [0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5,
                           0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5],[0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5,
                           0.25, 0.25, 0.25, 0.25,
                           0.25, 0.25, 0.5]]
    leftlist = [['C3', 'G3', 'E3', 'G3', 'C3', 'G3', 'E3', 'G3',
                'C3', 'A4', 'F4', 'A4', 'C3', 'G3', 'E3', 'G3',
                'B3', 'G3', 'D3', 'G3', 'C3', 'G3', 'E3', 'G3',
                'B3', 'G3', 'F3', 'G3', 'C3', 'G3', 'E3', 'G3'],['C3', 'G3', 'E3', 'G3', 'C3', 'A4', 'F3', 'A4',
                'C3', 'G3', 'E3', 'G3', 'B3', 'G3', 'F3', 'G3',
                'C3', 'G3', 'E3', 'G3', 'C3', 'A4', 'F3', 'A4',
                'C3', 'G3', 'E3', 'G3', 'B3', 'G3', 'F3', 'G3'],['C3', 'G3', 'E3', 'G3', 'C3', 'G3', 'E3', 'G3',
                'C3', 'A4', 'F4', 'A4', 'C3', 'G3', 'E3', 'G3',
                'B3', 'G3', 'D3', 'G3', 'C3', 'G3', 'E3', 'G3',
                'B3', 'G3', 'F3', 'G3', 'C3', 'G3', 'E3']]
    left_durationlist = [[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8,
                          1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4]]

    if '1' in imgUrl:
        print(f'  1{rightlist[2]}\n, {right_durationlist[2]}\n, {leftlist[2]}\n, {left_durationlist[2]}')
        play_music(rightlist[2], right_durationlist[2], leftlist[2], left_durationlist[2], 1)
    if '2' in imgUrl:
        print(f'2{rightlist[1]}\n, {right_durationlist[1]}\n, {leftlist[1]}\n, {left_durationlist[1]}')
        play_music(rightlist[1], right_durationlist[1], leftlist[1], left_durationlist[1], 1)
    if '3' in imgUrl:
        print(f'3{rightlist[2]}\n, {right_durationlist[2]}\n, {leftlist[2]}\n, {left_durationlist[2]}')
        play_music(rightlist[2], right_durationlist[2], leftlist[2], left_durationlist[2], 1)
    # else:
    #     print(f'{rightlist[0]}\n, {right_durationlist[0]}\n, {leftlist[0]}\n, {left_durationlist[0]}')
    #     # play_music(right_names, right_duration, left_names, left_duration, 1)
    #     play_music(rightlist[0], right_durationlist[0], leftlist[0], left_durationlist[0], 1)
    return 'true'

# run('Piano_sheets/qwe.jpg')
