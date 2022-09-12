import numpy as np
import cv2

#============================================זיהוי שורות===========================
def lines_detection(imgUrl):
    '''
    the function finds the 5 lines in one musical line' and returns it's positions

    :param imgUrl:
    :return: list of line's starting points
    '''
    linesList=[]
    pt=[]
    pt1=[]
    img = cv2.imread(imgUrl)
    #img = cv2.resize(image, (0,0), fx=2, fy=2)
    height, width, channels = img.shape   #גודל התמונה
    blank_image = np.zeros((height,width,3), np.uint8)  #יצירת תמונה חדשה ריקה שתכיל רק את השורות
    blank_image.fill(255)  #מילוי צבע לבן
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Threshold1 = 150
    Threshold2 = 300
    edges = cv2.Canny(gray,Threshold1,Threshold2,apertureSize = 5)
    rho = 0.05  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 150  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    for line in lines:
        x1,y1,x2,y2 =line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        #cv2.imshow('img', img)
        linesList.append(line)
        pt.append((x1,y1))
        pt1.append((x2,y2))
        # print('('+str(x1)+','+str(y1)+')'+'('+str(x2)+','+str(y2)+')')
        # cv2.waitKey(0)

    # y1List.sort()
    # print(y1List)
    # cv2.imshow('img',img)
    # cv2.imshow('newwhite',blank_image)
    # cv2.waitKey(0)
    # cv2.imwrite('Output/result lines.jpg', img)
    # cv2.imwrite('Output/result lines blank.jpg', img)

    #מיון לפי ערך ה Y (מלמעלה ללמטה)
    pt=sorted(pt, key=lambda k: [k[1], k[0]])
    pt1=sorted(pt1, key=lambda k: [k[1], k[0]])
    print(pt)
    print(pt1)
    # מחזירה שתי רשימות של מיקומי השורות ממוינות לפי סדר הY מלמעלה ללמטה
    return pt,pt1

    # wholeLines=[linesList,pt,pt1]
    # print(wholeLines)