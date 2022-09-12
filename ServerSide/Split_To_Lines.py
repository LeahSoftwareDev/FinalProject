import cv2
# הפונקציה מחלקת את כל הדף לשורות - מחזירה מערך של שורות

def split_to_lines(imgUrl):
    '''
    the function gets the whole sheet and split it to single lines

    :param imgUrl
    :return: list o lines
    '''

    img = cv2.imread(imgUrl)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    th = 1
    H,W = img.shape[:2]
    uppers = [y for y in range(1,H-1) if hist[y]<=th and hist[y+1]>th] #קווים עליונים
    lowers = [y for y in range(1,H-1) if hist[y]>th and hist[y+1]<=th] # קווים תחתונים
    print(uppers)  #[3, 62, 154, 176, 235, 240, 263, 419, 422, 451, 544, 591]
    print(lowers)  #[17, 143, 157, 234, 236, 260, 408, 421, 437, 543, 557, 615]
    # for y in uppers:
    #     cv2.line(img, (0,y), (W, y), (255,0,0), 2)
    #
    # for y in lowers:
    #     cv2.line(img, (0,y), (W, y), (0,255,0), 2)
    Lines=[]
    j=0
    k=0
    for i in range(len(uppers)):
        if(lowers[i]-uppers[i]<100):
            continue
        Lines.append(img[uppers[i]:lowers[i],0:W])
        cv2.imshow('img' + str(i), Lines[j])
        cv2.waitKey(0)
        k += 1
        cv2.imwrite('Piano_sheets/Lines/moonlight_line' + str(k)+'.jpg', Lines[j])
        j += 1

    cv2.imshow("result", img)
    cv2.waitKey(0)
    return Lines
