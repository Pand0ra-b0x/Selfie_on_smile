import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

while True:
    _ ,img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, scaleFactor= 1.1, minNeighbors= 4)
    cnt = 500
    keyPressed = cv2.waitKey(1)
    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        smiles = smileCascade.detectMultiScale(grayImg, scaleFactor= 1.8, minNeighbors= 15)
        for x,y,w,h in smiles:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            print("Image "+ str(cnt) + "Saved")
            path = r'C:\Users\jayan\Desktop\selie on smile' + str(cnt) + '.jpg'
            cv2.imwrite(path, img)
            cnt += 1
            if(cnt >= 503):   
                break
                
    cv2.imshow('live video', img)
    if(keyPressed & 0xFF==ord('q')):
        break

video.release()                                  
cv2.destroyAllWindows()