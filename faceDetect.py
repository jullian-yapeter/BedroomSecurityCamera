import cv2
import datetime

file_out = 'intruder_log.txt'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def main(mirror=False):
    previmg = None
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if previmg is None:
            previmg = gray
            trackImg = gray
        else:
            trackImg = cv2.absdiff(previmg,gray)
            previmg = gray
            _,trackImg = cv2.threshold(trackImg, 50, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(trackImg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > 5000:
                    [x, y, w, h] = cv2.boundingRect(c)
                    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=5)
                    cv2.putText(img,'Intruder Alert',org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,255))
                    with open(file_out, "r+") as fout:
                        fout.seek(0, 2)
                        fout.write(str(datetime.datetime.now())+'\n')
                        fout.close()
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            cv2.imwrite('intruder_face.jpg',roi_color)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        cv2.imshow('my tracker', trackImg)
        if cv2.waitKey(1) == 32:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

main(mirror=False)
