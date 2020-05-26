import cv2
import matplotlib.pyplot as plt

face_file = "./data/haarcascade_frontalface_alt.xml"
casecade = cv2.CascadeClassifier(face_file)

# 이미지를 읽어서 그레이스케일로 변환하기
img = cv2.imread('./data/girl.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 인식하기
face_list = casecade.detectMultiScale(img_gray, minSize=(150, 150))

# 결과 확인하기
if len(face_list) == 0:
    print('실패')
    quit()

# 인식한 부분 표시하기
for (x, y, w, h) in face_list:
    print('얼굴의 좌표 : ' , x, y, w, h)
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=20)

# 이미지 출력하기
cv2.imwrite('./data/girl_face.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


