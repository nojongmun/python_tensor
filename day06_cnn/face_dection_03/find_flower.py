import cv2
import matplotlib.pyplot as plt

img = cv2.imread("flower.jpg")
img = cv2.resize(img, (300, 169))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
im2 = cv2.threshold(gray, 140, 240, cv2.THRESH_BINARY_INV)[1]

# 화면 횐쪽에 변환한 이미지 출력하기
plt.subplot(1, 2, 1)
plt.imshow(im2, cmap='gray')

# 윤곽 검출하기
cnts = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
for pt in cnts:
    x, y, w, h = cv2.boundingRect(pt)

    # 너무 크거나 작은 부분 제거
    if w < 30 or w > 200: continue
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig("flower2.png")
plt.show()