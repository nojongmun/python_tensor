import cv2

# 웹 카메라로부터 입력받기
cap = cv2.VideoCapture(0)
while True:
    # 카메라 이미지 읽어 들이기
    _, frame = cap.read()
    # 이미지 축소
    frame = cv2.resize(frame, (500, 300))
    cv2.imshow('Camera', frame)
    k = cv2.waitKey(1) # 1분간 대기
    if k == 27 or k == 13:
        break  # ESC, ENTER

cap.release() # 카메라 해제
cv2.destroyAllWindows() # 윈도우 제거

