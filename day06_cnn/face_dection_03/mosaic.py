import cv2
import matplotlib.pyplot as plt
def mosaic(img, rect, size):
    # 모자이크 적용할 부분 추출하기
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect =img[y1:y2, x1:x2]
    # 축소하고 확대하기
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)
    # 모자이크 적용하기
    img2 = img.copy()
    img2[y1: y2, x1:x2] = i_mos
    return img2
if __name__ == '__main__':
    img = cv2.imread('./data/cat.jpg')
    mos = mosaic(img, (50, 50, 200, 200), 10)
    plt.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
    plt.show()