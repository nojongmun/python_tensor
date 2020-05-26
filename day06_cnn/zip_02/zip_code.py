import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
mnist = tf.keras.datasets.mnist
from tensorflow.keras.datasets.cifar10 import load_data
import cv2
import matplotlib.pyplot as plt

# 인풋, 아웃풋, 드롭아웃 확률을 입력받기 위한 플레이스 홀더
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
class Detect:
    def __init__(self):
        pass

    '''
        다음 배치함수를 읽어오기 위한 유틸 함수 정의 
    '''
    @staticmethod
    def next_batch(num, data, labels):
        # num 갯수 만큼 랜덤한 샘플들과 레이블들을 리턴
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    @staticmethod
    def build_CNN_classfier(x):
        # 입력 이미지
        x_image = x
        # 첫번째 컴볼루션 레이어 - 하나의 그레이스케일 이미지를 64개의 특징으로 맵핑한다.
        W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
        b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # 첫번째 pool layer
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 두번째 컴볼루션 레이어 : 32개의 특징들 (feature)을 64개의 특징들을 맵핑한다.
        W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
        b_conv2 = tf.Variable(tf.constant(0.0, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # 두번째 pooling layer.
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 세번째 convolutional layer
        W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        # 네번째 convolutional layer
        W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

        # 다섯번째 convolutional layer
        W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
        b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
        # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
        # 이를 384개의 특징들로 맵핑(maping)합니다.
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
        h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_pred = tf.nn.softmax(logits)
        return y_pred, logits

# Cirar-10을 다운로드 하고 데이터를 불러옴
(x_train, y_train),(x_test, y_test) = load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 'scala' 형태의 레이블 (0 ~ 9)  # 스캐일에서 하나의 값을 추출한 것을 스칼라
y_train_ont_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_ont_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# 컨볼루션 레이어 뉴럴 네트워크(CNN) 그래프 생성
dec = Detect()
y_pred, logits = dec.build_CNN_classfier()

# 크로스 엔트로피를 비용함수로 정의
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

def detect_zipno(fname):
    # 이미지 읽어 들이기
    img = cv2.imread(fname)
    # 이미지 크기 구하기
    h, w = img.shape[:2]
    # 이미지의 오른쪽 윗부분만 추출하기
    img = img[0:h//2, w//3:]
    # 이미지 이진화(컬러 사진을 흑백사진으로 만듬)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    im2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
    # 윤관 검출하기
    cnts = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 추출한 이미지에서 윤괄 추출
    result = []
    for pt in cnts:
        x, y, w, h = cv2.boundingRect(pt)
        # 너무 크거나 작은 부분 제거
        if not (50 < w < 70): continue
        result.append([x, y, w, h])
    # 추출한 윤곽을 위치에 따라 정렬하기
    result = sorted(result, key=lambda  x: x[0])
    # 추출한 윤곽이 너무 가까운 것들 제거하기
    result2 = []
    lastx = -100
    for x, y, w, h in result:
        if(x - lastx) < 10 : continue
        result2.append([x, y, w, h])
        lastx = x
    # 초록색 테두리 출력
    for x, y, w, h in result2:
        cv2.rectangle((img,  (x,y), (x+w, y+h), (0,255,0), 3))
    return result2, img

if __name__ == '__main__':
    # 이미지를 지정해서 우편번호 추출하기
    cnts, img = detect_zipno('./zip_data/hagaki1.png')
    # 결과 출력하기
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.savefig("saved_image.png", dpi=200)
    plt.show()