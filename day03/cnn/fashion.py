# https://www.tensorflow.org/tutorials/keras/classification?hl=ko
# 초보자 - Keras 를 사용한 ML 기본 사항 - 기본 이이지 분류
# https://parksrazor.tistory.com/97
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class Fashion:

    def modeling(self)->object:
        fashion_mnist = keras.datasets.fashion_mnist
        # 로딩 되자 마자 데이터를 나눈다. ( 지도학습 )
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        '''
        print('트레인 행 : %d, 열 : %d, ' % (train_images.shape[0], train_images.shape[1]))
        print('테스트 행 : %d, 열 : %d, ' % (test_images.shape[0], test_images.shape[1]))
        실행결과
        트레인 행 : 60000, 열 : 28, 
        테스트 행 : 10000, 열 : 28, 
        
        이미지 보기 
        plt.figure()
        plt.imshow(train_images[3])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        '''

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.ylabel([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)  # 이미지 인식할 때 binary 사용
            plt.xlabel(class_names[train_labels[i]])

        # plt.show()
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        """        
        relu ( Recitified Linear Unit 정류한 선형 유닛)        
        미분 가능한 0과 1사이의 값을 갖도록 하는 알고리즘        
        softmax        
        nn (neural network )의 최상위층에서 사용되며 
        classfication 을 위한 function        
        결과를 확률값으로 해석하기 위한 알고리즘        
        """
        # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 모델 학습
        model.fit(train_images, train_labels, epochs=5)
        # 모델 평가
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('테스트 정확도 : ', test_acc)
        # 모델 예측
        predictions = model.predict(test_images)
        print(predictions[0])
        '''
        [1.2175708e-05 2.0410451e-09 1.4151816e-07 1.4517857e-09 2.2553499e-07
            9.1214469e-03 6.2548378e-07 3.9007466e-02 7.0864603e-06 9.5185083e-01]
        '''
        return [predictions, test_labels, test_images]

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
                                        color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    f = Fashion()
    model = f.modeling()
    predictions = model[0]
    test_lables = model[1]
    img = model[2]

    #i = 0
    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1,2,1)
    plot_image(i,predictions,test_lables, img)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_lables)
    plt.show()






