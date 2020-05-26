# https://parksrazor.tistory.com/82

import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import array, zeros_like
class ColorCat:
    # 없으면 지도학습을 하지 않음
    # def __init__(self):
    #     pass
    def show(self):
        # No module named 'PIL' => pip install pillow
        image_path = tf.keras.utils.get_file('cat.jpg', 'http://bit.ly/33U6mH9')
        image = plt.imread(image_path)
        titles = ['RGB Image', 'Red channel', 'Green channel', 'Blue channel']
        # cmaps = [None, plt.cm.Reds_r, plt.cm.Greens_r, plt.cm.Blues_r]
        colors = range(-1, 3)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3))
        objs = zip(axes, titles, colors)
        for ax, title, color in objs:
            ax.imshow(self.channel(image, color))
            ax.set_title(title)
            ax.set_xticks(())
            ax.set_yticks(())
        plt.show()

    @staticmethod
    def channel(image, color):
        if color not in (0, 1, 2): return image
        c = image[..., color]
        z = zeros_like(c)
        return array([(c, z, z), (z, c, z), (z, z, c)][color]).transpose(1, 2, 0)


if __name__ == '__main__':
    instance = ColorCat()
    instance.show()
