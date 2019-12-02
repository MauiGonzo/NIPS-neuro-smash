from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def label_images(start, end):
    def show_image(img, title, onclick):
        plt.imshow(img)
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.title(title)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()

    def onclickRed(event):
        global red
        red = (int(event.xdata + 1.5), int(event.ydata + 1.5))
        plt.close()

    def onclickBlue(event):
        global blue
        blue = (int(event.xdata + 1.5), int(event.ydata + 1.5))
        plt.close()

    def onclickOverlap(event):
        global overlap
        overlap = event.xdata is not None and event.ydata is not None
        plt.close()

    for i in range(start, end):
        img = np.array(Image.open(f'data/{i}_label.png'))
        show_image(img, 'Select Red', onclickRed)
        show_image(img, 'Select Blue', onclickBlue)
        show_image(img,
                   'Click on image to select overlap, '
                   'click off image to select non-overlap',
                   onclickOverlap)

        print(i, red, blue, overlap)
        with open('labels.csv', 'a') as f:
            f.write(f'{i}, {red[0]}, {red[1]}, {blue[0]}, {blue[1]}, {overlap}\n')


if __name__ == '__main__':
    red = (0, 0)
    blue = (0, 0)
    overlap = False

    label_images(0, 166)
