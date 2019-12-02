from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

for i in range(666, 1000):
    img = np.array(Image.open(f'images/{i}_label.png'))

    blue = (0, 0)

    def onclick(event):
        global blue
        blue = (event.xdata, event.ydata)
        plt.close()

    plt.imshow(img)
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Select Blue')
    plt.show()

    red = (0, 0)

    def onclick(event):
        global red
        red = (event.xdata, event.ydata)
        plt.close()
    
    plt.imshow(img)
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Select Red')
    plt.show()

    print(i, blue, red)
    with open('labels.csv', 'a') as f:
        f.write(f'{i}, {blue[0]}, {blue[1]}, {red[0]}, {red[1]}\n')