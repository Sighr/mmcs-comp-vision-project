import cv2
import numpy as np

colors = [[30, 200, 0],
          [0, 100, 100],
          [15, 130, 240],
          [30, 0, 200],
          [240, 130, 15],
          [50, 50, 50],
          [244, 80, 196],
          [255, 80, 15],
          [70, 130, 4]
          ]


def main():
    image = cv2.imread('coins.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray / 10, dtype=np.uint8)

    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 0)
    opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    lower_borders = cv2.filter2D(gray, -1, prewitt)
    upper_borders = cv2.filter2D(gray, -1, -prewitt)
    left_borders = cv2.filter2D(gray, -1, prewitt.T)
    right_borders = cv2.filter2D(gray, -1, -prewitt.T)
    borders = lower_borders + upper_borders + left_borders + right_borders
    thresh, bin_borders = cv2.threshold(borders, 0, 255, cv2.THRESH_OTSU)
    bin_borders_dilated = cv2.dilate(bin_borders, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    combined = cv2.bitwise_and(bin_borders_dilated, closed)
    close_combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    trash_cnts, hierarchy = cv2.findContours(close_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coins = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(coins, trash_cnts, -1, 255, -1, maxLevel=1, hierarchy=hierarchy)

    # cnts = cv2.findContours(coins, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    labels_num, labeled = cv2.connectedComponents(coins)
    infos = []
    for i in range(1, labels_num):
        current_image = np.array(labeled == i, dtype=np.uint8)
        info = {
            'area': np.sum(current_image),
            'image': current_image,
            'color': colors[i - 1]
        }
        infos.append(info)

    infos.sort(key=lambda x: x['area'])

    diffs = [infos[i]['area'] - infos[i - 1]['area'] for i in range(1, labels_num - 1)]
    border = np.argmax(diffs) + 1

    little = np.zeros(image.shape, dtype=np.uint8)
    big = np.zeros(image.shape, dtype=np.uint8)
    for i in range(border):
        cnt, _ = cv2.findContours(infos[i]['image'], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(little, cnt, 0, infos[i]['color'], -1)

    for i in range(border, labels_num - 1):
        cnt, _ = cv2.findContours(infos[i]['image'], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cnt, 0, infos[i]['color'], -1)

    to_show = [
        ('image', image),
        ('gray', gray),
        ('binarized', binarized),
        ('opened', opened),
        ('closed', closed),
        ('lower_borders', lower_borders),
        ('upper_borders', upper_borders),
        ('left_borders', left_borders),
        ('right_borders', right_borders),
        ('borders', borders),
        ('bin_borders', bin_borders),
        ('bin_borders_dilated', bin_borders_dilated),
        ('combined', combined),
        ('close_combined', close_combined),
        ('coins', coins),
        ('little', little),
        ('big', big),
    ]

    for img_name, img in to_show:
        cv2.imshow(img_name, img)
        cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
