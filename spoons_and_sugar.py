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
    image = cv2.imread('spoons_and_sugar.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, binarized = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    labels_num, labeled = cv2.connectedComponents(opened)
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

    spoons = np.zeros(image.shape, dtype=np.uint8)
    sugars = np.zeros(image.shape, dtype=np.uint8)
    for i in range(border):
        cnt, _ = cv2.findContours(infos[i]['image'], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sugars, cnt, 0, infos[i]['color'], -1)

    for i in range(border, labels_num - 1):
        cnt, _ = cv2.findContours(infos[i]['image'], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(spoons, cnt, 0, infos[i]['color'], -1)

    cv2.imshow('image', image)
    cv2.imshow('gray', gray)
    cv2.imshow('binarized', binarized)
    cv2.imshow('opened', opened)
    cv2.imshow('spoons', spoons)
    cv2.imshow('sugars', sugars)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
