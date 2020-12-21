import cv2
import numpy as np


def main():
    image = cv2.imread("car_plates.png")
    copy = np.copy(image)
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(image)
    for x, y, w, h in plates:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('copy', copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # canny = cv2.Canny(image, 50, 100)
    #
    # to_show = [
    #     ('image', image),
    #     ('canny', canny),
    # ]
    #
    # for img_name, img in to_show:
    #     cv2.imshow(img_name, img)
    #     cv2.waitKey()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
