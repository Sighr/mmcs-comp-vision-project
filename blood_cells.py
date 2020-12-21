import cv2
import numpy as np


def main():
    circle5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    circle3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    circle7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    circle11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    image = cv2.imread('blood_cells.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    threshold, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, circle5)

    blurred_gradient = cv2.GaussianBlur(gradient, (5, 5), 2)

    gradient_threshold, binarized_gradient = cv2.threshold(blurred_gradient, 0, 255, cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(binarized_gradient, cv2.MORPH_OPEN, circle5)
    # clsd = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, circle11)

    # canny = cv2.Canny(gray, gradient_threshold * 2, threshold * 2)
    # dilated = cv2.dilate(canny, circle3)
    # canny_bl = cv2.GaussianBlur(dilated, (15, 15), 5)
    #
    # new_canny = cv2.Canny(canny_bl, gradient_threshold, threshold)
    #
    # canny_and_opened = cv2.bitwise_and(new_canny, opened)
    #
    # dl = cv2.dilate(canny_and_opened, circle12)
    # clsd = cv2.morphologyEx(binarized_gradient, cv2.MORPH_CLOSE, circle5)


    # canny_and_opened has everything we need with some extra noise. pretend to delete it based on surroundings

    # dilated = cv2.dilate(canny_and_opened, circle3)
    # image_borders = np.array((dilated / 255) * gray, dtype=np.uint8)
    # borders_blurred = cv2.filter2D(image_borders, -1, np.ones([3, 3]) / 9)

    not_opened = cv2.bitwise_not(opened)
    opened_not_opened = cv2.morphologyEx(not_opened, cv2.MORPH_OPEN, circle7)
    class_num, result = cv2.connectedComponents(opened_not_opened)
    print(class_num)
    result = result.astype(np.uint8) * 4

    cv2.imshow('image', image)
    cv2.imshow('gray', gray)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow('binarized', binarized)
    # cv2.imshow('gradient', gradient)
    # cv2.imshow('binarized_gradient', binarized_gradient)
    cv2.imshow('opened', opened)
    # cv2.imshow('canny', canny)
    # cv2.imshow('dilated', dilated)
    # cv2.imshow('canny_bl', canny_bl)
    # cv2.imshow('canny_and_opened', canny_and_opened)
    # cv2.imshow('dl', dl)
    # cv2.imshow('clsd', clsd)
    # cv2.imshow('image_borders', image_borders)
    # cv2.imshow('borders_blurred', borders_blurred)
    cv2.imshow('result', result)
    cv2.waitKey()


if __name__ == '__main__':
    main()
