import cv2


def main():
    image = cv2.imread('blood_cells.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    blurred_gradient = cv2.GaussianBlur(gradient, (5, 5), 2)

    _, binarized_gradient = cv2.threshold(blurred_gradient, 0, 255, cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(binarized_gradient, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    cv2.imshow('image', image)
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('gradient', gradient)
    cv2.imshow('binarized_gradient', binarized_gradient)
    cv2.imshow('opened', opened)
    cv2.waitKey()


if __name__ == '__main__':
    main()
