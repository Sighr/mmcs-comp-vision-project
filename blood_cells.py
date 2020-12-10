import cv2


def main():
    image = cv2.imread('blood_cells.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    # gradient = cv2.spatialGradient(image)

    cv2.imshow('image', image)
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    # cv2.imshow('graident', gradient)
    cv2.waitKey()


if __name__ == '__main__':
    main()
