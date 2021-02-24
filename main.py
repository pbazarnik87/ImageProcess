import cv2 as cv
import basic_operations


def main():
    """
    main
    """

    '#show original'
    original = basic_operations.load_image(r'pic\face.png')
    cv.imshow("original", original)

    # color2gray
    gray = basic_operations.convert_bgr_to_gray(r'pic\face.png')
    cv.imshow("gray", gray)

    # blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow("blur", blur)

    '#otsu'
    otsu = basic_operations.otsu(blur)
    otsu_img = otsu[1]
    cv.imshow("otsu", otsu_img)

    '#edges Canny'
    edges = cv.Canny(otsu_img, 0, 200)
    cv.imshow("canny", edges)

    '#count circles'
    cir = basic_operations.circles_detect(original, edges)
    cv.imshow('detected circles', cir)
    cv.waitKey(0)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        otsu = basic_operations.otsu(blur)
        otsu_img = otsu[1]
        edges = cv.Canny(otsu_img, 0, 200)
        cir = basic_operations.circles_detect(frame, edges)
#
        cv.imshow('frame', cir)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
