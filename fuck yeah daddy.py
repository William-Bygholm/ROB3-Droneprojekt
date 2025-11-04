import cv2

img = cv2.resize(cv2.imread('Billeder/IMG_20251104_094219.jpg'), (0,0), fx=0.1, fy=0.1)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()