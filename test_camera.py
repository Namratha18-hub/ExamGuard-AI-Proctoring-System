import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("ExamGuard Camera Test", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()