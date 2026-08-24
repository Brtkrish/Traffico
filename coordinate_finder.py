import cv2

# Load your video frame
cap = cv2.VideoCapture("west.mp4")
ret, frame = cap.read()


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click 4 points of your lane", frame)


cv2.imshow("Click 4 points of your lane", frame)
cv2.setMouseCallback("Click 4 points of your lane", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()