import os
import cv2
import numpy as np
from collections import deque
from keras.models import load_model

model = load_model('models/qdModel.h5')

# Set variables
img = np.zeros((480, 640, 3), dtype=np.uint8)
pts = deque(maxlen=512)
drawing = False  # True if mouse is pressed

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pts.appendleft((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            pts.appendleft((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def main():
    while True:
        # Draw the user's lines on the blackboard
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(img, pts[i - 1], pts[i], (255, 255, 255), 7)

        # Put text on the display
        cv2.putText(img, 'Drawing App', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the blackboard
        cv2.imshow('Quickdraw', img)

        # Press 'Esc' to exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create a window and set the draw_line callback function
    cv2.namedWindow('Quickdraw')
    cv2.setMouseCallback('Quickdraw', draw_line)

    main()
