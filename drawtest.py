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
    global img
    global pts

    # Create a window and set the draw_line callback function
    cv2.namedWindow('Quickdraw')
    cv2.setMouseCallback('Quickdraw', draw_line)
    emojis = get_QD_emojis()
    pred_class = 0

    while True:
        # Draw the user's lines on the blackboard
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(img, pts[i - 1], pts[i], (255, 255, 255), 7)

        # Get the contour of the drawing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        cnts, heir = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # Set text on the image
        cv2.putText(img, 'Draw an apple!', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        print(len(cnts))

        if len(cnts) == 0:
            if len(pts):
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(img_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                img_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
                if len(img_cnts) >= 1:
                    cnt = max(img_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = img_gray[y:y + h, x:x + w]
                        print("Keras predict:")
                        pred_probab, pred_class = keras_predict(model, digit)
                        print(pred_class, pred_probab)

            pts = deque(maxlen=512)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img = overlay(img, emojis[pred_class], 400, 250, 100, 100)

        cv2.imshow('Quickdraw', img)

        # Press escape to exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def get_QD_emojis():
    emojis_folder = 'qd_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))

if __name__ == '__main__':
    main()
