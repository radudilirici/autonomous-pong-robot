import torch
import torch.nn as nn
import numpy as np
import cv2

NET_IMG_SIZE = (84, 84)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


############################################################


from picamera import PiCamera

# from picamera.array import PiRGBArray
from time import sleep, time
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2 as cv
from gpiozero import AngularServo

SHOW_CAMERA = True

corner_color = (100, 50, 30)

servoPIN = 17
servo = AngularServo(servoPIN, min_angle=-45, max_angle=45)
servo_move_angle = 25
servo.angle = 0

BAR_LENGTH = 60
BAR_WIDTH = 10
BAR_OFFSET = 50
BALL_SIZE = 10
BIG_SCREEN_SIZE = (750, 450)
NET_ORIGINAL_RATIO = {
    "x": NET_IMG_SIZE[0] / BIG_SCREEN_SIZE[0],
    "y": NET_IMG_SIZE[1] / BIG_SCREEN_SIZE[1],
}


def order_corners(corner_list):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = corner_list.sum(axis=1)
    rect[0] = corner_list[np.argmin(s)]
    rect[3] = corner_list[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(corner_list, axis=1)
    rect[1] = corner_list[np.argmin(diff)]
    rect[2] = corner_list[np.argmax(diff)]
    return rect


def get_corners(gray):
    c = cv.goodFeaturesToTrack(gray, 30, 0.01, 10)
    if c is None:
        return None
    c = c.reshape((c.shape[0], -1))
    c = order_corners(c).astype("int")
    s = 3  # scale. crop a little more
    c[0] += np.array([1, 1]) * s
    c[1] += np.array([-1, 1]) * s
    c[2] += np.array([1, -1]) * s
    c[3] += np.array([-1, -1]) * s
    return c


TRANSFORM_SIZE = (200, 150)


def four_point_transform(image, corners):
    """`corners` have to be: upper left, upper right, lower left, lower right"""
    dst_rect = np.array(
        [
            [0, 0],
            [TRANSFORM_SIZE[0], 0],
            [0, TRANSFORM_SIZE[1]],
            [TRANSFORM_SIZE[0], TRANSFORM_SIZE[1]],
        ],
        dtype="float32",
    )
    corners = corners.astype("float32")
    try:
        M = cv.getPerspectiveTransform(corners, dst_rect)
        warped = cv.warpPerspective(image, M, TRANSFORM_SIZE)
    except:
        return None
    return warped


def get_centers(img):
    """returns a tuple (ball_center, bar1_center, bar2_center)"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
    # labels_no, labels = cv.connectedComponents(thresh)

    centers = []
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)
        if M["m00"] == 0:
            return None
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if thresh[cY][cX]:
            centers.append({"x": cX, "y": cY})
    if len(centers) != 3:
        return None
    centers.sort(key=lambda a: a["x"])
    centers[0], centers[1] = centers[1], centers[0]  # mingea e intre cele doua bare
    return centers


def move_robot(action):
    action -= 1
    servo.angle = action * servo_move_angle


CAMERA_RESOLUTION = (400, 304)

model = "./pipong-best2_old"

net = DQN((4, 84, 84), 3)
net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
# net.load_state_dict(torch.load('./data.pkl'))

null_state = np.zeros((84, 84))
state = np.array([null_state, null_state, null_state, null_state], dtype=np.float32)

last_action = 1

with PiCamera(resolution=CAMERA_RESOLUTION, framerate=20) as camera:
    camera.rotation = 180
    stream = BytesIO()
    screen, corners, centers = None, None, None
    frame_no = 1
    for _ in camera.capture_continuous(stream, format="bgr", use_video_port=True):
        stream.seek(0)
        img = np.array(Image.frombytes("RGB", camera.resolution, stream.read()))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)

        if frame_no <= 3:
            corners = get_corners(thresh)
            frame_no += 1

        if corners is not None:
            screen = four_point_transform(img, corners)

        if screen is not None:
            centers = get_centers(screen)
            if centers is not None:
                b_c, p1_c, p2_c = centers
                last_state = np.zeros(NET_IMG_SIZE, dtype=np.float32)
                b_c["x"] *= 750 / 200
                p1_c["x"] *= 750 / 200
                p2_c["x"] *= 750 / 200
                b_c["y"] *= 450 / 150
                p1_c["y"] *= 450 / 150
                p2_c["y"] *= 450 / 150

                for i in range(BAR_LENGTH // (-2), BAR_LENGTH // 2):
                    for j in range(BAR_WIDTH // (-2), BAR_WIDTH // 2):
                        if i >= 84 or j >= 84:
                            continue
                        y1 = int((p1_c["y"] + i) * NET_ORIGINAL_RATIO["y"])
                        x1 = int((p1_c["x"] + j) * NET_ORIGINAL_RATIO["x"])
                        y2 = int((p2_c["y"] + i) * NET_ORIGINAL_RATIO["y"])
                        x2 = int((p2_c["x"] + j) * NET_ORIGINAL_RATIO["x"])
                        last_state[y1][x1] = 1
                        last_state[y2][x2] = 1
                for i in range(BALL_SIZE // (-2), BALL_SIZE // 2):
                    for j in range(BALL_SIZE // (-2), BALL_SIZE // 2):
                        if i >= 84 or j >= 84:
                            continue
                        y = int((b_c["y"] + i) * NET_ORIGINAL_RATIO["y"])
                        x = int((b_c["x"] + j) * NET_ORIGINAL_RATIO["x"])
                        last_state[y][x] = 1

                state[:3] = state[1:]
                state[3] = last_state
                state_v = torch.tensor(np.array([state], copy=False))
                q_vals = net(state_v).data.numpy()[0]
                action = np.argmax(q_vals)

                action = 2 - action

                move_robot(action)

        if corners is not None and SHOW_CAMERA:
            for x, y in corners:
                cv.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), corner_color, 3)

        if SHOW_CAMERA:
            cv.imshow("cam", img)
            # cv.imshow("thresh", thresh)
            # if screen is not None:
            #    cv.imshow("screen", screen)
            if state is not None:
                cv.imshow("state", state[-1])

        cv.waitKey(1)
        stream.seek(0)
        stream.truncate()
