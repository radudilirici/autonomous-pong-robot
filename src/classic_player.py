from picamera import PiCamera
#from picamera.array import PiRGBArray
from time import sleep, time
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2 as cv
from gpiozero import AngularServo

SHOW_CAMERA = False

servoPIN = 17
servo = AngularServo(servoPIN, min_angle=-40, max_angle=40)
servo_move_angle = 17
servo_left_angle = -13.5
servo_right_angle = 19
servo.angle = 0

ball_color = (50, 50, 180)
ball_color_2 = (100, 100, 180)
bar1_color = (150, 180, 50)
bar2_color = (100, 130, 20)
ball_pred_color = (30, 250, 250)
bar_pred_color = (30, 200, 190)
corner_color = (100, 50, 30)


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


TRANSFORM_SIZE = (400, 300)


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
    if len(centers) < 3:
        return None
    centers.sort(key=lambda a: a["x"])
    centers[0], centers[1] = centers[1], centers[0]
    return centers[:3]


def get_centers_fast(thresh, b_c, p1_c, p2_c):
    """returns a tuple (ball_center, bar1_center, bar2_center)"""

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
    centers[0], centers[1] = centers[1], centers[0]
    return centers


def ball_dir_changed(first_ball_center, last_ball_center, new_ball_center):
    x_changed = np.sign(last_ball_center["x"] - first_ball_center["x"]) != np.sign(
        new_ball_center["x"] - last_ball_center["x"]
    )
    y_changed = np.sign(last_ball_center["y"] - first_ball_center["y"]) == -np.sign(
        new_ball_center["y"] - last_ball_center["y"]
    )

    return x_changed or y_changed


def predict_ball_y_old(ball_center_old, ball_center_new, bar1_center, bar2_center):
    ball_dir = {
        "x": ball_center_new["x"] - ball_center_old["x"],
        "y": ball_center_new["y"] - ball_center_old["y"],
    }
    if ball_dir["x"] == 0:  # the ball didn't move on the X axis. something is wrong
        return None
    ball_dir["y"] /= abs(ball_dir["x"])
    ball_dir["x"] /= abs(ball_dir["x"])  # normalize

    pred = 0
    if ball_dir["x"] > 0:
        pred = ball_center_new["y"] + ball_dir["y"] * (
            bar2_center["x"] - ball_center_new["x"] - 10
        )
    else:
        pred = ball_center_new["y"] + ball_dir["y"] * (
            ball_center_new["x"] - bar1_center["x"] + 10
        )

    pred = int(pred)
    game_ceiling, game_floor = 5, 295
    while not game_ceiling <= pred <= game_floor:
        if pred < game_ceiling:
            pred = game_ceiling + (game_ceiling - pred)
        elif pred > game_floor:
            pred = game_floor - (pred - game_floor)

    return pred, ball_dir["x"]


def predict_ball_y(first_ball_center, new_ball_center, no_steps, bar_x):
    ball_dir = {
        "x": (new_ball_center["x"] - first_ball_center["x"]) / no_steps,
        "y": (new_ball_center["y"] - first_ball_center["y"]) / no_steps,
    }
    if ball_dir["x"] == 0:  # the ball didn't move on the X axis. something is wrong
        return None
    ball_dir["y"] /= abs(ball_dir["x"])
    ball_dir["x"] /= abs(ball_dir["x"])  # normalize

    if ball_dir["x"] < 0:
        return last_ball_center["y"] + ball_dir["y"] * abs(
            bar_x - last_ball_center["x"]
        )
    else:
        return None


def get_real_ball_y(pred):
    pred = int(pred)
    game_ceiling, game_floor = 5, TRANSFORM_SIZE[1] - 5
    while not game_ceiling <= pred <= game_floor:
        if pred < game_ceiling:
            pred = game_ceiling + (game_ceiling - pred)
        elif pred > game_floor:
            pred = game_floor - (pred - game_floor)

    return pred


inertia = 0
inertia_unit = 1 / 3


def predict_bar_y(old_bar_y, new_bar_y):
    global inertia
    pred = new_bar_y + abs(new_bar_y - old_bar_y) * inertia
    return int(pred)


def move_robot(pred_ball_y, bar_pred, bar_y, ball_dir):
    global inertia, inertia_div
    if ball_dir == "right":
        pred_ball_y = TRANSFORM_SIZE[1] // 2  # center

    if pred_ball_y > (bar_pred + 15) >= (bar_y + 15):  # the ball is lower
        if servo.angle != servo_right_angle:
            servo.angle = servo_right_angle
    elif pred_ball_y < (bar_pred - 15) <= (bar_y + 15):  # the ball is higher
        if servo.angle != servo_left_angle:
            servo.angle = servo_left_angle
    else:
        if servo.angle != 0:
            servo.angle = 0

    if servo.angle != 0:
        if abs(inertia + np.sign(servo.angle) * inertia_unit) <= 3:
            inertia += np.sign(servo.angle) * inertia_unit
    else:
        if servo.angle > 0:
            inertia -= inertia_unit
        elif servo.angle < 0:
            inertia += inertia_unit


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CAMERA_RESOLUTION = (400, 304)
(
    first_ball_center,
    last_ball_center,
    new_ball_center,
    old_bar1_center,
    bar1_center,
    bar2_center,
) = (None, None, None, None, None, None)
screen, corners, centers = None, None, None
no_steps = 0

with PiCamera(resolution=CAMERA_RESOLUTION, framerate=30) as camera:
    stream = BytesIO()
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
            if (
                first_ball_center is None
                or centers[0] is None
                or centers[1] is None
                or centers[2] is None
            ):
                first_ball_center, old_bar1_center, bar2_center = centers
                last_ball_center = None
                no_steps = 0
            else:
                # Now we have at least two ball centers, so we can predict where the ball will land
                new_ball_center, bar1_center, bar2_center = centers
                no_steps += 1
                if last_ball_center is None:
                    last_ball_center = new_ball_center
                else:
                    if ball_dir_changed(
                        first_ball_center, last_ball_center, new_ball_center
                    ):
                        first_ball_center = last_ball_center
                        no_steps = 1

                # Apply "inertia"
                bar_pred = predict_bar_y(old_bar1_center["y"], bar1_center["y"])

                if SHOW_CAMERA:
                    cv.rectangle(
                        screen,
                        (first_ball_center["x"] - 2, first_ball_center["y"] - 2),
                        (first_ball_center["x"] + 2, first_ball_center["y"] + 2),
                        ball_color_2,
                        3,
                    )
                    cv.rectangle(
                        screen,
                        (new_ball_center["x"] - 2, new_ball_center["y"] - 2),
                        (new_ball_center["x"] + 2, new_ball_center["y"] + 2),
                        ball_color,
                        3,
                    )
                    cv.rectangle(
                        screen,
                        (bar1_center["x"] - 2, bar1_center["y"] - 2),
                        (bar1_center["x"] + 2, bar1_center["y"] + 2),
                        bar1_color,
                        3,
                    )
                    cv.rectangle(
                        screen,
                        (bar2_center["x"] - 2, bar2_center["y"] - 2),
                        (bar2_center["x"] + 2, bar2_center["y"] + 2),
                        bar2_color,
                        3,
                    )

                ball_pred_y = predict_ball_y(
                    first_ball_center, new_ball_center, no_steps, bar1_center["x"]
                )
                if ball_pred_y is not None:
                    y = get_real_ball_y(ball_pred_y)
                    x = bar1_center["x"]
                    move_robot(y, bar_pred, bar1_center["y"], "left")
                    if SHOW_CAMERA:
                        cv.rectangle(
                            screen, (x - 2, y - 2), (x + 2, y + 2), ball_pred_color, 3
                        )  # prediction
                else:
                    move_robot(None, bar_pred, bar1_center["y"], "right")

                old_bar1_center = bar1_center
                last_ball_center = new_ball_center

        if SHOW_CAMERA:
            # cv.imshow("Camera", img)
            # if corners is not None:
            #    for x, y in corners:
            #        cv.rectangle(img, (x-2, y-2), (x+2, y+2), corner_color, 3)
            # cv.imshow("Corners", img)
            # cv.imshow("Black and White", thresh)
            if screen is not None:
                cv.imshow("Screen", screen)

        cv.waitKey(1)
        stream.seek(0)
        stream.truncate()
