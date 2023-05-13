import pygame
from time import time
import Adafruit_ADS1x15
import threading

adc = Adafruit_ADS1x15.ADS1015()
GAIN = 1
low_threshold = 1000
high_threshold = 1500

DISPLAY_HEIGHT = 450
DISPLAY_WIDTH = 750
BORDER_SIZE = 3
IS_FULL_SCREEN = False
GO_FULL_SCREEN = False

BAR_LENGTH = 60
BAR_WIDTH = 10
BAR_STEP = 6

BALL_SIZE = 10
INITIAL_BALL_SPEED = 5
SPEED_CHANGE = 1
MAX_SPEED = 12

BAR_OFFSET = 50
p1_x = BAR_OFFSET
p2_x = DISPLAY_WIDTH - BAR_OFFSET

UP, DOWN, RIGHT, LEFT = -1, 1, 1, -1

p1_y = p2_y = ball_x = ball_y = 0
user_input = [0, 0]


def display_game(display):
    global p1_y
    global p2_y
    global ball_x
    global ball_y
    global GO_FULL_SCREEN
    global IS_FULL_SCREEN
    global DISPLAY_WIDTH
    global DISPLAY_HEIGHT
    global BORDER_SIZE

    clock = pygame.time.Clock()

    while True:
        display.fill((0, 0, 0))

        bar_1 = pygame.Rect(p1_x, p1_y, BAR_WIDTH, BAR_LENGTH)
        pygame.draw.rect(display, (255, 255, 255), bar_1)

        bar_2 = pygame.Rect(p2_x, p2_y, BAR_WIDTH, BAR_LENGTH)
        pygame.draw.rect(display, (255, 255, 255), bar_2)

        ball = pygame.Rect(ball_x, ball_y, BALL_SIZE, BALL_SIZE)
        pygame.draw.rect(display, (255, 255, 255), ball)

        pygame.draw.rect(
            display, (255, 255, 255), (0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), BORDER_SIZE
        )  # border

        pygame.display.update()
        clock.tick(80)


def update_user_input(display):
    global user_input
    global GO_FULL_SCREEN
    global IS_FULL_SCREEN

    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif (
                event.type is pygame.KEYDOWN and event.key == pygame.K_e
            ) or event.type == pygame.MOUSEBUTTONUP:
                pygame.quit()
                exit()

        key_input = pygame.key.get_pressed()
        joystick1 = adc.read_adc(0, gain=GAIN)
        joystick2 = adc.read_adc(1, gain=GAIN)
        if key_input[pygame.K_q] or joystick1 >= high_threshold:
            user_input[0] = UP
        elif key_input[pygame.K_a] or joystick1 <= low_threshold:
            user_input[0] = DOWN
        else:
            user_input[0] = 0
        if key_input[pygame.K_p] or joystick2 <= low_threshold:
            user_input[1] = UP
        elif key_input[pygame.K_l] or joystick2 >= high_threshold:
            user_input[1] = DOWN
        else:
            user_input[1] = 0


def ball_collides_left(
    ball_x, ball_y, ball_dir_x, p1_y, BAR_WIDTH, BAR_LENGTH, BALL_SIZE, BALL_SPEED
):
    if ball_dir_x == LEFT and ball_x <= p1_x + BAR_WIDTH + BALL_SPEED // 2:
        if ball_y > p1_y - BALL_SIZE and ball_y < p1_y + BAR_LENGTH + BALL_SIZE - 1:
            return 1
        return -1
    return 0


def ball_collides_right(
    ball_x, ball_y, ball_dir_x, p2_y, BAR_LENGTH, BALL_SIZE, BALL_SPEED
):
    if ball_dir_x == RIGHT and p2_x - BALL_SPEED // 2 - BALL_SIZE <= ball_x:
        if ball_y > p2_y - BALL_SIZE and ball_y < p2_y + BAR_LENGTH + BALL_SIZE - 1:
            return 1
        return -1
    return 0


def ball_collides_up(ball_y, BALL_SPEED, BORDER_SIZE):
    return ball_y <= BALL_SPEED // 2 + BORDER_SIZE


def ball_collides_down(ball_y, DISPLAY_HEIGHT, BALL_SIZE, BALL_SPEED, BORDER_SIZE):
    return ball_y >= DISPLAY_HEIGHT - BALL_SIZE - BALL_SPEED // 2 - BORDER_SIZE


def ball_got_out(ball_x, DISPLAY_WIDTH, BALL_SIZE):
    return ball_x <= p1_x or ball_x > p2_x + BAR_WIDTH


def update_positions(user_input):
    global DISPLAY_HEIGHT
    global DISPLAY_WIDTH
    global BORDER_SIZE
    global BAR_LENGTH
    global BALL_SIZE
    global BAR_STEP
    global INITIAL_BALL_SPEED
    global UP
    global DOWN
    global RIGHT
    global LEFT

    global p1_y
    global p2_y
    global ball_x
    global ball_y

    clock = pygame.time.Clock()

    while True:
        # reset game setup
        p1_y = p2_y = (DISPLAY_HEIGHT - BAR_LENGTH) // 2 + 2
        ball_x = (DISPLAY_WIDTH - BALL_SIZE) // 2
        ball_y = (DISPLAY_HEIGHT - BALL_SIZE) // 2
        ball_dir_x = LEFT
        ball_dir_y = 0

        BALL_SPEED = INITIAL_BALL_SPEED

        while True:
            new_p1_y = p1_y + user_input[0] * BAR_STEP
            if 0 <= new_p1_y < DISPLAY_HEIGHT - BAR_LENGTH:
                p1_y = new_p1_y

            new_p2_y = p2_y + user_input[1] * BAR_STEP
            if 0 <= new_p2_y < DISPLAY_HEIGHT - BAR_LENGTH:
                p2_y = new_p2_y

            # updating ball position
            ball_x += ball_dir_x * BALL_SPEED
            ball_y += ball_dir_y * BALL_SPEED

            left_c = ball_collides_left(
                ball_x,
                ball_y,
                ball_dir_x,
                p1_y,
                BAR_WIDTH,
                BAR_LENGTH,
                BALL_SIZE,
                BALL_SPEED,
            )
            if left_c == 1:
                ball_dir_x = RIGHT
                ball_dir_y = (ball_y + BALL_SIZE / 2 - (p1_y + BAR_LENGTH / 2)) / (
                    BAR_LENGTH / 2
                )
                if BALL_SPEED < MAX_SPEED:
                    BALL_SPEED += SPEED_CHANGE

            right_c = ball_collides_right(
                ball_x, ball_y, ball_dir_x, p2_y, BAR_LENGTH, BALL_SIZE, BALL_SPEED
            )
            if right_c == 1:
                ball_dir_x = LEFT
                ball_dir_y = (ball_y + BALL_SIZE / 2 - (p2_y + BAR_LENGTH / 2)) / (
                    BAR_LENGTH / 2
                )
                if BALL_SPEED < MAX_SPEED:
                    BALL_SPEED += SPEED_CHANGE

            if ball_collides_up(ball_y, BALL_SPEED, BORDER_SIZE) or ball_collides_down(
                ball_y, DISPLAY_HEIGHT, BALL_SIZE, BALL_SPEED, BORDER_SIZE
            ):
                ball_dir_y = -ball_dir_y

            clock.tick(60)

            if ball_got_out(ball_x, DISPLAY_WIDTH, BALL_SIZE):
                break


def main():
    pygame.init()
    pygame.display.set_caption("Pong")
    display = pygame.display.set_mode(
        (DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.FULLSCREEN
    )
    pygame.mouse.set_visible(False)

    input_thread = threading.Thread(target=update_user_input, args=(display,))
    moves_thread = threading.Thread(target=update_positions, args=(user_input,))
    display_thread = threading.Thread(target=display_game, args=(display,))

    input_thread.start()
    moves_thread.start()
    display_thread.start()


if __name__ == "__main__":
    main()
