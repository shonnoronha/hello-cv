import pygame as pg, sys
from pygame.locals import *
import time
import cv2 as cv
import mediapipe as mp

CAM_WIDTH = 480
CAM_HEIGHT = 640

cap = cv.VideoCapture(0)

cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
tip_ids = [i for i in range(4, 21, 4)]

number_of_hands = None

XO = 'x'
winner = None
draw = False
width = 400
height = 400
white = (255, 255, 255)
line_color = (10, 10, 10)

TTT = [[None] * 3, [None] * 3, [None] * 3]

pg.init()
fps = 30
CLOCK = pg.time.Clock()
screen = pg.display.set_mode((width, height + 100), 0, 32)
pg.display.set_caption("Tic Tac Toe")

opening = pg.image.load('tic tac opening.png')
x_img = pg.image.load('x.png')
o_img = pg.image.load('o.png')

x_img = pg.transform.scale(x_img, (80, 80))
o_img = pg.transform.scale(o_img, (80, 80))
opening = pg.transform.scale(opening, (width, height + 100))


def game_opening():
    screen.blit(opening, (0, 0))
    pg.display.update()
    time.sleep(1)
    screen.fill(white)
    pg.draw.line(screen, line_color, (width / 3, 0), (width / 3, height), 7)
    pg.draw.line(screen, line_color, (width / 3 * 2, 0), (width / 3 * 2, height), 7)
    pg.draw.line(screen, line_color, (0, height / 3), (width, height / 3), 7)
    pg.draw.line(screen, line_color, (0, height / 3 * 2), (width, height / 3 * 2), 7)
    draw_status()


def draw_status():
    global draw

    if winner is None:
        message = XO.upper() + "'s Turn"
    else:
        message = winner.upper() + " won!"
    if draw:
        message = 'Game Draw!'

    font = pg.font.Font(None, 30)
    text = font.render(message, True, (255, 255, 255))

    screen.fill((0, 0, 0), (0, 400, 500, 100))
    text_rect = text.get_rect(center=(width / 2, 500 - 50))
    screen.blit(text, text_rect)
    pg.display.update()


def check_win():
    global TTT, winner, draw

    for row in range(0, 3):
        if (TTT[row][0] == TTT[row][1] == TTT[row][2]) and (TTT[row][0] is not None):
            winner = TTT[row][0]
            pg.draw.line(screen, (250, 0, 0), (0, (row + 1) * height / 3 - height / 6),
                         (width, (row + 1) * height / 3 - height / 6), 4)
            break

    for col in range(0, 3):
        if (TTT[0][col] == TTT[1][col] == TTT[2][col]) and (TTT[0][col] is not None):
            winner = TTT[0][col]
            pg.draw.line(screen, (250, 0, 0), ((col + 1) * width / 3 - width / 6, 0),
                         ((col + 1) * width / 3 - width / 6, height), 4)
            break

    if (TTT[0][0] == TTT[1][1] == TTT[2][2]) and (TTT[0][0] is not None):
        winner = TTT[0][0]
        pg.draw.line(screen, (250, 70, 70), (50, 50), (350, 350), 4)

    if (TTT[0][2] == TTT[1][1] == TTT[2][0]) and (TTT[0][2] is not None):
        winner = TTT[0][2]
        pg.draw.line(screen, (250, 70, 70), (350, 50), (50, 350), 4)

    if all([all(row) for row in TTT]) and winner is None:
        draw = True
    draw_status()


def drawXO(row, col):
    global TTT, XO
    if row == 1:
        posx = 30
    if row == 2:
        posx = width / 3 + 30
    if row == 3:
        posx = width / 3 * 2 + 30

    if col == 1:
        posy = 30
    if col == 2:
        posy = height / 3 + 30
    if col == 3:
        posy = height / 3 * 2 + 30
    TTT[row - 1][col - 1] = XO
    if XO == 'x':
        screen.blit(x_img, (posy, posx))
        XO = 'o'
    else:
        screen.blit(o_img, (posy, posx))
        XO = 'x'
    pg.display.update()


def userClick(pos):
    if pos == 1:
        row, col = 1, 1
    elif pos == 2:
        row, col = 1, 2
    elif pos == 3:
        row, col = 1, 3
    elif pos == 4:
        row, col = 2, 1
    elif pos == 5:
        row, col = 2, 2
    elif pos == 6:
        row, col = 2, 3
    elif pos == 7:
        row, col = 3, 1
    elif pos == 8:
        row, col = 3, 2
    elif pos == 9:
        row, col = 3, 3
    else:
        row, col = None, None

    if row and col and TTT[row - 1][col - 1] is None:
        global XO
        drawXO(row, col)
        check_win()


def reset_game():
    global TTT, winner, XO, draw
    time.sleep(3)
    XO = 'x'
    draw = False
    game_opening()
    winner = None
    TTT = [[None] * 3, [None] * 3, [None] * 3]


game_opening()


while True:
    
    _, img = cap.read()
    lm_list = []
    lm_list_two = []
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    r_count = 0
    l_count = 0
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        for hand_no, hand in enumerate(results.multi_hand_landmarks):
            for idx, hand_coordinates in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(hand_coordinates.x * w), int(hand_coordinates.y * h)
                if number_of_hands == 1:
                    lm_list.append([idx, cx, cy])
                elif number_of_hands == 2:
                    if hand_no == 0:
                        lm_list.append([idx, cx, cy])
                    else:
                        lm_list_two.append([idx, cx, cy])
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    if lm_list:
        right_hand = lm_list[5][1] < lm_list[17][1]
        fingers = []
        for i in tip_ids:
            if i == 4 and right_hand:
                if lm_list[i][1] < lm_list[i - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            elif i == 4 and not right_hand:
                if lm_list[i][1] > lm_list[i - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lm_list[i][2] < lm_list[i - 3][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        r_count = sum(fingers)
    
    if lm_list_two:
        right_hand = lm_list_two[5][1] < lm_list_two[17][1]
        fingers_two = []
        for i in tip_ids:
            if i == 4 and right_hand:
                if lm_list_two[i][1] < lm_list_two[i - 1][1]:
                    fingers_two.append(1)
                else:
                    fingers_two.append(0)
            elif i == 4 and not right_hand:
                if lm_list_two[i][1] > lm_list_two[i - 1][1]:
                    fingers_two.append(1)
                else:
                    fingers_two.append(0)
            else:
                if lm_list_two[i][2] < lm_list_two[i - 3][2]:
                    fingers_two.append(1)
                else:
                    fingers_two.append(0)
        l_count = sum(fingers_two)
    t_count = (l_count + r_count)

    if t_count != 0 and t_count != 10:
        userClick(t_count)
        if winner or draw:
            reset_game()
    
    for event in pg.event.get():
        if event.type == QUIT:
            pg.quit()
            sys.exit()

    cv.putText(img, str(t_count), (20, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 355, 0))
    cv.imshow('Feed', img)
    cv.waitKey(1)
    pg.display.update()
    CLOCK.tick(fps)
