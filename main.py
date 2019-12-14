import argparse
import copy
import random
import sys

import pygame

# configurations
# board_total_height, complete_rows, calc_bumpiness, count_holes
random.seed(42)

AI_DELAY = 100
_WEIGHTS = [-0.510066, 0.760666, -0.35663, -0.184483]

config = {'cell_size': 20, 'cols': 10, 'rows': 20, 'delay': 750, 'maxfps': 60}

COLOR_GRID = (50, 50, 50)
COLORS = [
    (0, 0, 0),
    (180, 0, 255),
    (0, 150, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 120, 0),
    (0, 220, 220),
    (255, 255, 0),
]


# helpers
def printit(func):
    def f(*args, **kwargs):
        rv = func(*args, **kwargs)
        print(rv)
        return rv

    return f


def print_instance(board):
    for row in board:
        print(' '.join([str(col) for col in row]))


def time_convert(millis):
    secs = int((millis / 1000) % 60)
    mins = int((millis / (1000 * 60)) % 60)
    return f'{mins}:{secs}'


# define the 7 shapes of the single mino
mino_shapes = [
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 2, 2], [2, 2, 0]],  # S
    [[3, 3, 0], [0, 3, 3]],  # Z
    [[4, 0, 0], [4, 4, 4]],  # J
    [[0, 0, 5], [5, 5, 5]],  # L
    [[6, 6, 6, 6]],  # I
    [[7, 7], [7, 7]],  # O
]


class Bag7:
    def __init__(self, getter, maxpeek=50):
        self.getter = getter
        self.maxpeek = maxpeek
        self.b = [next(getter) for _ in range(maxpeek)]
        self.i = 0

    def pop(self):
        result = self.b[self.i]
        self.b[self.i] = next(self.getter)
        self.i += 1
        if self.i >= self.maxpeek:
            self.i = 0
        return result

    def peek(self, n=3):
        if not 0 <= n <= self.maxpeek:
            raise ValueError('bad peek argument')
        nthruend = self.maxpeek - self.i
        if n <= nthruend:
            result = self.b[self.i : self.i + n]
        else:
            result = self.b[self.i :] + self.b[: n - nthruend]
        return result


def shape_rand():
    rng = list(range(7))
    while True:
        random.shuffle(rng)
        for i in rng:
            yield i


def rotate_clockwise(shape):
    return [
        [shape[y][x] for y in range(len(shape) - 1, -1, -1)]
        for x in range(len(shape[0]))
    ]


def rotate_counter_clockwise(shape):
    return [
        [shape[y][x] for y in range(len(shape))]
        for x in range(len(shape[0]) - 1, -1, -1)
    ]


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def valid_state(board):
    for row in board:
        for col, cell in enumerate(row):
            if cell > len(mino_shapes):
                return False
    return True


def remove_row(board, row):
    del board[row]
    return [[0 for i in range(config['cols'])]] + board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    board = [[0 for x in range(config['cols'])] for y in range(config['rows'])]
    board += [[1 for x in range(config['cols'])]]
    return board


# heuristic utilities
def complete_rows(board):
    rv = 0
    for row in board[:-1]:
        if 0 not in row:
            rv += 1
    return rv


def board_heights(board):
    flipped = [False] * config['cols']
    rv = [0] * config['cols']
    for i, row in enumerate(reversed(board[:-1])):
        for j, cell in enumerate(row):
            if not cell and not flipped[j]:
                rv[j] = i
                flipped[j] = True
            elif cell:
                flipped[j] = False
    return rv


def board_max_height(board):
    return max(board_heights(board))


def board_total_height(board):
    return sum(board_heights(board))


def calc_bumpiness(board):
    heights = board_heights(board)
    rv = 0
    for icol in range(1, config['cols']):
        rv += abs(heights[icol] - heights[icol - 1])
    return rv


def count_holes(board):
    rv = [0] * config['cols']
    prevs = [0] * config['cols']
    for i, row in enumerate(reversed(board)):
        for j, cell in enumerate(row):
            if cell:
                if i > prevs[j] + 1:
                    rv[j] += i - prevs[j] - 1
                prevs[j] = i
    return sum(rv)


class TetrisApp:
    def __init__(self, enable_ai=False, sprint=0):
        pygame.init()
        # TODO key repeat for A, S, D only
        # pygame.key.set_repeat(250, 25)

        self.enable_ai = enable_ai
        self.width = config['cell_size'] * (config['cols'] + 8)
        self.height = config['cell_size'] * config['rows']
        self.gen = Bag7(shape_rand())
        self.hold = []
        self.next_minos = None
        self.holded = False
        self.timelapse = None
        self.lines_cleared = 0
        self.is_sprint = True if sprint > 0 else False
        self.sprint_target = sprint
        self.weights = None

        self.screen = pygame.display.set_mode((self.width, self.height))

        # if enable_ai:
        #     pygame.event.set_blocked(pygame.KEYDOWN)
        #     pygame.event.set_blocked(pygame.KEYUP)
        # do not need mouse movement
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.init_game()

    def new_mino(self, s=None):
        if s is None:
            self.mino = mino_shapes[self.gen.pop()]
            self.holded = False
        elif len(s) == 0:
            self.mino = mino_shapes[self.gen.pop()]
            self.holded = True
        else:
            self.mino = s
            self.holded = True

        self.mino_x = int(config['cols'] / 2 - len(self.mino[0]) / 2)
        self.mino_y = 0

        self.next_minos = self.gen.peek()

        if check_collision(self.board, self.mino, (self.mino_x, self.mino_y)):
            self.gameover = True

    def init_game(self):
        self.board = new_board()
        self.new_mino()
        self.hold = []
        self.holded = False

    def evaluate(self, board):
        features = [
            # board_max_height(board),
            board_total_height(board),
            complete_rows(board),
            calc_bumpiness(board),
            count_holes(board),
        ]
        if self.weights is None:
            self.weights = [_WEIGHTS[_] for _ in range(len(features))]

        res = 0.0
        for w, f in zip(self.weights, features):
            res += w * f

        return res

    @printit
    def info(self):
        return self.evaluate(self.board)

    def ai(self):
        if not self.gameover and not self.paused:
            mino_directions = [
                self.mino,
                rotate_clockwise(self.mino),
                rotate_clockwise(rotate_clockwise(self.mino)),
                rotate_counter_clockwise(self.mino),
            ]
            locks = []

            for mino in mino_directions:
                for x in range(config['cols'] - len(mino[0]) + 1):
                    for y in range(config['rows']):
                        if check_collision(self.board, mino, (x, y)):
                            joint = join_matrixes(
                                copy.deepcopy(self.board), mino, (x, y)
                            )
                            if valid_state(joint):
                                locks.append(joint)
                            break

            scores = [self.evaluate(l) for l in locks]

            if len(scores) == 0:
                self.gameover = True
            else:
                self.board = locks[scores.index(max(scores))]
                self.new_mino()
                self.clear_lines()

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = pygame.font.Font(
                pygame.font.get_default_font(), 12
            ).render(line, False, (255, 255, 255), (0, 0, 0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(
                msg_image,
                (
                    self.width // 2 - msgim_center_x,
                    self.height // 2 - msgim_center_y + i * 22,
                ),
            )

    def draw_timer(self):
        msg_image = pygame.font.Font(
            pygame.font.get_default_font(), 32
        ).render(
            time_convert(pygame.time.get_ticks()),
            False,
            (255, 255, 255),
            (0, 0, 0),
        )
        msgim_width, msgim_height = msg_image.get_size()

        self.screen.blit(
            msg_image, (self.width - msgim_width, self.height - msgim_height)
        )

    def draw_score(self):
        msg_image = pygame.font.Font(
            pygame.font.get_default_font(), 32
        ).render(str(self.lines_cleared), False, (255, 255, 255), (0, 0, 0))
        _, msgim_height = msg_image.get_size()

        self.screen.blit(msg_image, (0, self.height - msgim_height))

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        COLORS[val],
                        pygame.Rect(
                            (off_x + x) * config['cell_size'],
                            (off_y + y) * config['cell_size'],
                            config['cell_size'],
                            config['cell_size'],
                        ),
                        0,
                    )

    def draw_grid(self, board, offset):
        off_x, off_y = offset
        for y, row in enumerate(board):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (
                    off_x * config['cell_size'],
                    (off_y + y) * config['cell_size'],
                ),
                (
                    (off_x + len(row)) * config['cell_size'],
                    (off_y + y) * config['cell_size'],
                ),
            )

        for x, col in enumerate(board[0]):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (
                    (off_x + x) * config['cell_size'],
                    off_y * config['cell_size'],
                ),
                (
                    (off_x + x) * config['cell_size'],
                    (off_y + len(board)) * config['cell_size'],
                ),
            )

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.mino_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > config['cols'] - len(self.mino[0]):
                new_x = config['cols'] - len(self.mino[0])
            if not check_collision(
                self.board, self.mino, (new_x, self.mino_y)
            ):
                self.mino_x = new_x

    def quit(self):
        self.center_msg('Exiting...')
        pygame.display.update()
        sys.exit(0)

    def soft_drop(self):
        if not self.gameover and not self.paused:
            self.mino_y += 1
            if check_collision(
                self.board, self.mino, (self.mino_x, self.mino_y)
            ):
                self.board = join_matrixes(
                    self.board, self.mino, (self.mino_x, self.mino_y)
                )
                self.new_mino()
                self.clear_lines()

    def clear_lines(self):
        while True:
            for i, row in enumerate(self.board[:-1]):
                if 0 not in row:
                    self.board = remove_row(self.board, i)
                    self.lines_cleared += 1
                    if (
                        self.is_sprint
                        and self.lines_cleared >= self.sprint_target
                    ):
                        self.gameover = True
                    break
            else:
                break

    def hard_drop(self):
        if not self.gameover and not self.paused:
            while not check_collision(
                self.board, self.mino, (self.mino_x, self.mino_y)
            ):
                self.mino_y += 1
            self.board = join_matrixes(
                self.board, self.mino, (self.mino_x, self.mino_y)
            )
            self.new_mino()
            self.clear_lines()

    def rotate_right(self):
        if not self.gameover and not self.paused:
            new_mino = rotate_clockwise(self.mino)
            if not check_collision(
                self.board, new_mino, (self.mino_x, self.mino_y)
            ):
                self.mino = new_mino

    def rotate_left(self):
        if not self.gameover and not self.paused:
            new_mino = rotate_counter_clockwise(self.mino)
            if not check_collision(
                self.board, new_mino, (self.mino_x, self.mino_y)
            ):
                self.mino = new_mino

    def hold_mino(self):
        if not self.gameover and not self.paused:
            if not self.holded:
                self.mino, self.hold = self.hold, self.mino
                self.new_mino(self.mino)

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def restart_game(self):
        self.init_game()
        self.gameover = False

    def run(self):
        key_actions = {
            'ESCAPE': self.quit,
            'a': lambda: self.move(-1),
            'd': lambda: self.move(1),
            's': self.soft_drop,
            'w': self.hard_drop,
            'k': self.rotate_right,
            'm': self.rotate_left,
            'p': self.toggle_pause,
            'q': self.hold_mino,
            'i': self.info,
            'SPACE': self.start_game,
            'BACKSPACE': self.restart_game,
        }

        self.gameover = False
        self.paused = False

        pygame.time.set_timer(pygame.USEREVENT + 1, config['delay'])

        dont_burn_my_cpu = pygame.time.Clock()

        while True:
            self.screen.fill((0, 0, 0))
            if self.gameover:
                if self.timelapse is None:
                    self.timelapse = pygame.time.get_ticks()
                self.center_msg(
                    f'{time_convert(self.timelapse)} lapsed!!! '
                    'Press space to continue'
                )
            else:
                if self.paused:
                    self.center_msg('Paused')
                else:
                    self.draw_matrix(self.board, (4, 0))
                    self.draw_matrix(self.mino, (self.mino_x + 4, self.mino_y))
                    self.draw_matrix(self.hold, (0, 0))
                    for i, n in enumerate(self.next_minos):
                        self.draw_matrix(mino_shapes[n], (14, i * 5))
                    self.draw_grid(self.board, (4, 0))
                    self.draw_score()
                    self.draw_timer()

            pygame.display.update()

            if not self.enable_ai:
                for event in pygame.event.get():
                    if event.type == pygame.USEREVENT + 1:
                        self.soft_drop()
                    elif event.type == pygame.QUIT:
                        self.quit()
                    elif event.type == pygame.KEYDOWN:
                        for key in key_actions:
                            if event.key == eval('pygame.K_' + key):
                                key_actions[key]()
            else:
                self.ai()
                pygame.time.delay(AI_DELAY)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()
                    elif event.type == pygame.KEYDOWN:
                        for key in ['ESCAPE', 'p', 'i', 'SPACE', 'BACKSPACE']:
                            if event.key == eval('pygame.K_' + key):
                                key_actions[key]()

            dont_burn_my_cpu.tick(config['maxfps'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tetris')
    parser.add_argument('-ai', action='store_true', help='enable ai or not')
    parser.add_argument('-s', '--sprint', type=int, default=0, help='sprint')
    args = parser.parse_args()

    App = TetrisApp(enable_ai=args.ai, sprint=args.sprint)
    App.run()
