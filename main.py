import argparse
import copy
import random
import sys

import pygame

# configurations
random.seed(42)

# board_total_height, complete_rows, calc_bumpiness, count_holes
_WEIGHTS = [-0.510066, 0.760666, -0.35663, -0.184483]
# _WEIGHTS = [0, 0, 0, 0]

CONFIG = {'cell_size': 30, 'cols': 10, 'rows': 20, 'delay': 250, 'maxfps': 60}
AI_DELAY = 0

COLOR_GRID = (50, 50, 50)
COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    3: (180, 0, 255),
    5: (0, 150, 0),
    7: (255, 0, 0),
    9: (0, 0, 255),
    11: (255, 120, 0),
    13: (0, 220, 220),
    15: (255, 255, 0),
}


# helpers
def encode_instance(obj):
    return ''.join(''.join(map(str, row)) for row in obj)


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


def gen_n_sequences(n, fix=False):
    s = [list(range(n))]
    if not fix:
        if n < 2:
            return s
        for i in range(n - 1):
            l = list(range(n))
            for j in range(i, n - 1):
                l[j], l[j + 1] = l[j + 1], l[j]
                s.append(copy.deepcopy(l))
    else:
        if n < 3:
            return s
        # fix the first hold position
        for i in range(1, n - 1):
            l = list(range(n))
            for j in range(i, n - 1):
                l[j], l[j + 1] = l[j + 1], l[j]
                s.append(copy.deepcopy(l))
    return s


# define the 7 shapes of the single mino
MINO_SHAPE_SYMBOLS = (0, 1, 3, 5, 7, 9, 11, 13, 15)
MINO_SHAPES = [
    [[0, 3, 0], [3, 3, 3]],  # T
    [[0, 5, 5], [5, 5, 0]],  # S
    [[7, 7, 0], [0, 7, 7]],  # Z
    [[9, 0, 0], [9, 9, 9]],  # J
    [[0, 0, 11], [11, 11, 11]],  # L
    [[13, 13, 13, 13]],  # I
    [[15, 15], [15, 15]],  # O
]
MINO_ROTS = {
    '030333': [
        [[0, 3, 0], [3, 3, 3]],
        [[0, 3], [3, 3], [0, 3]],
        [[3, 0], [3, 3], [3, 0]],
        [[3, 3, 3], [0, 3, 0]],
    ],
    '055550': [[[0, 5, 5], [5, 5, 0]], [[5, 0], [5, 5], [0, 5]]],
    '770077': [[[7, 7, 0], [0, 7, 7]], [[0, 7], [7, 7], [7, 0]]],
    '900999': [
        [[9, 0, 0], [9, 9, 9]],
        [[0, 9], [0, 9], [9, 9]],
        [[9, 9], [9, 0], [9, 0]],
        [[9, 9, 9], [0, 0, 9]],
    ],
    '0011111111': [
        [[0, 0, 11], [11, 11, 11]],
        [[11, 11], [0, 11], [0, 11]],
        [[11, 0], [11, 0], [11, 11]],
        [[11, 11, 11], [11, 0, 0]],
    ],
    '13131313': [[[13, 13, 13, 13]], [[13], [13], [13], [13]]],
    '15151515': [[[15, 15], [15, 15]]],
}


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
            if cell not in MINO_SHAPE_SYMBOLS:
                return False
    return True


def remove_row(board, row):
    del board[row]
    return [[0 for i in range(CONFIG['cols'])]] + board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    board = [[0 for x in range(CONFIG['cols'])] for y in range(CONFIG['rows'])]
    board += [[1 for x in range(CONFIG['cols'])]]
    return board


# heuristic utilities
def complete_rows(board):
    rv = 0
    for row in board[:-1]:
        if 0 not in row:
            rv += 1
    return rv


def board_heights(board):
    flipped = [False] * CONFIG['cols']
    rv = [0] * CONFIG['cols']
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
    for icol in range(1, CONFIG['cols']):
        rv += abs(heights[icol] - heights[icol - 1])
    return rv


def count_holes(board):
    rv = [0] * CONFIG['cols']
    prevs = [0] * CONFIG['cols']
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
        pygame.key.set_repeat(250, 25)

        self.enable_ai = enable_ai
        self.width = CONFIG['cell_size'] * (CONFIG['cols'] + 8)
        self.height = CONFIG['cell_size'] * CONFIG['rows']
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
            self.mino = MINO_SHAPES[self.gen.pop()]
            self.holded = False
        elif len(s) == 0:
            self.mino = MINO_SHAPES[self.gen.pop()]
            self.holded = True
        else:
            self.mino = s
            self.holded = True

        self.mino_x = int(CONFIG['cols'] / 2 - len(self.mino[0]) / 2)
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

        score = 0.0
        for w, f in zip(self.weights, features):
            score += w * f

        return score

    @printit
    def info(self):
        return self.evaluate(self.board)

    def get_future_sequences(self, n=3):
        combi = []
        if not self.hold:
            elements = [self.mino] + [
                MINO_SHAPES[m] for m in self.next_minos[: n - 1]
            ]
        else:
            elements = [self.mino, self.hold] + [
                MINO_SHAPES[m] for m in self.next_minos[: n - 2]
            ]
        for s in gen_n_sequences(n, fix=self.holded):
            combi.append(list(map(lambda x: elements[x], s)))
        return combi

    def ai(self):
        if not self.gameover and not self.paused:
            memoize = set()
            future_boards = []
            for idx_seq, sequence in enumerate(self.get_future_sequences()):
                prev_boards = [(copy.deepcopy(self.board), None, None)]
                for mino in sequence:
                    next_boards = []
                    for prev_board, first_board, first_mino in prev_boards:
                        for rot_mino in MINO_ROTS[encode_instance(mino)]:
                            for x in range(
                                CONFIG['cols'] - len(rot_mino[0]) + 1
                            ):
                                for y in range(CONFIG['rows']):
                                    if check_collision(
                                        prev_board, rot_mino, (x, y)
                                    ):
                                        next_board = join_matrixes(
                                            copy.deepcopy(prev_board),
                                            rot_mino,
                                            (x, y),
                                        )
                                        if valid_state(next_board):
                                            if (
                                                first_board is None
                                                and first_mino is None
                                            ):
                                                next_boards.append(
                                                    (
                                                        next_board,
                                                        next_board,
                                                        mino,
                                                    )
                                                )
                                            else:
                                                next_boards.append(
                                                    (
                                                        next_board,
                                                        first_board,
                                                        first_mino,
                                                    )
                                                )
                                        break
                    print('seq:', idx_seq)
                    print('mino:')
                    print_instance(mino)
                    print('len next:', len(next_boards))
                    print('-----')
                    prev_boards = next_boards
                    if len(next_boards) == 0:  # no need proceed to next mino
                        break
                print('len prev:', len(prev_boards))
                flag = 0
                for i, (b, *_) in enumerate(prev_boards):
                    if encode_instance(b) not in memoize:
                        flag += 1
                        memoize.add(encode_instance(b))
                        future_boards.append(prev_boards[i])
                if flag:
                    print('updated future!', flag)
                print('len future:', len(future_boards))

            if len(future_boards) == 0:
                self.gameover = True
            else:
                scores = [self.evaluate(b) for b, *_ in future_boards]
                print(len(scores))
                print('>>>>>')
                _, first_board, first_mino = future_boards[
                    scores.index(max(scores))
                ]
                if first_mino == self.mino:
                    self.board = first_board
                    self.new_mino()
                    self.clear_lines()
                else:
                    self.hold_mino()

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = pygame.font.Font(
                pygame.font.get_default_font(), 20
            ).render(line, False, (255, 255, 255), (0, 0, 0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(
                msg_image,
                (
                    self.width // 2 - msgim_center_x,
                    self.height // 2 - msgim_center_y + i * 30,
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
                            (off_x + x) * CONFIG['cell_size'],
                            (off_y + y) * CONFIG['cell_size'],
                            CONFIG['cell_size'],
                            CONFIG['cell_size'],
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
                    off_x * CONFIG['cell_size'],
                    (off_y + y) * CONFIG['cell_size'],
                ),
                (
                    (off_x + len(row)) * CONFIG['cell_size'],
                    (off_y + y) * CONFIG['cell_size'],
                ),
            )

        for x, col in enumerate(board[0]):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (
                    (off_x + x) * CONFIG['cell_size'],
                    off_y * CONFIG['cell_size'],
                ),
                (
                    (off_x + x) * CONFIG['cell_size'],
                    (off_y + len(board)) * CONFIG['cell_size'],
                ),
            )

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.mino_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > CONFIG['cols'] - len(self.mino[0]):
                new_x = CONFIG['cols'] - len(self.mino[0])
            if not check_collision(
                self.board, self.mino, (new_x, self.mino_y)
            ):
                self.mino_x = new_x

    def quit(self):
        self.center_msg('Exiting...')
        pygame.display.update()
        sys.exit(0)

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

        pygame.time.set_timer(pygame.USEREVENT + 1, CONFIG['delay'])

        dont_burn_my_cpu = pygame.time.Clock()

        while True:
            self.screen.fill((0, 0, 0))
            if self.gameover:
                if self.timelapse is None:
                    self.timelapse = pygame.time.get_ticks()
                self.center_msg(
                    f'{time_convert(self.timelapse)} lapsed! '
                    'Space to continue'
                )
            else:
                self.draw_matrix(self.board, (4, 0))
                self.draw_matrix(self.mino, (self.mino_x + 4, self.mino_y))
                self.draw_matrix(self.hold, (0, 0))
                for i, n in enumerate(self.next_minos):
                    self.draw_matrix(MINO_SHAPES[n], (14, i * 5))
                self.draw_grid(self.board, (4, 0))
                self.draw_score()
                self.draw_timer()
                if self.paused:
                    self.center_msg('Paused')

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

            dont_burn_my_cpu.tick(CONFIG['maxfps'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tetris')
    parser.add_argument('-ai', action='store_true', help='enable ai or not')
    parser.add_argument('-s', '--sprint', type=int, default=0, help='sprint')
    args = parser.parse_args()

    App = TetrisApp(enable_ai=args.ai, sprint=args.sprint)
    App.run()
