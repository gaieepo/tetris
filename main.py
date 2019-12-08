import random
import sys

import pygame

# configurations
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


def printit(func):
    def f(*args, **kwargs):
        rv = func(*args, **kwargs)
        print(rv)
        return rv

    return f


# define the shapes of the single mino
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
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(250, 25)

        self.width = config['cell_size'] * (config['cols'] + 8)
        self.height = config['cell_size'] * config['rows']
        self.gen = Bag7(shape_rand())
        self.hold = []
        self.next_minos = None
        self.holded = False

        self.screen = pygame.display.set_mode((self.width, self.height))
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

    def ai(self):
        print('board max height:', board_max_height(self.board))
        print('board total height:', board_total_height(self.board))
        print('complete rows:', complete_rows(self.board))
        print('bumpiness:', calc_bumpiness(self.board))
        print('holes:', count_holes(self.board))

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
        sys.exit()

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
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
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
            while True:
                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = remove_row(self.board, i)
                        break
                else:
                    break

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
            'i': self.ai,
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
                self.center_msg('Game Over!!! Press space to continue')
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

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.soft_drop()
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in key_actions:
                        if event.key == eval('pygame.K_' + key):
                            key_actions[key]()

            dont_burn_my_cpu.tick(config['maxfps'])


if __name__ == "__main__":
    App = TetrisApp()
    App.run()
