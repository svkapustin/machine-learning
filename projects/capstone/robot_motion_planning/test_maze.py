import sys
import getopt
import numpy as np
from core_lib import Defs
from core_lib import Cell
from core_lib import Grid

W = '\033[0m'  # white
R = '\033[31m' # red
Y = '\033[33m' # yellow
B = '\033[1m'  # bold
V = B+R+'|'+W
cell_len = 5
coord_len = 3
# Wall indicators. 
W_TOP, W_RIGHT, W_BOTTOM, W_LEFT= 1, 2, 4, 8

class TestMaze:
    ''' Class that outputs the maze and one of path of the robot on the console,
    distance to center, coordinates, empty cells.
    '''

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.dim = int(f.next())

            walls = []
            for line in f:
                walls.append(map(int, line.split(',')))

            self.walls = np.array(walls)

    def show(self, path):
        walls = self.walls.T

        out = '\n'
        out += ' ' * (coord_len + 1)
        for i in range(self.dim):
            out += '{:^6d}'.format(i)
        out += '\n'
        out += ' '*coord_len + B+R+ '+' + self.dim*(cell_len*'-' + '+') + W + '\n'

        for i in range(self.dim):
            out += '{:^3d}'.format(i) + V

            for j in range(self.dim):
                v = walls[self.dim-i-1][j]
                fmt = '{:5s}'
                label = ''

                if path[i][j] != None:
                    label = Y+ path[i][j].center(cell_len) +W

                if (v & W_RIGHT) != W_RIGHT:
                    fmt += V
                else:
                    fmt += ' '
                out += fmt.format(label)

            out += '\n'+ ' '*coord_len + V

            for j in range(self.dim):
                v = walls[self.dim-i-1][j]
                fmt = '{:5s}'
                fill = ''

                if (v & W_BOTTOM) != W_BOTTOM:
                    fill = B+R+'_' * cell_len + W
                else:
                    fill = ' '

                if (v & W_RIGHT) != W_RIGHT:
                    fmt += V
                else:
                    fmt += ' '
                out += fmt.format(fill)

            out += '\n'

        return out

def help():
    print '''
    test_maze.py -m <maze_spec> [-r <cells_to_mark> | -d | -c]

    -m|--maze-spec= - maze specification file
    -r|--marks=     - show the marks from specified file, which is to contain
        python list with tuples as (y,x) coordinates. Example file content:
        [(11,0), (5,6), (5,7)]
    -d|--distance=  - show distance to centre
    -c|--coord=     - show coordinates
    -h              - this help
    '''

def main(argv):
    filename = ''
    show_marks = ''
    show_dist = False
    show_coord = False

    try:
        opts, args = getopt.getopt(argv, 'hm:r:dc',
                ['maze-spec=','marks=','dist=','coord='])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt in ("-m", "--maze-spec"):
            filename = arg
        elif opt in ("-r", "--marks"):
            show_marks = arg
        elif opt in ("-d", "--dist"):
            show_dist = True
        elif opt in ("-c", "--coord"):
            show_coord = True

    if not filename:
        help()
        sys.exit()

    m = TestMaze(filename)
    path = np.empty(m.dim * m.dim, dtype='object').reshape(m.dim, m.dim)

    if show_marks:
        with open(show_marks) as f:
            cells = eval(f.read())

        cells.append((m.dim-1, 0))
        for cell in cells:
            path[cell[0]][cell[1]] = '*'
    elif show_coord:
        for i in range(m.dim):
            for j in range(m.dim):
                path[i][j] = '{},{}'.format(i,j)
    elif show_dist:
        g = Grid(m.dim)

        for i in range(m.dim):
            for j in range(m.dim):
                c_to_e = g.distance_to_goal((i, j))
                path[i][j] = Y+ '{:d}'.format(c_to_e).center(cell_len) +W

    print m.show(path)

if __name__ == '__main__':
    main(sys.argv[1:])
