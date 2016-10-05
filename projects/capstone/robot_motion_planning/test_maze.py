import sys
import getopt
import numpy as np
from core_lib import Defs
from core_lib import Cell
from core_lib import Grid

class TestMaze:
    ''' Class that outputs the maze and path of the robot on the console '''

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.dim = int(f.next())

            walls = []
            for line in f:
                walls.append(map(int, line.split(',')))

            self.walls = np.array(walls)

    def show(self, path, dist = False):
        W = '\033[0m'  # white
        R = '\033[31m' # red
        B = '\033[1m'  # bold
        V = B+R+'|'+W

        g = Grid(self.dim)
        walls = self.walls.T
        cell_len = 5
        out = B+R+ '+' + self.dim * (cell_len * '-' + '+') + W + '\n'

        for i in range(self.dim):
            out += V

            '''
            for j in range(self.dim):
                v = walls[self.dim-i-1][j]
                fmt = '{:5s}'
                fill = ' '

                if (v & Defs.W_RIGHT) != Defs.W_RIGHT:
                    fmt += V
                else:
                    fmt += ' '

                out += fmt.format(fill)
            out += '\n' + V
            '''

            # Output motion symbol in corresponding postion:
            # ^ - up, v - down, < - left, > - right
            for j in range(self.dim):
                v = walls[self.dim-i-1][j]
                fmt = '{:5s}'
                label = ''

                if path[i][j] != None:
                    label = path[i][j].center(cell_len)
                else:
                    if dist:
                        curr = (i, j)
                        #s_to_c = g.distance((self.dim - 1, 0), curr)
                        c_to_e = g.distance_to_goal(Cell(curr))
                        label = '{:d}'.format(c_to_e)
                        label = label.center(cell_len)

                if (v & Defs.W_RIGHT) != Defs.W_RIGHT:
                    fmt += V
                else:
                    fmt += ' '
                out += fmt.format(label)

            out += '\n'+V

            for j in range(self.dim):
                v = walls[self.dim-i-1][j]
                fmt = '{:5s}'
                fill = ''

                if (v & Defs.W_BOTTOM) != Defs.W_BOTTOM:
                    fill = B+R+'_' * cell_len + W
                else:
                    fill = ' '

                if (v & Defs.W_RIGHT) != Defs.W_RIGHT:
                    fmt += V
                else:
                    fmt += ' '

                out += fmt.format(fill)

            out += '\n'

        return out

def help():
    print 'test_maze.py -m <maze_spec> [-r <cells_to_mark>]'

def main(argv):
    filename = ''
    marks = ''

    try:
        opts, args = getopt.getopt(argv, 'hm:r:', ['maze-spec=','marks='])
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
            marks = arg

    if not filename:
        help()
        sys.exit()

    m = TestMaze(filename)
    path = np.empty(m.dim * m.dim, dtype='object').reshape(m.dim, m.dim)
    show_dist = True

    if not marks:
        for i in range(m.dim):
            for j in range(m.dim):
                path[i][j] = '{},{}'.format(i,j)
    else:
        show_dist = False

        with open(marks) as f:
            cells = eval(f.read())

        cells.append((m.dim-1, 0))
        for cell in cells:
            path[cell[0]][cell[1]] = '*'

    print m.show(path, show_dist)

if __name__ == '__main__':
    main(sys.argv[1:])
