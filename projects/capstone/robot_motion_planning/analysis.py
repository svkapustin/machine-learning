import sys
import getopt
import re
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

    def __init__(self, maze_spec):
        with open(maze_spec, 'rb') as f:
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

def stats(log_file, show_marks, path, dim):
    fields = {
            'OpM':0, 'OpL':1, 'OpP':2,
            'ShM':3, 'ShL':4, 'ShP':5,
            'ScM':6, 'ScL':7, 'ScP':8,
            'CsM':9, 'CsL':10, 'CsP':11}

    rvisiting = re.compile('Visiting: Cell\((\(\d+, \d+\)),.*mode: (\d+)')
    rdisc = re.compile('New cell.*mode: (\d+)')
    rgoal = re.compile('Goals.* '\
            'OpM: (\d+), OpL: (\d+), OpP: (.*), '\
            'ShM: (\d+), ShL: (\d+), ShP: (.*), '\
            'ScM: (\d+), ScL: (\d+), ScP: (.*), '\
            'CsM: (\d+), CsL: (\d+), CsP: (.*)')
    discovered = {
            Defs.START_CENTER_MODE:0,
            Defs.CENTER_START_MODE:0,
            Defs.RUN_MODE:0 }
    visited = {
            Defs.START_CENTER_MODE:[],
            Defs.CENTER_START_MODE:[],
            Defs.RUN_MODE:[] }
    goal = ()

    with open(log_file) as f:
        for line in f:
            r = rdisc.search(line)
            if r:
                discovered[int(r.group(1))] += 1
                continue
            r = rvisiting.search(line)
            if r:
                visited[int(r.group(2))].append(r.group(1))
                continue
            r = rgoal.search(line)
            if r:
                goal = r.groups()

    if not goal:
        raise Exception('Goal not found')

    idx = fields.get(show_marks, fields['OpP'])
    cells = eval(goal[idx])
    for cell in cells:
        path[cell[0]][cell[1]] = '*'

    total = float(dim**2)
    modes = ['Center', 'Origin', 'Optimal']
    disc, visits, moves, scores = [], [], [], ['','','']

    for i in range(3):
        disc.append(100 * discovered[i]/total)
        visits.append(100 * len(set(visited[i]))/total)
        moves.append(len(visited[i]))
        scores.append(.0)
    visits[2] += 1
    scores[2] = '{:5.1f}'.format(1./30 * (moves[0] + moves[1]) + moves[2])

    out = '\nTotal cells: {}\n'.format(int(total))
    out +='\nMode    | Discovered | Visited | Moves | Score '
    out +='\n--------+------------+---------+-------+-------'
    for i in range(3):
        out += '\n{:7s} | {:9.1f}% | {:6.1f}% | {:5} | {}'.format(
            modes[i], disc[i], visits[i], moves[i], scores[i])
    print out

def help():
    print '''
    analysis.py -m <maze_spec> [ -l <log> -r (OhP | ShP | ScP | CsP) | -d | -c ) ]

    -m - maze specification file
    -l - log file
    -r - show marks (requires log parameter). Options:
         OhP - optimal path 
         ShP - non-shorted path
         ScP - start-to-center path 
         CsP - center-to-start path
    -d - show distance to centre
    -c - show coordinates
    -h - this help
    '''

def main(argv):
    maze_spec = ''
    log_file = ''
    show_marks = ''
    show_dist = False
    show_coord = False

    try:
        opts, args = getopt.getopt(argv, 'hm:l:r:dc')
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt in ("-m"):
            maze_spec = arg
        elif opt in ("-l"):
            log_file = arg
        elif opt in ("-r"):
            show_marks = arg
        elif opt in ("-d"):
            show_dist = True
        elif opt in ("-c"):
            show_coord = True

    if not maze_spec:
        help()
        sys.exit()

    m = TestMaze(maze_spec)
    path = np.empty(m.dim * m.dim, dtype='object').reshape(m.dim, m.dim)

    if log_file:
        stats(log_file, show_marks, path, m.dim)
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
