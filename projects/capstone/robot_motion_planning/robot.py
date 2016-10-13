import logging
import numpy as np
import heapq
import sys
import math
from collections import deque
from core_lib import Defs
from core_lib import Cell
from core_lib import Grid
from core_lib import LogFilter

log = logging.getLogger(__name__)
log.addFilter(LogFilter())

def init_log():
    fmt='%(asctime)s %(levelname)5s [%(name_lineno)12s] %(message)s'
    logging.basicConfig(filename='output.log', filemode='w', level=logging.DEBUG,
            format=fmt)

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        init_log()
        self.dim = maze_dim
        self.heading = Defs.NORTH
        self.grid = Grid(self.dim)
        self.path = deque()
        self.mode = Defs.START_CENTER_MODE

        self.cell = Cell((self.dim-1, 0), None, None, None, [None, None, 0, None])
        self.cell.parent = self.cell
        self.cell.set_cost(0, self.grid.distance_to_goal(self.cell.loc))
        self.grid.cells[self.cell.loc] = self.cell

        self.start = self.cell
        self.center = None
        self.start_center_path = deque()
        self.center_start_path = deque()
        self.moves = 0
        self.center_start_moves = 0
        self.start_center_moves = 0

    def select_next(self, sensors):
        ''' Select next cell to move to during maze exploration. If the grid
        provides a single cell, that would be a neighour with 1 step away.
        Otherwise, the grid would provide all cells in sorted order of the distance
        between current and next unvisited cells + f-cost of the unvisited cell.

        >>> r = Robot(12); s = [1,1,1]
        >>> r.grid.cells = { \
                (2,4):Cell((2,4), None, 17, 3, [0,0,1,0], 1, 144), \
                (2,5):Cell((2,5), None, 16, 3, [0,None,1,None], 0), \
                (3,4):Cell((3,4), None, 16, 3, [1,3,5,0], 1), \
                (3,5):Cell((3,5), None, 15, 2, [1,2,0,1], 1), \
                (3,6):Cell((3,6), None, 14, 2, [0,1,1,2], 1), \
                (3,7):Cell((3,7), None, 15, 3, [None,0,0,3], 0), \
                (4,4):Cell((4,4), None, 11, 2, [2,2,4,0], 1), \
                (4,5):Cell((4,5), None, 12, 1, [0,1,0,1], 1), \
                (4,6):Cell((4,6), None, 13, 1, [1,0,0,2], 1), \
                (5,4):Cell((5,4), None, 10, 1, [3,0,3,0], 1), \
                (6,4):Cell((6,4), None, 9,  1, [4,0,2,0], 1), \
                (7,4):Cell((7,4), None, 8,  2, [5,2,1,1], 1), \
                (8,4):Cell((8,4), None, 9,  3, [6,None,0,None], 0) }
        >>> r.grid.unvisited = {\
                (2,5):r.grid[(2,5)], (3,7):r.grid[(3,7)], (8,4):r.grid[(8,4)] }
        >>> r.cell = r.grid[(3,4)]
        >>> p = r.select_next(s); r.grid.coord(p)
        [(3, 5), (2, 5)]
        >>> _ = r.grid.unvisited.pop((2,5))
        >>> p = r.select_next(s); r.grid.coord(p)
        [(3, 5), (3, 6), (3, 7)]
        >>> _ = r.grid.unvisited.pop((3,7))
        >>> p = r.select_next(s); r.grid.coord(p)
        [(4, 4), (5, 4), (6, 4), (7, 4), (8, 4)]
        '''
        path = self.grid.build_path_on_deadend(self.cell, sensors)

        if len(path) > 0:
            log.info('Deadend: {}. Escape path: {}'.format(
                self.cell, self.grid.coord(path)))
            return path

        cells = self.grid.get_unvisited(self.cell)
        tree = {self.cell.loc : (0, self.cell)}
        paths = deque()

        for end in cells:
            success = self.grid.build_tree(self.cell, end, tree, 1, self.mode)

            if success:
                def selector(cell): return tree[cell.loc][1]
                path = self.grid.follow_parent(self.cell, end, selector)
                paths.append(path)

        path = deque()

        if len(paths) > 0:
            def selector(path):
                weight = 1.5
                end = path[len(path)-1]
                h_cost = self.grid.distance_to_goal(end.loc)
                cost = 0 if h_cost == 0 else len(path) + weight * h_cost

                log.debug('From {} to {}, steps: {}, h-cost: {}, cost: {}, '\
                        'path : {}'.format(self.cell.loc, end.loc, len(path),
                            h_cost, cost, self.grid.coord(path)))
                return cost

            path = sorted(paths, key=selector)[0]

            log.debug('Selected path from {}: {}'.format(self.cell.loc,
                self.grid.coord(path)))

        return path

    def next_move(self, sensors):
        ''' Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        self.grid.on_visit(self.cell, self.heading, sensors, self.mode)

        if self.grid.distance_to_goal(self.cell.loc) == 0:
            if self.on_goal_reached():
                return 'Reset','Reset'

        if len(self.path) > 0:
            cell = self.path.popleft()
        else:
            self.path = self.select_next(sensors)

            if len(self.path) > 0:
                cell = self.path.popleft()
            else:
                log.error('No more cells to visit. Current: {}'.format(self.cell))
                sys.exit(-1)

        rotation, movement = self.move_instr(cell)
        self.moves += 1

        return rotation, movement

    def move_instr(self, next_cell):
        ''' Create an instruction to move to next cell and update robot's
        parameters: current cell, heading and steps taken.

        Examples:
        Robot heading North 0:
            (7, 4) -> (6, 4): (-1,0) -> North -> (  0, 1)
            (7, 4) -> (7, 5): (0, 1) -> East  -> ( 90, 1)
            (7, 4) -> (8, 4): (1, 0) -> South -> (  0,-1)
            (7, 4) -> (7, 3): (0,-1) -> West  -> (-90, 1)

        Robot heading East  1:
            (7, 4) -> (6, 4): (-1,0) -> North -> (-90, 1)
            (7, 4) -> (7, 5): (0, 1) -> East  -> (  0, 1)
            (7, 4) -> (8, 4): (1, 0) -> South -> ( 90, 1)
            (7, 4) -> (7, 3): (0,-1) -> West  -> (  0,-1)

        >>> r= Robot(12)
        >>> r.heading= Defs.NORTH; r.cell= Cell((7,4)); r.move_instr(Cell((7,5)))
        (90, 1)
        >>> r.heading= Defs.SOUTH; r.cell= Cell((7,4)); r.move_instr(Cell((6,4)))
        (0, -1)
        >>> r.heading= Defs.WEST; r.cell= Cell((7,4)); r.move_instr(Cell((8,4)))
        (-90, 1)
        >>> r.heading= Defs.EAST; r.cell= Cell((7,4)); r.move_instr(Cell((6,4)))
        (-90, 1)
        >>> r.heading= Defs.EAST; r.cell= Cell((7,4)); r.move_instr(Cell((7,5)))
        (0, 1)
        >>> r.heading= Defs.EAST; r.cell= Cell((7,4)); r.move_instr(Cell((8,4)))
        (90, 1)
        >>> r.heading= Defs.EAST; r.cell= Cell((7,4)); r.move_instr(Cell((7,3)))
        (0, -1)
        >>> r.heading= Defs.NORTH; r.cell= Cell((11,0)); r.move_instr(Cell((9,0)))
        (0, 2)
        >>> r.heading= Defs.SOUTH; r.cell= Cell((11,1)); r.move_instr(Cell((11,4)))
        (-90, 3)
        '''
        (y1, x1), (y2, x2) = self.cell.loc, next_cell.loc
        yd, xd = (y2 - y1), (x2 - x1)
        ys = int(math.copysign(1, yd)) if yd != 0 else 0
        xs = int(math.copysign(1, xd)) if xd != 0 else 0

        new_heading = Defs.HEADINGS[(ys, xs)]
        rotation, movement, hoffset = Defs.MOVES[(new_heading - self.heading) % 4]
        movement = abs(yd + xd) * movement

        self.cell = next_cell
        self.heading = (self.heading + hoffset) % 4

        log.info('Next move (rotation, movement, heading): {}'.format(
                (rotation, movement, Defs.HEADINGSS[self.heading])))

        return rotation, movement
        
    def save_path(self, start_loc, end, path):
        ''' Save the shortest path for the current phase.
        '''
        start = self.grid[start_loc]
        path.extend(self.grid.follow_parent(start, self.cell))

    def reset(self, next_cell, goals):
        ''' Reset robot and grid to perform next phase of navigation.
        '''
        self.grid.reset()
        self.grid.set_goals(goals)
        self.start = self.cell
        self.start.parent = self.start
        self.start.set_cost(0, self.grid.distance_to_goal(self.start.loc))
        next_cell.parent = self.start
        next_cell.set_cost(0, self.grid.distance_to_goal(next_cell.loc))
        self.path = deque([next_cell])

    def on_goal_reached(self):
        ''' Since the robot reached its destination, change the robot's strategy.
        The strategy can be: (a) explore the maze further (b) reset to the starting
        point to run along the determined root with as fewer steps as possible.
        '''
        result = False
        path = deque()

        if self.mode == Defs.START_CENTER_MODE:
            self.save_path(self.start.loc, self.cell, self.start_center_path)
            self.start_center_moves = self.moves
            self.center = self.cell

            next_cell = self.cell.parent
            self.reset(next_cell, [self.start.loc])
            self.start.visits = 1
            self.path = deque([next_cell])
            self.mode = Defs.CENTER_START_MODE
            self.moves = 0
        elif self.mode == Defs.CENTER_START_MODE:
            self.save_path(self.start.loc, self.cell, self.center_start_path)
            self.center_start_path.pop()
            self.center_start_path.reverse()
            self.center_start_path.append(self.center)
            self.center_start_moves = self.moves

            next_cell = self.cell.parent
            self.reset(next_cell, None)
            self.start.visits = 0
            self.heading = Defs.NORTH
            self.mode = Defs.RUN_MODE
            self.moves = 0
            self.optimize_path()
            result = True
        elif self.mode == Defs.RUN_MODE:
            result = True
        return result

    def optimize_path(self):
        scp = self.grid.coord(self.start_center_path)
        csp = self.grid.coord(self.center_start_path)
        shp = self.select_short_legs(scp, csp) 
        opp = self.merge_steps(shp)

        self.path.clear()
        for loc in opp:
            self.path.append(self.grid[loc])

        log.info('Goals reached. '\
                'OpM: {}, OpL: {}, OpP: {}, '\
                'ShM: {}, ShL: {}, ShP: {}, '\
                'ScM: {}, ScL: {}, ScP: {}, '\
                'CsM: {}, CsL: {}, CsP: {}'.format(
            len(opp), len(opp), opp,
            len(shp), len(shp), shp,
            self.start_center_moves, len(scp), scp,
            self.center_start_moves, len(csp), csp))

    def select_short_legs(self, a, b):
        ''' When paths intersect at a cell, one leg leading to that cell can
        be more efficient. Here select the most efficient leg.

        >>> r = Robot(12)
        >>> a = [(10, 0), (9, 0), (8, 0), (7, 0), (6, 0), (5, 0), (5, 1), (5, 2), \
                (6, 2), (6, 3), (7, 3), (7, 4), (7, 5), (7, 6), (8, 6), (8, 5),\
                (9, 5), (9, 6), (10, 6), (10, 7), (11, 7), (11, 8), (11, 9),\
                (11, 10), (11, 11), (10, 11), (9, 11), (8, 11), (7, 11), (6, 11),\
                (5, 11), (4, 11), (4, 10), (5, 10), (6, 10), (6, 9), (5, 9),\
                (5, 8), (6, 8), (6, 7), (5, 7), (5, 6)]
        >>> b = [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 2), (11, 3),\
                (11, 4), (10, 4), (9, 4), (9, 5), (10, 5), (11, 5), (11, 6),\
                (10, 6), (10, 7), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),\
                (10, 11), (9, 11), (8, 11), (8, 10), (8, 9), (8, 8), (7, 8),\
                (6, 8), (6, 7), (5, 7), (5, 6)]
        >>> r.select_short_legs(a,b)
        [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 2), (11, 3), (11, 4), (10, 4), (9, 4), (9, 5), (9, 6), (10, 6), (10, 7), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (10, 11), (9, 11), (8, 11), (8, 10), (8, 9), (8, 8), (7, 8), (6, 8), (6, 7), (5, 7), (5, 6)]
        '''
        max_path, min_path = (a,b) if len(a) > len(b) else (b,a)
        max_start, min_start = 0, 0
        locs = []

        for max_end in range(len(max_path)):
            loc = max_path[max_end]

            if loc in min_path:
                min_end = min_path.index(loc)
                max_delta, min_delta = max_end - max_start, min_end - min_start

                if max_delta >= min_delta:
                    locs.extend(min_path[min_start:min_end]) 
                else:
                    locs.extend(max_path[max_start:max_end]) 

                max_start, min_start = max_end, min_end

        locs.append(min_path[len(min_path)-1])
        return locs

    def merge_steps(self, locations):
        ''' Merge the steps of single-step path so as to travel up to 3 steps
        at a time.

        >>> r = Robot(12)
        >>> locs = [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 2),\
                (11, 3), (11, 4), (10, 4), (9, 4), (9, 5), (9, 6), (10, 6),\
                (10, 7), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (10, 11),\
                (9, 11), (8, 11), (8, 10), (8, 9), (8, 8), (7, 8), (6, 8), (6, 7),\
                (5, 7), (5,6)]
        >>> r.merge_steps(locs)
        [(9, 0), (9, 1), (11, 1), (11, 4), (9, 4), (9, 6), (10, 6), (10, 7), (11, 7), (11, 10), (11, 11), (8, 11), (8, 8), (6, 8), (6, 7), (5, 7), (5, 6)]
        '''

        new_path = []
        start = (self.dim-1, 0)
        priv_loc = start
        priv_heading = Defs.NORTH
        steps = 0

        for i,loc in enumerate(locations):
            steps += 1 
            heading = Defs.HEADINGS[(loc[0] - priv_loc[0], loc[1] - priv_loc[1])]

            if priv_heading != heading or steps == 3:
                new_path.append(priv_loc)
                priv_heading = heading
                start = priv_loc
                steps = 0
            priv_loc = locations[i]

        new_path.append(locations[len(locations)-1])
        return new_path

if __name__ == '__main__':
    import doctest
    r = Robot(12)
    doctest.run_docstring_examples(r.move_instr, globals())
