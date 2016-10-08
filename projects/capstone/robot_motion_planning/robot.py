import logging
import numpy as np
import heapq
import sys
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
        self.mode = Defs.START_CENTRE_PHASE
        self.steps = 0
        self.cell = Cell((self.dim-1, 0), None, None, None, [None, None, 0, None])
        self.cell.parent = self.cell
        self.cell.set_cost(0, self.grid.distance_to_goal(self.cell.loc))
        self.grid.cells[self.cell.loc] = self.cell
        self.start = self.cell
        self.centre = None
        self.start_centre_path = deque()
        self.centre_start_path = deque()

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
            success = self.grid.build_tree(self.cell, end, tree, 1)

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
        self.grid.on_visit(self.cell, self.heading, sensors)

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

        >>> r= Robot(12); r.cell= Cell((7,4));
        >>> r.heading= Defs.NORTH; r.move_instr(Cell((7,5)))
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
        '''
        (y1, x1), (y2, x2) = self.cell.loc, next_cell.loc
        new_heading = Defs.HEADINGS[ (y2 - y1, x2 - x1) ]
        rotation, movement, hoffset = Defs.MOVES[(new_heading - self.heading) % 4]

        self.cell = next_cell
        self.heading = (self.heading + hoffset) % 4
        self.steps += abs(movement)

        log.info('Next move (rotation, movement, heading): {}'.format(
                (rotation, movement, Defs.HEADINGSS[self.heading])))

        return rotation, movement
        
    def save_path(self, start_loc, end, path):
        ''' Save the shortest path for the current phase.
        '''
        start = self.grid[start_loc]
        path.extend(self.grid.follow_parent(start, self.cell))

        log.info('Steps {}, goal: {}, path: \n{}'.format(
            len(path), end, self.grid.coord(path)))

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
        log.info('Reached goal: {}. Mode: {}, steps: {}'.format(
                self.cell, self.mode, self.steps))

        if self.mode == Defs.START_CENTRE_PHASE:
            self.save_path(self.start.loc, self.cell, self.start_centre_path)
            self.centre = self.cell
            next_cell = self.cell.parent
            self.reset(next_cell, [self.start.loc])
            self.start.visits = 1
            self.path = deque([next_cell])
            self.mode = Defs.CENTRE_START_PHASE
            return False
        elif self.mode == Defs.CENTRE_START_PHASE:
            self.save_path(self.start.loc, self.cell, self.centre_start_path)
            next_cell = self.cell.parent
            self.reset(next_cell, None)
            self.start.visits = 0
            self.heading = Defs.NORTH
            self.mode = Defs.RUN_PHASE
            self.steps = 0

            if len(self.start_centre_path) > len(self.centre_start_path):
                self.path = self.centre_start_path
                self.path.pop()
                self.path.reverse()
                self.path.append(self.centre)
            else:
                self.path = self.start_centre_path
            return True
        elif self.mode == Defs.RUN_PHASE:
            return True

if __name__ == '__main__':
    main()
