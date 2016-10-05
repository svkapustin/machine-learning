import numpy as np
import heapq
import sys
import copy
from collections import deque

import logging
log = logging.getLogger(__name__)

class Defs:
    # Wall indicators. 
    W_TOP, W_RIGHT, W_BOTTOM, W_LEFT= 1, 2, 4, 8
    NORTH, EAST, SOUTH, WEST        = 0, 1, 2, 3
    LEFT, CENTER, RIGHT             = 0, 1, 2 
    START_CENTRE_PHASE                = 0
    CENTRE_START_PHASE                = 1
    RUN_PHASE                       = 2
    HEADINGS = {(-1,0):NORTH, (0,1):EAST, (1,0):SOUTH, (0,-1):WEST}
    MOVES = [ (0,1,0), (90,1,1), (0,-1,0), (-90,1,-1) ]
    HEADINGSS = ['N','E','S','W']

################################################################################

class Cell:
    def __init__(self, loc, parent = None, g_cost = None, f_cost = None,
            viable = [None,None,None,None], visits=0, deadend=0):
        self.loc = loc
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.viable = copy.deepcopy(viable)
        self.visits = visits
        self.deadend = deadend

    def __repr__(self):
        p = None

        if self.parent != None:
            p = self.parent.loc

        return 'Cell({}, {}, {}, {}, {}, {}, {})'.format(
                self.loc, p, self.g_cost, self.f_cost, self.viable, self.visits, 
                self.deadend)

    def __eq__(self, other):
        '''
        >>> a= Cell((3,4), None, 5, 2, [0,1,1,0]); b= Cell((3,4), None, 1, 8)
        >>> a == b
        True
        '''
        if isinstance(other, self.__class__):
            return self.loc == other.loc
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_path_defined(self):
        '''
        >>> a= Cell((3,4), None, 5, 2, [0,1,1,0]); a.is_path_defined()
        True
        >>> a.viable[3] = None; a.is_path_defined()
        False
        '''
        return None not in self.viable

    def set_cost(self, g_cost, f_cost):
        self.g_cost = g_cost
        self.f_cost = f_cost

################################################################################

class Grid:
    def __init__(self, dim):
        self.dim = dim
        self.cells = {}
        self.goals = []
        self.unvisited = {}
        self.set_goals()
    
    def __setitem__(self, loc, cell):
        self.cells[loc] = cell

    def __getitem__(self, loc):
        return self.cells.get(loc, None)

    def coord(self, cells):
        ''' Represent a cell as a location.
        '''
        return [cell.loc for cell in cells]

    def set_goals(self, goals=None):
        ''' Set goal to the maze centre or to the parameter goals. The goal format is
        cell location tuple, (y,x).
        '''
        if goals == None:
            a = self.dim / 2 - 1
            b = self.dim / 2
            self.goals = [ [a, a], [a, b], [b, b], [b, a] ]
            self.cells[(self.dim-1, 0)] = Cell((self.dim-1, 0))
        else:
            self.goals = goals

    def reset(self):
        ''' Reset cells' parameters except for viable and deadend.'''
        for k,c in self.cells.items():
            c.visits = 0
            c.parent = None
            c.set_cost(None, None)
        self.unvisited.clear()

    def distance(self, source_loc, target_loc):
        ''' Calculate Manhattan distance. '''
        (y1, x1) = source_loc
        (y2, x2) = target_loc
        return abs(x1 - x2) + abs(y1 - y2)

    def distance_to_goal(self, source):
        ''' Calculate the distance from source to goal. Since the centre goal may
        occupy 4 locations, provide the minimum distance. '''
        distances = []
        for goal_loc in self.goals:
            distances.append(self.distance(source.loc, goal_loc))
        return min(distances)

    def get_unvisited(self, source):
        ''' Return unvisited cells in sorted order according to distance from
        the source and cells' f-cost.

        >>> g= Grid(12)
        >>> g.unvisited = {\
                (2,5):Cell((2,5), None, 16, 3, [0,None,1,None], 0), \
                (3,7):Cell((3,7), None, 15, 3, [None,0,0,3], 0), \
                (8,4):Cell((8,4), None, 9,  3, [6,None,0,None], 0) }
        >>> source = Cell((3,4)); g.get_unvisited(source)
        [(2, 5), (3, 7), (8, 4)]
        >>> source = Cell((5,4)); g.get_unvisited(source)
        [(8, 4), (2, 5), (3, 7)]
        >>> source = Cell((3,6)); g.get_unvisited(source)
        [(3, 7), (2, 5), (8, 4)]
        '''
        def selector(k):
            cell = self.unvisited[k]
            dist = self.distance(source.loc, cell.loc)
            return dist + cell.f_cost

        cells = sorted(self.unvisited, key=selector)
        return cells

    def set_cost(self, cell_from, cells):
        ''' Calculate g-cost from cell_from to each cell in list. Update g-cost if
        it's lower from the previous one. Also, update the f-cost if g-cost is
        updated. '''
        for cell_to in cells:
            new_g_cost = cell_from.g_cost + 1

            if cell_to.g_cost == None or new_g_cost < cell_to.g_cost:
                cell_to.parent = cell_from
                heuristic = self.distance_to_goal(cell_to)
                cell_to.set_cost(new_g_cost, new_g_cost + heuristic)

    def on_visit(self, cell, heading, sensors):
        ''' Called by robot when it visits a cell. Here the cell's viable paths
        are updated. Also, cell's neighbours are created and added to
        unvisited-cell map if the neighbour is never visited before.

        >>> cell = Cell((11,0), None, 0, 10); cell.viable = [11,0,0,0]
        >>> g, sensors, heading = Grid(12), [0, 11, 0], Defs.NORTH
        >>> g.on_visit(cell, heading, sensors)
        >>> cell.viable[Defs.WEST] == 0
        True
        >>> cell.viable[Defs.NORTH] == 11
        True
        >>> cell.viable[Defs.EAST] == 0
        True
        >>> cell = Cell((9,0), None, 2, 8, [9,3,2,0]); g[(9,0)] = cell
        >>> sensors, heading = [9, 3, 2], Defs.EAST
        >>> g.on_visit(cell, heading, sensors)
        >>> cell.viable[Defs.NORTH] == 9
        True
        >>> cell.viable[Defs.EAST] == 3
        True
        >>> cell.viable[Defs.SOUTH] == 2
        True
        '''
        cell.visits += 1
        self.unvisited.pop(cell.loc, None)
        offset = (heading - 1)

        for direction in range(len(sensors)):
            polar_heading = offset % 4
            cell.viable[polar_heading] = sensors[direction]
            offset += 1

        # Add/update neighbours.
        neighbours = self.neighbours(cell, False, True)

        for neighbour in neighbours:
            self.unvisited[neighbour.loc] = neighbour
            log.debug('Unvisited: {}'.format(neighbour))

    def neighbours(self, cell, add_visited, set_cost):
        ''' Create a list of nodes relative to the current node's position. Depending
        on add_visisted and set_cost parameters, filter out neighbours already
        visisted and update their cost.

        >>> g = Grid(12)
        >>> cell = Cell((11,0), None, 0, 10, [11,0,0,0])
        >>> cells = g.neighbours(cell, False, True)
        >>> cells[0].loc == (10, 0)
        True
        >>> cell = Cell((9,0), None, 2, 8, [9,3,2,0])
        >>> cells = g.neighbours(cell, False, True)
        >>> cells[0].loc == (8, 0)
        True
        >>> cells[1].loc == (9, 1)
        True
        >>> cells[2].loc == (10, 0)
        True
        '''
        (y, x) = cell.loc
        locs = [ (y-1, x), (y, x+1), (y+1, x), (y, x-1) ]
        cells = []

        for polar_heading in range(len(cell.viable)):
            if cell.viable[polar_heading] > 0:
                loc = locs[polar_heading]
                neighbour = self.get_cell(cell, polar_heading, loc)

                if neighbour.deadend == 0:
                    if neighbour.visits == 0:
                        cells.append(neighbour)
                    elif add_visited:
                        cells.append(neighbour)

        if set_cost and len(cells) > 0:
            self.set_cost(cell, cells)

        return cells

    def get_cell(self, cell, polar_heading, loc):
        ''' Provide an existing cell or create a new one.

        >>> g = Grid(12); curr = Cell((9,0), None, 0, 0, [9,3,2,0])
        >>> g.get_cell(curr, Defs.NORTH, (8,0))
        Cell((8, 0), (9, 0), None, None, [8, None, 3, None], 0, 0)
        >>> g.get_cell(curr, Defs.EAST, (9,1))
        Cell((9, 1), (9, 0), None, None, [None, 2, None, 1], 0, 0)
        >>> #for c in g.cells.values(): print(c) 
        '''
        neighbour = self.cells.get(loc, None)

        if neighbour == None:
            neighbour = Cell(loc, cell)
            self.cells[loc] = neighbour
            oppos_heading = (polar_heading + 2) % 4

            if cell.viable[oppos_heading] != None and \
                cell.viable[polar_heading] != None:
                neighbour.viable[oppos_heading] = cell.viable[oppos_heading] + 1
                neighbour.viable[polar_heading] = cell.viable[polar_heading] - 1
        else:
            if not neighbour.parent:
                neighbour.parent = cell

        return neighbour

    def build_reverse_path(self, start, end):
        ''' Use parent cells to build a path from an end to start cell.
        '''
        path = deque([end])
        parent = end.parent
        cells = dict()
        
        while parent != None and parent != start:
            if parent.loc in cells:
                log.error('Duplicate parent: {}'.format(parent))
                break
            else:
                cells[parent.loc] = parent
            path.appendleft(parent)
            parent = parent.parent

        if parent == start:
            return path
        else:
            log.error('Failed to reverse to start cell. ' \
                    'Last: {}, start: {}'.format(parent,start))
            path.clear()
            return path

    def build_reverse_path_from_tree(self, start, end, tree):
        ''' Use parent cells to build a path from an end to start cell.
        '''
        path = deque([end])
        parent = tree[end.loc][1]
        cells = dict()
        
        while parent != None and parent != start:
            if parent.loc in cells:
                log.error('Duplicate parent: {}'.format(parent))
                break
            else:
                cells[parent.loc] = parent
            path.appendleft(parent)
            parent = tree[parent.loc][1]

        if parent == start:
            return path
        else:
            log.error('Failed to build path to start. ' \
                    'Cell: {}, start: {}'.format(parent,start))
            path.clear()
            return path
