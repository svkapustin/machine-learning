import numpy as np
import heapq
import sys
import copy
from collections import deque
import logging

class LogFilter(logging.Filter):
    def filter(self, record):
        record.name_lineno = "%s:%d" % (record.name, record.lineno)
        return True

log = logging.getLogger(__name__)
log.addFilter(LogFilter())

class Defs:
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

    def distance_to_goal(self, source_loc):
        ''' Calculate the distance from source to goal. Since the centre goal may
        occupy 4 locations, provide the minimum distance. '''
        distances = []
        for goal_loc in self.goals:
            distances.append(self.distance(source_loc, goal_loc))
        return min(distances)

    def get_unvisited(self, source):
        ''' Return a list of unvisited cell locations.
        '''
        cells = []
        out = ''
        for key,cell in self.unvisited.iteritems():
            out += '\n\t{}'.format(cell)
            cells.append(cell)
        log.debug('Unvisited cells: {}'.format(out))

        return cells

    def set_cost(self, cell_from, cells):
        ''' Calculate g-cost from cell_from to each cell in list. Update g-cost if
        it's lower from the previous one. Also, update the f-cost if g-cost is
        updated. '''
        for cell_to in cells:
            new_g_cost = cell_from.g_cost + 1

            if cell_to.g_cost == None or new_g_cost < cell_to.g_cost:
                cell_to.parent = cell_from
                heuristic = self.distance_to_goal(cell_to.loc)
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

        log.info('Visiting: {}, heading: {}, sensors: {}'.format(
            cell, Defs.HEADINGSS[heading], sensors))

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
        else:
            if not neighbour.parent:
                neighbour.parent = cell

        oppos_heading = (polar_heading + 2) % 4

        if cell.viable[oppos_heading] != None and \
            cell.viable[polar_heading] != None:
            neighbour.viable[oppos_heading] = cell.viable[oppos_heading] + 1
            neighbour.viable[polar_heading] = cell.viable[polar_heading] - 1

        return neighbour

    def build_tree(self, start, end, tree, steps):
        ''' Create a tree that represents a path from start to end cell.

        >>> c34 = Cell((3,4), None, 0, 0, [1,3,5,0])
        >>> c35 = Cell((3,5), None, 0, 0, [1,2,0,1])
        >>> c36 = Cell((3,6), None, 0, 0, [0,1,1,2])
        >>> c37 = Cell((3,7), None, 0, 0, [1,0,0,3])
        >>> c24 = Cell((2,4), None, 0, 0, [0,0,6,0]); c24.deadend=12*12
        >>> c25 = Cell((2,5), None, 0, 0, [0,3,1,0])
        >>> c26 = Cell((2,6), None, 0, 0, [2,2,0,1])
        >>> c27 = Cell((2,7), None, 0, 0, [0,1,1,2])
        >>> c28 = Cell((2,8), None, 0, 0, [1,0,0,3])
        >>> c16 = Cell((1,6), None, 0, 0, [1,0,1,3])
        >>> c17 = Cell((1,7), None, 0, 0, [1,None,0,None]) # end cell; unexplored
        >>> c18 = Cell((1,8), None, 0, 0, [0,2,1,1])
        >>> c06 = Cell((0,6), None, 0, 0, [0,1,2,3])
        >>> c07 = Cell((0,7), None, 0, 0, [0,2,1,4])
        >>> g = Grid(12)
        >>> g.cells = { \
                                          c06.loc:c06, c07.loc:c07,\
                                          c16.loc:c16, c17.loc:c17, c18.loc:c18,\
                c24.loc:c24, c25.loc:c25, c26.loc:c26, c27.loc:c27, c28.loc:c28,\
                c34.loc:c34, c35.loc:c35, c36.loc:c36, c37.loc:c37 }
        >>> tree = {c34.loc:(0,c34)}
        >>> g.build_tree(c34, c17, tree, 1)
        True
        >>> def selector(cell): return tree[cell.loc][1]
        >>> p = g.follow_parent(c34, c17, selector)
        >>> g.coord(p)
        [(3, 5), (3, 6), (3, 7), (2, 7), (2, 8), (1, 8), (1, 7)]
        '''
        success = False

        if end.loc in tree and steps > tree[end.loc][0]:
            return success

        cells = self.neighbours(start, True, False)

        for cell in cells:
            if cell.loc in tree and steps > tree[cell.loc][0]:
                continue
            if cell == end:
                success = True
                tree[cell.loc] = (steps, start)
                break
            if cell.is_path_defined() == False:
                continue

            tree[cell.loc] = (steps, start)

            success |= self.build_tree(cell, end, tree, steps + 1)
        return success

    def follow_parent(self, start, end, parent_selector=None):
        ''' Use parent cells to build a path from an end to start cell.
        '''
        if not parent_selector:
            def selector(cell): return cell.parent
            parent_selector = selector

        path = deque([end])
        cells = dict()
        parent = parent_selector(end)

        while parent != None and parent != start:
            if parent.loc in cells:
                log.error('Duplicate parent: {}'.format(parent))
                break
            else:
                cells[parent.loc] = parent
            path.appendleft(parent)
            parent = parent_selector(parent)

        if parent == start:
            return path
        else:
            log.error('Failed to build path to start. ' \
                    'Cell: {}, start: {}'.format(parent,start))
            path.clear()
            return path

    def build_path_on_deadend(self, cell, sensors):
        ''' Called by robot. If it is determined that the cell is a deadend, set
        deadend score on this cell to a maximum number of cells in this grid plus
        one. Propagate the status with decreasing score to all parents that have
        at least 2 walls closed (or less than 3 walls open).

        >>> g = Grid(12)
        >>> c1 = Cell(None, None,0,0,[1,5,1,0])
        >>> c2 = Cell(None, c1,  0,0,[0,0,2,1])
        >>> c3 = Cell(None, c2,  0,0,[2,1,0,0])
        >>> c4 = Cell(None, c3,  0,0,[1,0,1,0])
        >>> c5 = Cell(None, c4,  0,0,[0,1,2,0])
        >>> c6 = Cell(None, c5,  0,0,[0,0,0,1])
        >>> g.build_path_on_deadend(c6, [0,0,0])[0]
        Cell(None, None, 0, 0, [0, 1, 2, 0], 0, 144)
        >>> c6
        Cell(None, None, 0, 0, [0, 0, 0, 1], 0, 145)
        >>> c2
        Cell(None, None, 0, 0, [0, 0, 2, 1], 0, 141)
        >>> c1
        Cell(None, None, 0, 0, [1, 5, 1, 0], 0, 0)
        '''
        path = deque()

        if None not in cell.viable:
            if sum(sensors) == 0:
                cell.deadend = self.dim * self.dim + 1
                deadend_score = cell.deadend

                while cell.parent != None:
                    path.append(cell.parent)

                    if cell.parent.viable.count(0) >= 2:
                        deadend_score -= 1
                        cell.parent.deadend = deadend_score
                        cell = cell.parent
                    else:
                        break
        return path
