# search.py
# ---------
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack

    open_list = Stack() # empty stack
    visited_list = [] # empty list for visited nodes
    path = [] # empty list for our agent's path
    action_cost = 0  # cost of each movement starting at 0

    # Get state space start position
    start_position = problem.getStartState() 

    # Push the start position onto the stack
    open_list.push((start_position, path, action_cost))

    # While the stack is not empty...
    while not open_list.isEmpty():

        # Pop the top node from the stack
        current_node = open_list.pop()

        # Set the current position
        position = current_node[0]

        # Set the path
        path = current_node[1]

        # Push the current position to the visited list if it is not visited
        if position not in visited_list:
            visited_list.append(position)

        # Returns the final path if the current position is goal
        if problem.isGoalState(position):
            return path

        # Gets successors of the current node
        successors = problem.getSuccessors(position)

        # Pushes the current node's successors to the stack if they are not visited
        for item in successors:
            if item[0] not in visited_list:
                new_position = item[0]
                new_path = path + [item[1]]
                open_list.push((new_position, new_path, item[2]))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    open_list = Queue() # empty queue
    visited_list = [] # empty list for visited nodes
    path = [] # empty list for our agent's path
    action_cost = 0  # cost of each movement starting at 0

    # Get state space start position
    start_position = problem.getStartState()

    # Push the start position to the Queue
    open_list.push((start_position, path, action_cost))

    # While the queue is not empty...
    while not open_list.isEmpty():

        # Pop the node in front of the queue
        current_node = open_list.pop()

        # Set the current position
        position = current_node[0]

        # Set the path
        path = current_node[1]

        # Push the current position to the visited list if it is not visited
        if position not in visited_list:
            visited_list.append(position)

        # Returns the final path if the current position is goal
        if problem.isGoalState(position):
            return path

        # Gets successors of the current node
        successors = problem.getSuccessors(position)

        # Push the current node's successors to the Queue if they are not visited
        for item in successors:
            # Check visited and open list
            if item[0] not in visited_list and item[0] not in (node[0] for node in open_list.list):
                new_position = item[0]
                new_path = path + [item[1]]
                open_list.push((new_position, new_path, item[2]))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    open_list = PriorityQueue() # empty priority queue
    visited_list = [] # empty list for visited nodes
    path = [] # empty list for our agent's path
    priority = 0 # set priority to 0

    # Get state space start position
    start_position = problem.getStartState()

    # Push the start position to the PriorityQueue
    open_list.push((start_position, path), priority)

    # While the priority queue is not empty...
    while not open_list.isEmpty():

        # Pop the node in front of the queue
        current_node = open_list.pop()

        # Set the current position
        position = current_node[0]

        # Set the path
        path = current_node[1]

        # Pushes the current position to the visited list if it is not visited
        if position not in visited_list:
            visited_list.append(position)

        # Returns the final path if the current position is goal
        if problem.isGoalState(position):
            return path

        # Gets successors of the current node
        successors = problem.getSuccessors(position)

        # Gets the priority of an existing node in the open list
        def getPriorityOfNode(priority_queue, node):
            for item in priority_queue.heap: # iterate through priority queue
                if item[2][0] == node: # until we reach the desired node
                    return problem.getCostOfActions(item[2][1]) # return the priority of the node

        # Push the current node's successors to the PriorityQueue if they are not visited
        for item in successors:
            # Check visited and open list
            if item[0] not in visited_list and (item[0] not in (node[2][0] for node in open_list.heap)):
                new_path = path + [item[1]]
                new_priority = problem.getCostOfActions(new_path)
                open_list.push((item[0], new_path), new_priority)

            # If the successor is already in the open list, we check its priority
            elif item[0] not in visited_list and (item[0] in (node[2][0] for node in open_list.heap)):
                old_priority = getPriorityOfNode(open_list, item[0])
                new_priority = problem.getCostOfActions(new_path)

                # Updates priority of the successor if the value of new priority is less than that of the old one
                if old_priority > new_priority:
                    new_path = path + [item[1]]
                    open_list.update((item[0], new_path), new_priority)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    open_list = PriorityQueue() # empty priority queue
    visited_list = [] # empty list for visited nodes
    path = [] # empty list for our agent's path
    priority = 0 # set priority to 0

    # Get state space start position
    start_position = problem.getStartState()

    # Pushes the start position to the PriorityQueue
    open_list.push((start_position, path), priority)

    # While the priority queue is not empty...
    while not open_list.isEmpty():

        # Pop the node in front of the queue
        current_node = open_list.pop()

        # Set the current position
        position = current_node[0]

        # Set the path
        path = current_node[1]

        # Returns the final path if the current position is goal
        if problem.isGoalState(position):
            return path

        # Push the current position to the visited list if it is not visited
        if position not in visited_list:
            visited_list.append(position)

            # Gets successors of the current node
            successors = problem.getSuccessors(position)

            # Push the current node's successors to the PriorityQueue if they are not visited
            for item in successors:
                if item[0] not in visited_list:
                    new_position = item[0]
                    new_path = path + [item[1]]

                    # Update priority of the successor using f(n) function
                    """ g(n): Current cost from start state to the current position. """
                    g = problem.getCostOfActions(new_path)

                    """ h(n): Estimate of the lowest cost from the current position to the goal state. """
                    h = heuristic(new_position, problem)

                    """ f(n): Estimate of the lowest cost of the solution path
                              from start state to the goal state passing through the current position """
                    f = g + h

                    new_priority = f
                    open_list.push((new_position, new_path), new_priority)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
