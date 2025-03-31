# search.py
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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

    start_state = problem.getStartState()

    # 栈中每个元素是一个三元组：(当前状态, 到达当前状态的动作列表, 总成本)
    stack = Stack()
    stack.push((start_state, [], 0))

    # 使用集合记录已访问状态，避免走重复路径
    visited = set()

    while not stack.isEmpty():
        current_state, actions, cost = stack.pop()

        # 如果当前状态是目标状态，直接返回到达该处的动作序列
        if problem.isGoalState(current_state):
            return actions

        # 避免重复访问
        if current_state not in visited:
            visited.add(current_state)

            # 遍历当前状态的所有 successor
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # 将新的状态、动作序列、代价推入栈中
                    new_actions = actions + [action]
                    stack.push((successor, new_actions, cost + step_cost))

    # 未找到目标，返回空列表
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    start_state = problem.getStartState()

    # 队列中每个元素是：(当前状态, 到达该状态的动作序列, 累计代价)
    queue = Queue()
    queue.push((start_state, [], 0))

    visited = set()  # 记录已经访问过的状态

    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()

        # 如果当前状态是目标状态，返回动作序列
        if problem.isGoalState(current_state):
            return actions

        # 跳过已访问状态
        if current_state not in visited:
            visited.add(current_state)

            # 遍历后继状态
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    new_actions = actions + [action]
                    queue.push((successor, new_actions, cost + step_cost))

    # 如果没有找到目标状态，返回空列表
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    start_state = problem.getStartState()

    # 优先队列中每个元素是：(当前状态, 到达该状态的动作序列, 当前总代价)
    queue = PriorityQueue()
    queue.push((start_state, [], 0), 0)

    # 记录已访问的状态，以及访问它们的最小代价
    visited = set()

    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()

        # 如果当前状态已经访问过，就跳过（说明曾以更优路径访问过它）
        if current_state in visited:
            continue

        visited.add(current_state)

        # 如果当前状态是目标状态，返回路径
        if problem.isGoalState(current_state):
            return actions

        # 扩展当前状态的所有后继
        for successor, action, step_cost in problem.getSuccessors(current_state):
            if successor not in visited:
                new_cost = cost + step_cost
                new_actions = actions + [action]
                queue.push((successor, new_actions, new_cost), new_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    start_state = problem.getStartState()

    # 优先队列中的元素为：(当前状态, 到达该状态的动作序列, 当前代价 g(n))
    queue = PriorityQueue()
    queue.push((start_state, [], 0), 0)

    visited = set()  # 记录已经访问过的状态

    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()

        # 如果当前状态已访问，说明之前以更小代价访问过，跳过
        if current_state in visited:
            continue

        visited.add(current_state)

        # 如果到达目标，返回动作序列
        if problem.isGoalState(current_state):
            return actions

        # 遍历后继状态
        for successor, action, step_cost in problem.getSuccessors(current_state):
            if successor not in visited:
                g_cost = cost + step_cost
                h_cost = heuristic(successor, problem)
                f_cost = g_cost + h_cost
                queue.push((successor, actions + [action], g_cost), f_cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
