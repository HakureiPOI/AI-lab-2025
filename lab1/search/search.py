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

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

from util import Stack, Queue, PriorityQueue

def initialize_frontier(strategy: str):
    """根据搜索策略初始化 frontier"""
    if strategy == 'dfs':
        return Stack()
    elif strategy == 'bfs':
        return Queue()
    elif strategy in ['ucs', 'astar']:
        return PriorityQueue()
    else:
        raise ValueError(f"Unknown strategy {strategy}")

def push_to_frontier(frontier, state, actions, cost, strategy, heuristic, problem):
    """根据策略将节点加入 frontier"""
    if strategy in ['ucs', 'astar']:
        if strategy == 'astar':
            priority = cost + heuristic(state, problem)
        else:  # ucs
            priority = cost
        frontier.push((state, actions, cost), priority)
    else:  # dfs / bfs
        frontier.push((state, actions, cost))

def searchTemplate(problem: SearchProblem, strategy: str, heuristic=nullHeuristic):
    """
    通用搜索模板，支持 DFS、BFS、UCS 和 A*。
    :param problem: 搜索问题实例
    :param strategy: 搜索策略，'dfs', 'bfs', 'ucs', 'astar'
    :param heuristic: 启发式函数，仅 A* 使用
    :return: 从起始状态到目标状态的动作序列
    """
    start_state = problem.getStartState()
    frontier = initialize_frontier(strategy)
    push_to_frontier(frontier, start_state, [], 0, strategy, heuristic, problem)

    visited = set()

    while not frontier.isEmpty():
        current_state, actions, cost = frontier.pop()

        # 如果达到目标状态，直接返回路径
        if problem.isGoalState(current_state):
            return actions

        if current_state in visited:
            continue
        visited.add(current_state)

        # 遍历后继节点
        for successor, action, step_cost in problem.getSuccessors(current_state):
            if successor not in visited:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                push_to_frontier(frontier, successor, new_actions, new_cost, strategy, heuristic, problem)

    # 如果没有找到路径，返回空列表
    return []


def breadthFirstSearch(problem: SearchProblem):
    return searchTemplate(problem, 'bfs')

def depthFirstSearch(problem: SearchProblem):
    return searchTemplate(problem, 'dfs')

def uniformCostSearch(problem: SearchProblem):
    return searchTemplate(problem, 'ucs')

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    return searchTemplate(problem, 'astar', heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
