# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, math, util

from game import Agent, Actions

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # total score of eval fxn
        total_score = 0.0

        # make sure pacman grabs food nearby if no ghost
        newGhostPos = [ghost.getPosition() for ghost in newGhostStates]
        old_food = currentGameState.getFood()
        old_food = old_food.asList()
        for food in old_food:
            if food == newPos and (newPos not in newGhostPos):
                total_score += 1000000000000000000
        
        # find closest food
        closest_food = (0,0)
        closest_food_dist = -1
        for food in newFood.asList():
            dist = manhattanDistance(newPos, food)
            if dist < closest_food_dist or closest_food_dist == -1:
                closest_food_dist = dist
                closest_food = food

        # adjust to prefer positions closer to closest food
        total_score += 10000/closest_food_dist
        
        #if ghost is close don't go there
        for ghost in newGhostPos:
            if manhattanDistance(newPos, ghost) <= 1:
                total_score -= 100000000000


        """
        # adjust to prefer positions further from ghost
        ghost_dists = [manhattanDistance(newPos, ghost) for ghost in newGhostPos]
        closest_ghost_dist = min(ghost_dists)
        total_score -= closest_ghost_dist
        """

        """
        # if pacman closer to food than food to ghost, go toward food
        food_is_closer = True
        for ghost in newGhostPos:
            if manhattanDistance(newPos, closest_food) < manhattanDistance(closest_food, ghost):
                food_is_closer = False

        
        if food_is_closer:
            total_score += 10000000
        """

        """
        for ghost in newGhostStates:
            total_score = total_score + (4.0*manhattanDistance(newPos, ghost.getPosition()))
        
        for food in newFood:
            total_score = total_score + (3.0/manhattanDistance(newPos, food))
        """

        return total_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(self.index, self.depth, gameState)[1]

    def minimax(self, agentIndex, depth, gameState):
        # terminal node
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        # get all successors
        successors = []
        for action in gameState.getLegalActions(agentIndex):
            successors.append((gameState.generateSuccessor(agentIndex, action), action))
        # get scores of all successors
        vals = []
        for successor, action in successors:
            if (agentIndex + 1 == gameState.getNumAgents()):
                vals.append((self.minimax(0, depth - 1, successor)[0], action))
            else:
                vals.append((self.minimax(agentIndex + 1, depth, successor)[0], action))
        # if pacman, choose max score; if ghost, choose min score
        if agentIndex == 0:
            return max(vals, key=lambda x : x[0])
        else:
            return min(vals, key=lambda x : x[0])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(self.index, self.depth, gameState, -math.inf, math.inf)[1]

    def alphabeta(self, agentIndex, depth, gameState, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # Terminal / Leaf Node of the tree
            return (self.evaluationFunction(gameState), None)
        if agentIndex == 0:
            # Pacman's turn - Maximize Pacman's score
            v = (-math.inf, None)
            # Iterate over possible actions for Pacman
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                successorVal = (self.alphabeta(agentIndex + 1, depth, successor, alpha, beta)[0],
                                action)
                v = max(v, successorVal, key=lambda x : x[0])
                if v[0] > beta:
                    return v
                alpha = max(alpha, v[0])
            return v
        else:
            # Ghost's turn - Minimize Pacman score
            v = (math.inf, None)
            # Iterate over possible actions for the Ghost
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex + 1 == gameState.getNumAgents()):
                    # Curent ghost is the last agent to make a move,
                    #   so start a new turn.
                    successorVal = (self.alphabeta(0, depth - 1, successor, alpha, beta)[0],
                                    action)
                else:
                    successorVal = (self.alphabeta(agentIndex + 1, depth, successor, alpha, beta)[0],
                                    action)
                v = min(v, successorVal, key=lambda x : x[0])
                if v[0] < alpha:
                    return v
                beta = min(beta, v[0])
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(self.index, self.depth, gameState)[1]
        
    def expectimax(self, agentIndex, depth, gameState):
        # terminal node
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        # get all successors
        successors = []
        for action in gameState.getLegalActions(agentIndex):
            successors.append((gameState.generateSuccessor(agentIndex, action), action))
        # get scores of all successors
        vals = []
        for successor, action in successors:
            if (agentIndex + 1 == gameState.getNumAgents()):
                vals.append((self.expectimax(0, depth - 1, successor)[0], action))
            else:
                vals.append((self.expectimax(agentIndex + 1, depth, successor)[0], action))
        # if pacman, choose max score and corresponding action
        if agentIndex == 0:
            return max(vals, key=lambda x : x[0])
        # if ghost, average values and return first action (doesn't matter)
        else:
            i = 0
            for x in vals:
                i += x[0]
            return (i/len(vals), vals[0][1])
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The less food outstanding and the higher the current score, the higher
    the value. Grabbing the powerup and going closer to the closest food is encouraged, 
    and going near a ghost is discouraged unless you grab the powerup, which in that case
    going near a ghost is encouraged.
    """
    "*** YOUR CODE HERE ***"
    score = 0
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghost = currentGameState.getGhostStates()
    ghostList = [g.getPosition() for g in ghost]
    scaredTimes = [g.scaredTimer for g in ghost]

    # better to grab powerup
    if sum(scaredTimes) > 0:
        score += 10000

    # the less food outstanding the better
    score -= len(foodList)*1000

    # find closest food
    closest_food = (0,0)
    closest_food_dist = -1
    for food in foodList:
        dist = mazeDistance(pos, food, currentGameState)
        if dist < closest_food_dist or closest_food_dist == -1:
            closest_food_dist = dist
            closest_food = food

    # prefer positions closer to closest food
    score += 10/closest_food_dist
        
    # try not to go too close to ghost when no power up
    # favor chasing ghost if powered up
    for ghost in ghostList:
        #x = int(ghost[0])
        #y = int(ghost[1])
        #ghostPos = (x,y)
        ghost_dist = manhattanDistance(pos, ghost)
        #ghost_dist = mazeDistance(pos, ghostPos, currentGameState)
        if ghost_dist <= 1 and sum(scaredTimes) == 0:
            score -= 10000
        if sum(scaredTimes) > 0:
            score -= ghost_dist

    # higher score is better
    score += currentGameState.getScore()/10


    return score

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actions = []
    closed = set()
    state = problem.getStartState()
    #each node has current state and actions up to the state
    node = [state, actions]

    #create fringe queue and add first node
    fringe = util.Queue()
    fringe.push(node)

    #run to completion
    while(True):
        if fringe.isEmpty():
            return []
        #node is first node popped from fringe
        node = fringe.pop()
        #currentState is state of node
        currentState = node[0]
        
        #if at goal return actions
        if problem.isGoalState(currentState):
            return node[1];
        #if current node not previously expanded push child notes to fringe
        if currentState not in closed:
            closed.add(currentState)
            for child in problem.getSuccessors(currentState):
                new_node = [child[0], (node[1] + [child[1]])]
                fringe.push(new_node)

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

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

# Abbreviation
better = betterEvaluationFunction
