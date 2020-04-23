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
import random, util

from game import Agent

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

        # Code
        newGhostPos = successorGameState.getGhostPositions()
        newScore = successorGameState.getScore() - currentGameState.getScore()

        if len(newFood.asList()) == 0:
            fScore = 0
        else:
            fScore = 1/min(util.manhattanDistance(newPos, f) for f in newFood.asList())


        gScore = 0
        closeg = min([util.manhattanDistance(newPos, g) for g in newGhostPos])
        if closeg < 2:
            gScore = -500

        return newScore + fScore + gScore


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

        depth = self.depth * gameState.getNumAgents()
        stop = Directions.STOP

        LegalActions = gameState.getLegalActions(0)
        if stop in LegalActions:
            LegalActions.remove(stop)
        successors = []
        for action in LegalActions:
            successors.append(gameState.generateSuccessor(0, action))

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        v = []
        for state in successors:
            v.append(self.act(state, depth - 1, 1))
        maxV = max(v)

        maxI = []
        for i in range(0, len(v)):
            if v[i] == maxV: maxI.append(i)
        return LegalActions[random.choice(maxI)]

    def act(self, state, depth, agentIndex):
        numAgents = state.getNumAgents()

        if depth == 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        LegalActions = state.getLegalActions(agentIndex)
        if Directions.STOP in LegalActions: LegalActions.remove(Directions.STOP)
        successors = [state.generateSuccessor(agentIndex, action) for action in LegalActions]
        if agentIndex == 0:
            return max([self.act(state, depth - 1, (agentIndex + 1) % numAgents) for state in successors])
        else:
            return min([self.act(state, depth - 1, (agentIndex + 1) % numAgents) for state in successors])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        v1, v2 = float("-inf"), float("inf")
        agents = gameState.getNumAgents()
        depth = self.depth * agents
        stop = Directions.STOP

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        LegalActions = gameState.getLegalActions(0)
        if stop in LegalActions:
            LegalActions.remove(stop)

        best = None
        bestVal = float("-inf")

        for action in LegalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.act(successor, depth - 1,(1) % agents, v1, v2)
            if value > bestVal:
                bestVal, best = value, action
            v1 = max(v1, value)
        return best

    def act(self, state, depth, agentIndex, a=float("-inf"), b=float("inf")):
        numAgents = state.getNumAgents()
        if (state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state)
        LegalActions = state.getLegalActions(agentIndex)
        if Directions.STOP in LegalActions: LegalActions.remove(Directions.STOP)
        if agentIndex == 0:
            v = float("-inf")
            for action in LegalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, self.act(successor, depth - 1, (agentIndex + 1)% numAgents, a, b))
                if v > b: return v
                a = max(a, v)
            return v
        else:
            v = float("inf")
            for action in LegalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, self.act(successor, depth - 1, (agentIndex + 1) % numAgents, a, b))
                if v < a:
                    return v
                b = min(b, v)
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

        class Action:
            def __init__(self, val, action):
                self.val = val
                self.action = action

        def value(state, agentIndex, depth):

            if depth == 0 and agentIndex == 0:
                return Action(self.evaluationFunction(state), None)
            if agentIndex == 0:

                return max_Value(state, agentIndex, depth - 1)
            return exp_Value(state, agentIndex, depth)


        def max_Value(state, agentIndex, depth):
            mav = Action(float("-inf"), None)
            if len(state.getLegalActions(agentIndex)) == 0: return value(state, 0, 0)
            for action in state.getLegalActions(agentIndex):
                v = value(state.generateSuccessor(agentIndex, action),(agentIndex + 1) % state.getNumAgents(), depth)
                if v.val > mav.val: mav = Action(v.val, action)
            return mav


        def exp_Value(state, agentIndex, depth):
            n = len(state.getLegalActions(agentIndex))
            if n == 0: return value(state, 0, 0)
            val = sum([value(state.generateSuccessor(agentIndex, action), (agentIndex + 1) % state.getNumAgents(), depth).val
                       for action in state.getLegalActions(agentIndex)]) / n
            return Action(val, None)

        return value(gameState, 0, self.depth).action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    power = currentGameState.getCapsules()
    ghostpos = currentGameState.getGhostPositions()

    foodCo = 1
    powerCo = 1
    ghostCo = 1

    dist = food.width + food.height
    closef = (min([util.manhattanDistance(pos, f) for f in food.asList()] + [dist]))
    closeg = (min([util.manhattanDistance(pos, ghost) for ghost in ghostpos]))
    closep = (min([len(power)] + [util.manhattanDistance(powerpos, pos) for powerpos in power]))


    if len(food.asList()) == 0:
        foodScore = 1
    else:
        foodScore = 1/closef

    if len(power) == 0:
        powerScore = 1
    else:
        powerScore = 1 / closep

    if closeg < 1:
        ghostScore = -100
    else:
        ghostScore = 1 / closeg

    result = foodCo * foodScore + ghostCo * ghostScore + powerCo * powerScore + score
    return result


# Abbreviation
better = betterEvaluationFunction
