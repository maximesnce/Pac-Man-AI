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
import numpy as np

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


        "*** YOUR CODE HERE ***"
        newfood_list = newFood.asList()
        liste_score = []

        if len(newfood_list) == currentGameState.getFood().count():  
            liste_score.append(100000)
            for food in newfood_list :
                liste_score.append(manhattanDistance(food , newPos)) 
            
            score = - min(liste_score)

        else:
            score = 0

        for ghost in newGhostStates:  
            score -= 5**(1 - manhattanDistance(ghost.getPosition(), newPos))
        return score



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
        
        legal_actions = gameState.getLegalActions(0)

        list_value_actions = []
        next_move = None
        

        for move in legal_actions : 
            list_value_actions.append(self.Min_Evaluation(gameState.generateSuccessor(0, move), 1, 0))

        max_value = max(list_value_actions)
        next_move = legal_actions[list_value_actions.index(max_value)]

        return next_move

        
    def Min_Evaluation (self, gameState, agentIndex, depth):

        if (len(gameState.getLegalActions(agentIndex)) == 0): 
            return self.evaluationFunction(gameState)

        if (agentIndex < gameState.getNumAgents() - 1):
            list_evaluation = []
            for move in gameState.getLegalActions(agentIndex) :
                list_evaluation.append(self.Min_Evaluation(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth))
            min_value = min(list_evaluation)
            return min_value

        else:
            list_evaluation = [] 
            for move in gameState.getLegalActions(agentIndex) :
                list_evaluation.append(self.Max_Evaluation(gameState.generateSuccessor(agentIndex, move), depth + 1))
            min_value = min(list_evaluation)
            return min_value


    def Max_Evaluation(self, gamestate, depth) :

        if (depth == self.depth or len(gamestate.getLegalActions(0)) == 0) :
            return self.evaluationFunction(gamestate)
        else :
            list_evaluation = []
            for move in gamestate.getLegalActions(0) :
                list_evaluation.append(self.Min_Evaluation(gamestate.generateSuccessor(0, move), 1, depth))

            max_value = max(list_evaluation)
            return max_value




        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -np.inf
        beta = np.inf
        value_action = -np.inf

        legal_actions = gameState.getLegalActions(0)
        next_move = None

        for move in legal_actions : 
            value_action = self.Min_Evaluation(gameState.generateSuccessor(0, move), 1, 0, alpha, beta)
            if alpha < value_action :
                alpha = value_action
                next_move = move
        
        return next_move

        util.raiseNotDefined()

    def Min_Evaluation (self, gameState, agentIndex, depth, alpha, beta):
    
        if (len(gameState.getLegalActions(agentIndex)) == 0): 
            return self.evaluationFunction(gameState)

        action_value = np.inf
        legal_action = gameState.getLegalActions(agentIndex)

        for move in legal_action : 
            if (agentIndex < gameState.getNumAgents() - 1):
                action_value = min(action_value, self.Min_Evaluation(gameState.generateSuccessor(agentIndex,move), agentIndex + 1, depth, alpha, beta))
            
            else:
                action_value = min(action_value, self.Max_Evaluation(gameState.generateSuccessor(agentIndex, move), depth + 1, alpha, beta))


            if alpha > action_value :
                return action_value

            beta = min(beta, action_value)

        return action_value


    def Max_Evaluation(self, gamestate, depth, alpha, beta) :

        if (depth == self.depth or len(gamestate.getLegalActions(0)) == 0) :
            return self.evaluationFunction(gamestate)
        
        action_value = -np.inf
        legal_action = gamestate.getLegalActions(0)
        for move in legal_action :
            action_value = max(action_value, self.Min_Evaluation(gamestate.generateSuccessor(0, move), 1, depth, alpha, beta))
            

            if beta < action_value :
                return action_value
                
            alpha = max(alpha, action_value)
        
        return action_value


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
        legal_actions = gameState.getLegalActions(0)
        list_value_actions = []
        next_move = None
        

        for move in legal_actions : 
            list_value_actions.append(self.Min_Expected_Evaluation(gameState.generateSuccessor(0, move), 1, 0))

        max_value = max(list_value_actions)
        next_move = legal_actions[list_value_actions.index(max_value)]

        return next_move

    def Min_Expected_Evaluation(self, gameState, agentIndex, depth) :
        
        if (len(gameState.getLegalActions(agentIndex)) == 0): 
            return self.evaluationFunction(gameState)

        if (agentIndex < gameState.getNumAgents() - 1):
            list_evaluation = []
            for move in gameState.getLegalActions(agentIndex) :
                list_evaluation.append(self.Min_Expected_Evaluation(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth))
            expected_value = sum(list_evaluation)/(len(gameState.getLegalActions(agentIndex)))
            return expected_value

        else:
            list_evaluation = [] 
            for move in gameState.getLegalActions(agentIndex) :
                list_evaluation.append(self.Max_Expected_Evaluation(gameState.generateSuccessor(agentIndex, move), depth + 1))
            expected_value = sum(list_evaluation)/(len(gameState.getLegalActions(agentIndex)))
            return expected_value

    def Max_Expected_Evaluation(self, gamestate, depth) :

        if (depth == self.depth or len(gamestate.getLegalActions(0)) == 0) :
            return self.evaluationFunction(gamestate)
        
        else :
            list_evaluation = []
            for move in gamestate.getLegalActions(0) :
                list_evaluation.append(self.Min_Expected_Evaluation(gamestate.generateSuccessor(0, move), 1, depth))

            max_value = max(list_evaluation)
            return max_value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    GhostStates = currentGameState.getGhostStates()
    PacmanPosition = currentGameState.getPacmanPosition()

    food_list = (currentGameState.getFood()).asList() 
    capsule_list = currentGameState.getCapsules()

    len_food_list = len(food_list)
    len_capsule_list = len(capsule_list)

    score = 0

    if currentGameState.getNumAgents() > 1:
        distance_ghosts = []
        for ghost in GhostStates :
            distance_ghosts.append(manhattanDistance(PacmanPosition, ghost.getPosition()))
        min_distance_ghost = min(distance_ghosts)
        if (min_distance_ghost <= 1):
            return -10000
        score -= 1.0/min_distance_ghost


    curFood = PacmanPosition  
    for food in food_list:
        closestFood = min(food_list, key=lambda x: manhattanDistance(x, curFood))
        score += 1.0/(manhattanDistance(curFood, closestFood))
        curFood = closestFood
        food_list.remove(closestFood)


    current_capsule = PacmanPosition
    for capsule in capsule_list:
        closest_capsule = min(capsule_list, key=lambda x: manhattanDistance(x, current_capsule))
        score += 1.0/(manhattanDistance(current_capsule, closest_capsule))
        current_capsule = closest_capsule
        capsule_list.remove(closest_capsule)

    
    score += 8*(currentGameState.getScore())


    score -= 6*(len_food_list + len_capsule_list)

    return score

# Abbreviation
better = betterEvaluationFunction
