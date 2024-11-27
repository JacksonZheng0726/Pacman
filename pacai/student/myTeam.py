from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
from pacai.core.actions import Actions


def createTeam(firstIndex, secondIndex, isRed,
        first = 'OffensiveAgent',
        second = 'DefensiveAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indices.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffensiveAgent(firstIndex),
        DefensiveAgent(secondIndex),
    ]

from pacai.util import priorityQueue

def aStarSearch(gameState, start, goal, agentIndex, heuristic):
    """
    A* Search implementation to compute the shortest path to the goal.
    """
    walls = gameState.getWalls()
    pq = priorityQueue.PriorityQueue()
    pq.push((start, [], 0), 0)  # (current position, actions, current cost)

    visited = set()

    while not pq.isEmpty():
        currentPosition, actions, currentCost = pq.pop()

        if currentPosition in visited:
            continue
        visited.add(currentPosition)

        # Check if we reached the goal
        if currentPosition == goal:
            return actions  # Return the list of actions

        # Expand neighbors
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            nextPosition = (int(currentPosition[0] + dx), int(currentPosition[1] + dy))

            if not walls[nextPosition[0]][nextPosition[1]]:  # Check if the next position is valid
                newCost = currentCost + 1
                heuristicCost = heuristic(nextPosition, goal)
                pq.push((nextPosition, actions + [direction], newCost), newCost + heuristicCost)

    return []  # Return empty list if no path is found

def manhattanHeuristic(position, goal):
    """
    Manhattan distance heuristic for A* Search.
    """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

import random
class QLearningCaptureAgent(CaptureAgent):
    def __init__(self, index, alpha=0.5, gamma=0.9, epsilon=0.1):
        super().__init__(index)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.qValues = {}  # Q-Value table
    
    def getQValue(self, state, action):
        """
        Get the Q-value for a given state and action.
        Ensure state is converted to a hashable tuple.
        """
        if isinstance(state, dict):  # If already processed as a dictionary
            hashableState = tuple(state.items())
        else:
            hashableState = state  # Assume state is already processed as hashable

        return self.qValues.get((hashableState, action), 0.0)

    
    def computeActionFromQValues(self, gameState):
        """
        Compute the best action based on Q-values for a given state.
        """
        stateFeatures = self.getStateFeatures(gameState)  # Extract features from gameState
        hashableState = tuple(stateFeatures.items())  # Convert features to a hashable tuple

        # Fetch legal actions from the gameState
        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return None

        maxQValue = float('-inf')
        bestActions = []
        for action in legalActions:
            qValue = self.getQValue(hashableState, action)  # Use hashableState here
            if qValue > maxQValue:
                maxQValue = qValue
                bestActions = [action]
            elif qValue == maxQValue:
                bestActions.append(action)

        return random.choice(bestActions) if bestActions else None
    

    def getBoundaryPosition(self, gameState):
        """
        Get a list of positions on the boundary that the agent can move to.
        """
        walls = gameState.getWalls()
        layoutWidth = walls._width
        boundaryX = layoutWidth // 2
        if not self.red:
            boundaryX -= 1

        # Collect all accessible boundary positions
        boundaryPositions = [
            (boundaryX, y)
            for y in range(walls.height)
            if not walls[boundaryX][y]
        ]

        # Return the closest boundary position based on the agent's current location
        myPos = gameState.getAgentState(self.index).getPosition()
        return min(boundaryPositions, key=lambda pos: self.getMazeDistance(myPos, pos))


    def chooseAction(self, gameState):
        """
        Choose an action based on Q-values and game state.
        """
        # Use the original gameState to get legal actions
        actions = gameState.getLegalActions(self.index)
        
        # Simplified state for Q-learning (feature extraction)
        simplifiedState = self.getStateFeatures(gameState)

        # Use epsilon-greedy policy for action selection
        if random.random() < self.epsilon:
            # Exploration: choose a random legal action
            chosenAction = random.choice(actions)
        else:
            # Exploitation: choose the best action based on Q-values
            maxQValue = float('-inf')
            chosenAction = None
            for action in actions:
                qValue = self.getQValue(simplifiedState, action)
                if qValue > maxQValue:
                    maxQValue = qValue
                    chosenAction = action

        # Update Q-values based on the chosen action
        successor = self.getSuccessor(gameState, chosenAction)
        nextSimplifiedState = self.getStateFeatures(successor)
        reward = self.getReward(gameState, successor)
        self.update(simplifiedState, chosenAction, nextSimplifiedState, reward)

        return chosenAction


    def update(self, state, action, nextState, reward):
        """
        Update Q-values using the Q-learning update rule.
        """
        if not isinstance(state, dict):
            state = self.getStateFeatures(state)  # Extract features if raw state passed

        if not isinstance(nextState, dict):
            nextState = self.getStateFeatures(nextState)  # Extract features if raw state passed

        hashableState = tuple(state.items())
        hashableNextState = tuple(nextState.items())

        oldQValue = self.getQValue(hashableState, action)
        nextValue = self.computeValueFromQValues(nextState)
        self.qValues[(hashableState, action)] = (
            (1 - self.alpha) * oldQValue + self.alpha * (reward + self.gamma * nextValue)
        )

    
    def getStateFeatures(self, gameState):
        """
        Extract relevant features for Q-learning from the game state.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(gameState).asList()
        foodDist = min([self.getMazeDistance(myPos, food) for food in foodList]) if foodList else 0
        capsules = self.getCapsules(gameState)

        return {
            'foodCarried': self.foodCarried,
            'foodDist': foodDist,
            'capsules': len(capsules),
            'isHome': self.isInHomeTerritory(myPos, gameState),
        }


"""
class OffensiveAgent(CaptureAgent):

    def __init__(self, index):
        super().__init__(index)
        self.foodCarried = 0  # Tracks food collected by the agent

    def chooseAction(self, gameState):

        # Get all legal actions
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        maxEval = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            evalScore = self.evaluate(successor)
            if evalScore > maxEval:
                maxEval = evalScore
                bestAction = action

        return bestAction

    
    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        return successor

        walls = gameState.getWalls()
        layoutWidth = walls._width
        boundaryX = layoutWidth // 2
        if not self.red:  # Adjust for blue team
            boundaryX -= 1

        if self.red:
            return position[0] < boundaryX
        else:
            return position[0] > boundaryX

    def evaluate(self, successor):
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 1. Food distance (minimize distance to the nearest food)
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            foodDist = min([self.getMazeDistance(myPos, food) for food in foodList])
        else:
            foodDist = 0

        # 2. Capsule distance (minimize distance to the nearest capsule)
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            capsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
        else:
            capsuleDist = 100  # No capsules nearby

        # 3. Ghost avoidance (penalize being too close to ghosts)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        if len(ghosts) > 0:
            ghostDistances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            ghostDist = min(ghostDistances)
            if ghostDist <= 2:
                ghostPenalty = -200  # Strong penalty for being close
            elif ghostDist <= 5:
                ghostPenalty = -50  # Moderate penalty for being nearby
            else:
                ghostPenalty = 0
        else:
            ghostPenalty = 0

        # 5. Encourage capsule consumption when ghosts are close
        if len(capsules) > 0 and ghostPenalty < -50:  # Ghosts are nearby
            capsuleIncentive = 300 - 5 * capsuleDist  # Strongly incentivize eating capsules
        else:
            capsuleIncentive = 0

        # Weighted evaluation score
        evalScore = (
            -2 * foodDist  # Prioritize collecting food
            -3 * capsuleDist  # Incentivize capsules
            + ghostPenalty  # Penalize proximity to ghosts
            + capsuleIncentive  # Strong incentive for capsule consumption
        )

        return evalScore
"""


import random
from collections import Counter
from pacai.util import util
from pacai.core.directions import Directions


class OffensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.foodCarried = 0  # Tracks food collected by the agent
        self.visited = set()  # Tracks recently visited positions
        self.visitPenalty = 10  # Steps to avoid revisiting
    
    def manhattanDistance(self, pos1, pos2):
        """
        Compute the Manhattan distance between two positions.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def chooseAction(self, gameState):
        """
        Chooses an action using Q-Learning for high-level decision-making and A* for execution.
        Resolves oscillation by penalizing recently visited positions.
        """
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP

        # Avoid oscillation: Filter out actions leading to recently visited states
        myPos = gameState.getAgentState(self.index).getPosition()
        filteredActions = [
            a for a in actions if gameState.generateSuccessor(self.index, a).getAgentState(self.index).getPosition() not in self.visited
        ]
        if filteredActions:
            actions = filteredActions

        # Update visited positions to avoid loops
        self.visited.add(myPos)
        if len(self.visited) > self.visitPenalty:
            self.visited.pop()

        # High-level decision: Prioritize goals
        highLevelAction = self.decideHighLevelAction(gameState)

        # Execute the chosen action using A*
        return self.executeHighLevelAction(gameState, highLevelAction)

    def decideHighLevelAction(self, gameState):
        """
        Decide the high-level goal for the agent:
        - Return home if carrying too much food.
        - Eat capsules if ghosts are near.
        - Collect food otherwise.
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        foodCarriedThreshold = 5  # Example threshold for returning home
        ghostsNearbyThreshold = 5  # Distance to consider ghosts as "near"

        if self.foodCarried >= foodCarriedThreshold:
            return "returnHome"

        capsules = self.getCapsules(gameState)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        guards = [g for g in enemies if g.isGhost() and g.getPosition() is not None]
        if any(self.getMazeDistance(myPos, g.getPosition()) < ghostsNearbyThreshold for g in guards) and capsules:
            return "eatCapsule"

        return "collectFood"

    def executeHighLevelAction(self, gameState, highLevelAction):
        """
        Execute the high-level action using A* search.
        """
        myPos = gameState.getAgentState(self.index).getPosition()

        if highLevelAction == "collectFood":
            foodList = self.getFood(gameState).asList()
            if foodList:
                closestFood = min(foodList, key=lambda food: self.getMazeDistance(myPos, food))
                path = self.aStarSearch(gameState, myPos, closestFood, heuristic= self.manhattanDistance)
                return path[0] if path else random.choice(gameState.getLegalActions(self.index))

        elif highLevelAction == "returnHome":
            homeBoundary = self.getBoundaryPosition(gameState)
            path = self.aStarSearch(gameState, myPos, homeBoundary, heuristic= self.manhattanDistance )
            return path[0] if path else random.choice(gameState.getLegalActions(self.index))

        elif highLevelAction == "eatCapsule":
            capsules = self.getCapsules(gameState)
            if capsules:
                closestCapsule = min(capsules, key=lambda cap: self.getMazeDistance(myPos, cap))
                path = self.aStarSearch(gameState, myPos, closestCapsule, heuristic= self.manhattanDistance)
                return path[0] if path else random.choice(gameState.getLegalActions(self.index))

        # Default fallback to random action
        return random.choice(gameState.getLegalActions(self.index))
    
    from pacai.util import priorityQueue

    def aStarSearch(self, gameState, start, goal, heuristic):
        """
        A* Search implementation to compute the shortest path to the goal.
        """
        walls = gameState.getWalls()  # Access walls from the gameState parameter
        pq = priorityQueue.PriorityQueue()
        pq.push((start, [], 0), 0)  # (current position, actions, current cost)

        visited = set()

        while not pq.isEmpty():
            currentPosition, actions, currentCost = pq.pop()

            if currentPosition in visited:
                continue
            visited.add(currentPosition)

            # Check if we reached the goal
            if currentPosition == goal:
                return actions  # Return the list of actions

            # Expand neighbors
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(direction)
                nextPosition = (int(currentPosition[0] + dx), int(currentPosition[1] + dy))

                if not walls[nextPosition[0]][nextPosition[1]]:  # Check if the next position is valid
                    newCost = currentCost + 1
                    heuristicCost = heuristic(nextPosition, goal)
                    pq.push((nextPosition, actions + [direction], newCost), newCost + heuristicCost)

        return []  # Return empty list if no path is found




class DefensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Defensive agent decision-making using Q-Learning and A*.
        """
        # Identify invaders
        invaders = [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i).isPacman() and gameState.getAgentState(i).getPosition() is not None
        ]

        myPos = gameState.getAgentState(self.index).getPosition()

        if invaders:
            # Chase the closest invader
            closestInvader = min(invaders, key=lambda inv: self.getMazeDistance(myPos, inv.getPosition()))
            path = aStarSearch(gameState, myPos, closestInvader.getPosition(), self.index, manhattanHeuristic)
            return path[0] if path else Directions.STOP
        else:
            # Patrol critical food or boundary
            criticalFood = self.getCriticalFood(gameState)
            if criticalFood:
                path = aStarSearch(gameState, myPos, criticalFood, self.index, manhattanHeuristic)
                return path[0] if path else Directions.STOP

            # Patrol the boundary as a fallback
            patrolPos = self.getBoundaryPosition(gameState)
            path = aStarSearch(gameState, myPos, patrolPos, self.index, manhattanHeuristic)
            return path[0] if path else Directions.STOP


    def getCriticalFood(self, gameState):
        """
        Find the most critical food to defend (e.g., closest to the boundary).
        """
        foodDefending = self.getFoodYouAreDefending(gameState).asList()
        boundaryX = gameState.getWalls()._width // 2
        if not self.red:
            boundaryX -= 1

        if foodDefending:
            return min(foodDefending, key=lambda food: abs(food[0] - boundaryX))
        return None
