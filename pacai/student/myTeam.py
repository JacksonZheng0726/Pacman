from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
from pacai.core.actions import Actions
import random
from pacai.util import priorityQueue


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
def aStarSearch(gameState, start, goal, agentIndex, heuristic):
    """
    A* Search to Figure out the shortest path to the goal.
    """
    # It is a 2 dimensional array that represent the grid in the game
    walls = gameState.getWalls()
    # Create a priority queue to store the node during the a start search
    pq = priorityQueue.PriorityQueue()
    # Push the initial state of the agent into the queue, start = current position
    # actions of the agent during the search stored in the list and the cost
    pq.push((start, [], 0), 0)
    # create a set to store the visited node to avoid revisit the same node again
    visited = set()
    
    # As the priority queue was not empty, the loop won't terminate
    while not pq.isEmpty():
        # we extract the position, actions and the cost of the agent
        current_pos, actions, currentCost = pq.pop()
        # check if the current position visited already, continue executed, which
        # go back to the beginning of the loop
        if current_pos in visited:
            continue
        # Otherwise, add the position of the agent into the set that use to keep track
        # the visited node
        visited.add(current_pos)

        # if we reached the goal
        if current_pos == goal:
            # Return the list of actions
            return actions

        # loop through each direction
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # convert the direction to the appropriate coordinate vector like(0,1)
            dx, dy = Actions.directionToVector(direction)
            # adding the current position vector to the direction vector to get the 
            # coordinates of the neaby position
            next_pos = (int(current_pos[0] + dx), int(current_pos[1] + dy))

            # check if the agent's next position is a wall or not
            if not walls[next_pos[0]][next_pos[1]]:
                # if not, we increment the cost by 1
                new_cost = currentCost + 1
                # compute the estimated cost to the goal from the next position
                heuristic_cost = heuristic(next_pos, goal)
                # Pushes the neighbors into the priority queue with the newly updated position,
                # actions and costs
                pq.push((next_pos, actions + [direction], new_cost), new_cost + heuristic_cost)
    # if no path found, return a empty list
    return []
 
def manhattanHeuristic(position, goal):
    """
    Manhattan distance heuristic for A* Search.
    """
    """
    compute the horizontal and vertical difference between the current position
    and the goal and all them up to returned as the huristic value
    """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


"""
Implemented an Q-learning agent that allow the agent to make decision by learning the rewards in the 
environment. In the beginning, I initialized the learning rate, discount factor, exploration rate and a 
dictionary with the (state, action) as the key and the value would be the q-learning values. Within the 
Q learning agent class, I define a getQValue function, that return the Q-learning values with the 
corresponding key(state, action) pair, if no key found, then return 0.0 as q learning values. Then I create a 
function that compute the best action for the current state based on the Q values. I also created a boundary function
that return the closest accessible position on the boundary of the agent's territory. Also, I have generated an update
function that use to update the q-learning value of the agent by utilizing the q-learning formula: Q(s,a)=(1−α)⋅Q(s,a)+
α⋅(r+γ⋅ max​Q(s ′,a)). Also, I have a getStatefeature function that return the relevant information of the current 
gamestate, such as the amount of the food that the agent's carrying, the distance of the agent to the cloest food,
number of the capulse left and true or false that state whehter the agent still in the home territory

"""
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
        return self.qValues.get((state, action), 0.0)
    
    def best_action_Qval(self, gameState):
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
        if bestActions:
            return random.choice(bestActions) 
        else:
            return None


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
        position_list = []
        myPos = gameState.getAgentState(self.index).getPosition()
        for position in boundaryPositions:
            position_list.append(self.getMazeDistance(myPos, position))
        return min(position_list)


    def update(self, state, action, nextState, reward):
        """
        Update Q-values using the Q-learning update rule.
        """
        if not isinstance(state, dict):
            state = self.getStateFeatures(state)  # Extract features if raw state passed

        if not isinstance(nextState, dict):
            nextState = self.getStateFeatures(nextState)  # Extract features if raw state passed

        hashableState = tuple(state.items())

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
        food_list = []
        if foodList:
            for food in foodList:
                food_list.append(self.getMazeDistance(myPos, food))
            foodDist = min(food_list)
        else:
            foodDist = 0
        capsules = self.getCapsules(gameState)

        return {
            'foodCarried': self.foodCarried,
            'foodDist': foodDist,
            'capsules': len(capsules),
            'isHome': self.isInHomeTerritory(myPos, gameState),
        }

"""
I defined an offensiveAgent that inherit from the Q_learningAgent class. I initialized the number of the food carried
by the agent to be 0, initialized a empty set that use to keep track the visited position and set the visit penalty
to 10 in order t0 avoid revisit the same position again. Then I created an advance_action function that will force the
agent to keep eating the food and check if the agent carry too much food, then go back to the home territory, and check 
if the ghosts are near and capulses are available, prioritize eating the capulse first, so the agent will be safe since 
the ghost will be the scared mode after the capulse eaten by the agent. In addition, I generated another advance_action
function with the a star search. It first generate a foodlist that list out all the available in the current grid, then
loop through each food position in the list and find the cloest food position by utilizing the getMazeDistance function.
Then I use the a star search to find a path that enable the agent to reach reach the cloest food position in a fastest way.
if no path found, return a randomly choice of the legal actions. Furthermore, I usd the a star search to compute the cloest path 
from the current position to the boundary position. If no path found, it will return a randomly choice of the legal actions. 
similar to the food search, I get a list of capulses available in the grid and loop through each capulse position and compute 
the distance by using the getMazeDistance function to find the capulse that was closest to the agent. Then I use the a star
search to find the cloest path to reach the cloest capluse from the agent. If the path not found, it will return a random
legal actions. If none of the capulse, food or boundary advanced action executed, the program will select random legal actions
from the state 
"""
class OffensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.foodCarried = 0  
        self.visited = set()  
        self.visitPenalty = 10 
    
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

        # avoid the actions that would lead the agent to visit the game state that recently
        # visited
        myPos = gameState.getAgentState(self.index).getPosition()
        Action_filter = []
        for a in actions:
            successor = gameState.generateSuccessor(self.index, a)
            if successor.getAgentState(self.index).getPosition() not in self.visited:
                Action_filter.append(a)
        if Action_filter:
            actions = Action_filter

        # Update visited positions to avoid loops
        self.visited.add(myPos)
        if len(self.visited) > self.visitPenalty:
            self.visited.pop()

        # apply the customize strategy to choose the efficient actions
        # for the agent
        advanced_action = self.advanced_action(gameState)

        # apply the a star search with the choosen action
        return self.advanced_action_Star(gameState, advanced_action)

    def advanced_action(self, gameState):
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
        guards = []
        for g in enemies:
            if g.isGhost() and g.getPosition() is not None:
                guards.append(g)
        for g in guards:
            if(self.getMazeDistance(myPos, g.getPosition()) < ghostsNearbyThreshold and capsules):
                return "eatCapsule"

        return "collectFood"

    def advanced_action_Star(self, gameState, highLevelAction):
        """
        Execute the high-level action using A* search.
        """
        myPos = gameState.getAgentState(self.index).getPosition()

        if highLevelAction == "collectFood":
            foodList = self.getFood(gameState).asList()
            if foodList:
                closestFood = min(foodList, key=lambda food: self.getMazeDistance(myPos, food))
                path = self.aStarSearch(gameState, myPos, closestFood, heuristic= self.manhattanDistance)
                if path:
                    return path[0]
                else:
                    return random.choice(gameState.getLegalActions(self.index))

        elif highLevelAction == "returnHome":
            homeBoundary = self.getBoundaryPosition(gameState)
            path = self.aStarSearch(gameState, myPos, homeBoundary, heuristic= self.manhattanDistance )
            if path:
                return path[0]
            else:
                return random.choice(gameState.getLegalActions(self.index))

        elif highLevelAction == "eatCapsule":
            capsules = self.getCapsules(gameState)
            if capsules:
                closestCapsule = min(capsules, key=lambda cap: self.getMazeDistance(myPos, cap))
                path = self.aStarSearch(gameState, myPos, closestCapsule, heuristic= self.manhattanDistance)
                if path:
                    return path[0]
                else:
                    return random.choice(gameState.getLegalActions(self.index))

        # Default fallback to random action
        return random.choice(gameState.getLegalActions(self.index))

    def aStarSearch(self, gameState, start, goal, heuristic):
        """
        A* Search implementation to compute the shortest path to the goal.
        """
        walls = gameState.getWalls()  # Ensure this works in your context
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



"""
I defined a defensive agent class that was inherit from the q learning agent class, it mainly design to prevent
the opponent to score by priortizing defending the food around th boundary first. Inside the defensive agent class,
I have a getcriticalfood function that prioritize defending the food that are close to the boundary. First, I checked
the list of the foods on my side of the map that can eat by the opponent by using the self.getFoodYouAreDefending
function. Then I used the getWall function to figure out the layout of the pacman game. Then I checked, if the food
exist that was closed to the boundary, then I calculated the horizontal distance between the food and the agent. 
Otherwise, it will return None if the food not exist.
I have a chooseAction function that check if the opponent was invading our side by utilizing the getAgentState 
and the isPacman functions to check. Then retrieve the position of our current agent and check if there're invader, 
then using the getMazeDistance function to compute the distance between the agent with the invader and use the min
function to extract the shortest distance. Then I applied the a star search to find the shortest path from the agent
to the invader. And check if the path was not found, then will return Direction.stop, the agent will stop until see 
the invader moving. If no invader was present, the agent will just find the criical_food to defend. I check if the 
critical food exist, then using the a star search that I defined previously to find the shorest path to reach the 
critical food. If no path exist, then return Direction.stop. If the previous conditions are not executed, then I will
retrieve the boundary position of the current gamestate, then using the a star search to find the shortest path to 
reach the boundary, if no path found, then the agent will stop. Otherwise, the agent will just wandering around the 
boundary position to defend the invader. 
"""
class DefensiveAgent(QLearningCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        # Check if the opponent agent is invading our territory or not
        invaders = []
        for i in self.getOpponents(gameState):
            opponentState = gameState.getAgentState(i)
            if opponentState.isPacman() and opponentState.getPosition() is not None:
                invaders.append(opponentState )


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
                if path:
                    return path[0]
                else:
                    return Directions.STOP

            # Patrol the boundary as a fallback
            patrolPos = self.getBoundaryPosition(gameState)
            path = aStarSearch(gameState, myPos, patrolPos, self.index, manhattanHeuristic)
            if path:
                return path[0]
            else:
                return Directions.STOP

    def getCriticalFood(self, gameState):
        """
        Look for the food around the boundary to defend first
        """
        foodDefending = self.getFoodYouAreDefending(gameState).asList()
        boundaryX = gameState.getWalls()._width // 2
        if not self.red:
            boundaryX -= 1

        if foodDefending:
            return min(foodDefending, key=lambda food: abs(food[0] - boundaryX))
        return None
