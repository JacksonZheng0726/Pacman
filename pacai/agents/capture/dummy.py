import random

from pacai.agents.capture.capture import CaptureAgent

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
    
    def final(self, gamestate):
        # Inhert the final method defined in th CaptureAgent to reset the observation history of the game
        super().final(gamestate)

        # retrieve the final score
        Final_score = gamestate.getScore()

        # Figure out if the pacman agent won or lost the game
        if gamestate.isWin():
            print(f"Pacman won, the score was {Final_score}")
        elif gamestate.isLose():
            print(f"Pacman lost, the score was {Final_score}")
        else:
           print(f"it was a draw game, the score was {Final_score}")

    def getAction(self, gameState):
        super().final(gameState)

    def getCapsules(self, gameState):
        super().getCapsules(gameState)

    def getCapsulesYouAreDefending(self, gameState):
        super().getCapsules(gameState)

    def getCurrentObservation(self): 
        super().getCurrentObservation()
    
    def getFood(self, gameState):
        super().getFood(gameState)
    
    def getFoodYouAreDefending(self, gameState):
        """
        Returns the food you're meant to protect (i.e., that your opponent is supposed to eat).
        This is in the form of a `pacai.core.grid.Grid`
        where `m[x][y] = True` if there is food at (x, y) that your opponent can eat.
        """

        if (self.red):
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()
    
    def getOpponents(self, gameState):
        """
        Returns agent indices of your opponents. This is the list of the numbers
        of the agents (e.g., red might be 1, 3, 5)
        """

        if self.red:
            return gameState.getBlueTeamIndices()
        else:
            return gameState.getRedTeamIndices()
    
    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points using the builtin distancer.
        """

        return self.distancer.getDistance(pos1, pos2)
    
    def getPreviousObservation(self):
        """
        Returns the `pacai.core.gamestate.AbstractGameState` object corresponding to
        the last state this agent saw.
        That is the observed state of the game last time this agent moved,
        this may not include all of your opponent's agent locations exactly.
        """

        if (len(self.observationHistory) <= 1):
            return None

        return self.observationHistory[-2]
    
    def getScore(self, gameState):
        """
        Returns how much you are beating the other team by in the form of a number
        that is the difference between your score and the opponents score.
        This number is negative if you're losing.
        """

        if (self.red):
            return gameState.getScore()
        else:
            return gameState.getScore() * -1

    def getTeam(self, gameState):
        """
        Returns agent indices of your team. This is the list of the numbers
        of the agents (e.g., red might be the list of 1,3,5)
        """

        if (self.red):
            return gameState.getRedTeamIndices()
        else:
            return gameState.getBlueTeamIndices()
    
    def registerTeam(self, agentsOnTeam):
        """
        Fills the self.agentsOnTeam field with a list of the
        indices of the agents on your team.
        """

        self.agentsOnTeam = agentsOnTeam
    
    def observationFunction (self, state):
        super().getCurrentObservation()
    
    def loadAgent(name, index, args={}):
        agent = CaptureAgent.BaseAgent.loadAgent('pacai.agents.capture.dummy', index=1, args={})
        
        return agent