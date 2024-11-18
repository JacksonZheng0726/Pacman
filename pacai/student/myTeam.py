import random
from pacai.agents.capture.capture import CaptureAgent

def createTeam(firstIndex, secondIndex, isRed,
        first = 'DummyAgent',
        second = 'DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        DummyAgent(firstIndex),
        DummyAgent(secondIndex),
    ]

class DummyAgent(CaptureAgent):
    """
    A DummyAgent that moves randomly.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Picks an action randomly from the legal actions.
        """
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)