from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        # You can initialize Q-values here.
        self.Q_values = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        # look for the Q learning values that store with the key pair(state, action)
        if (state, action) in self.Q_values:
            return self.Q_values[(state, action)]
        else:
            return 0.0
         

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        # Get all the legal actions available in the state
        legal_action = self.getLegalActions(state)
        # If there is no legal action, return 0
        if not legal_action:
            return 0.0
        # initialize a empty list to store the Q learning val
        Value_Storage = []
        # retrieve all the q learning values and add to the list
        for action in legal_action:
            Value_Storage.append(self.getQValue(state, action))
        
        # return the largest q learning value in the list
        return max(Value_Storage)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        # Get all the legal actions available in the state
        legal_action = self.getLegalActions(state)
        # If there is no legal action, return 0
        if not legal_action:
            return None
        
        # retrieve the value of the best action
        best_val = self.getValue(state)
        best_action = []
        # iterate through each action in the legal actions
        for action in legal_action:
            # compare the value of the state, action pair with the best value
            if (self.getQValue(state, action) == best_val):
                # add all the best action into the list
                best_action.append(action)
        # break ties randomly for better action
        return random.choice(best_action)
    
    def update(self, state, action, nextState, reward):
        # return the current Q values of the state and action
        Q_val_current = self.getQValue(state, action)

        # compute the expected reward of agents conduct the action in the current gamestate
        # reward is receive from current state to the next state
        # discountRate determines how much the agents values the future reward or immediate reward
        # self.getValue(nextState) determine the max future reward for the agent
        expected_reward = reward + self.discountRate() * self.getValue(nextState)

        # the expected_reward - Q_val_current determines how far the expected_q values from the current 
        # Q values
        # self.Alpha is the learning adjust according to the differnce, rate between 0 and 1
        self.Q_values[(state, action)] = Q_val_current + self.Alpha() * (expected_reward - Q_val_current)
    
    def getAction(self, state):
        return random.choice(state.getLegalActions(self.index))

        
        
class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            raise NotImplementedError()
