# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        # first get list of states
        states = self.mdp.getStates()

        # update value "iterations" times
        for x in range(self.iterations):

          # make new Counter of next iteration values
          valuesNext = self.values.copy()

          # update value in all states
          for state in states:

            # terminal states have value 0
            if self.mdp.isTerminal(state):
              #self.values[state] = 0
              continue

            # get list of possible actions
            actions = self.mdp.getPossibleActions(state)

            # if no more actions and not terminal, value is 0
            if not actions:
              #valuesNext[state] = 0
              continue

            # new V is highest Q value
            highest = self.computeQValueFromValues(state, actions[0])
            for action in actions[1:]:
              currQ = self.computeQValueFromValues(state, action)
              if currQ > highest:
                highest = currQ
            valuesNext[state] = highest

            """
            #update Q values
            action = self.computeActionFromValues(state)
            if action is None:
              valuesNext[state] = 0
            else:
              qValue = self.computeQValueFromValues(state, action)
              valuesNext[state] = qValue
            """

          # update values
          self.values = valuesNext
    

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # getTransitionStatesAndProbs returns list of (nextState, prob) pairs
        qValue = 0
        transFxns = self.mdp.getTransitionStatesAndProbs(state, action)
        # x[1] is probability T(s,a,s')
        # x[0] is state s'
        for x in transFxns:
          # T(s,a,s')[R(s,a,s') + γV*(s')]
          qValue += x[1] * (self.mdp.getReward(state, action, x[0]) + (self.discount * self.values[x[0]]))
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # basically argmax Q*(s,a)

        # first get list of legal actions
        legalActions = self.mdp.getPossibleActions(state)

        # if legalActions is empty return None
        if not legalActions:
          return None

        # else return action with highest Q value

        # loop to find legal action with highest Q value
        highest = self.computeQValueFromValues(state, legalActions[0])
        highestAction = legalActions[0]

        for action in legalActions[1:]:
          currQValue = self.computeQValueFromValues(state, action)
          if currQValue > highest:
            highest = currQValue
            highestAction = action

        # return the legal action with highest Q value
        return highestAction
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # first get list of states
        states = self.mdp.getStates()

        # get number of states
        numStates = len(states)

        # update value "iterations" times
        for x in range(self.iterations):

          # work with value xth from first
          state = states[x % numStates]

          # terminal states have value 0
          if self.mdp.isTerminal(state):
            #self.values[state] = 0
            continue

          # get list of possible actions
          actions = self.mdp.getPossibleActions(state)

          # if no more actions and not terminal, value is 0
          if not actions:
            #valuesNext[state] = 0
            continue

          # new V is highest Q value
          highest = self.computeQValueFromValues(state, actions[0])
          for action in actions[1:]:
            currQ = self.computeQValueFromValues(state, action)
            if currQ > highest:
              highest = currQ
          self.values[state] = highest


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # first get list of states
        states = self.mdp.getStates()

        # a dictionary stores list of predecessors for each state
        predecessors = {}
        
        # get predecessors of all states
        for state in states:
          # create set of predecessors for the state
          preds = set()
          for s in states:
            possActions = self.mdp.getPossibleActions(s)
            successors1 = []
            successors = []
            for action in possActions:
              successors1 += self.mdp.getTransitionStatesAndProbs(s, action)
            successors = [trans[0] for trans in successors1]
            if state in successors:
              preds.add(s)

          # store list of predecessors in the dictionary
          predecessors[state] = preds
        

        # initialize empty priority queue
        statesQueue = util.PriorityQueue()

        # for all non-terminal states
        for s in states:

          # do nothing if terminal state
          if self.mdp.isTerminal(s):
            continue
          
          # find highest Q value of possible actions from s
          actions = self.mdp.getPossibleActions(s)
          highest = self.computeQValueFromValues(s, actions[0])
          for action in actions[1:]:
            currQ = self.computeQValueFromValues(s, action)
            if currQ > highest:
              highest = currQ
          
          # get abs value of diff between highest Q and value of S
          diff = abs(self.values[s] - highest)

          negDiff = diff * -1

          # push s into priority queue
          statesQueue.update(s, negDiff)

        # for iteration in 0,1,2,...,self.iterations-1
        for x in range(self.iterations):

          # terminate if priority queue is empty
          if statesQueue.isEmpty():
            break

          # pop a state s off priority queue
          s = statesQueue.pop()

          # find highest Q value of possible actions from s
          actions = self.mdp.getPossibleActions(s)
          highest = self.computeQValueFromValues(s, actions[0])
          for action in actions[1:]:
            currQ = self.computeQValueFromValues(s, action)
            if currQ > highest:
              highest = currQ

          # update value of s if not terminal
          if not self.mdp.isTerminal(s):
            self.values[s] = highest

          # for each predecessor p of s
          for p in predecessors[s]:

            # find highest Q value of possible actions from p
            actions = self.mdp.getPossibleActions(p)
            highest = self.computeQValueFromValues(p, actions[0])
            for action in actions[1:]:
              currQ = self.computeQValueFromValues(p, action)
              if currQ > highest:
                highest = currQ
            
            # get abs value of diff between highest Q and value of S
            diff = abs(self.values[p] - highest)
            negDiff = diff * -1
            # if diff > theta, push p onto statesQueue with priority -diff
            if diff > self.theta:
              statesQueue.update(p, negDiff)