import random
import math
import heapq
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from pprint import pprint 

trials = 100

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # Sets self.env = env, state = None, next_waypoint = None, default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        self.state = {}

        # Initialize any additional variables here

        # Use our own state instead of self.state because of program crashing
        self.q = {}
        self.old_state = None
        self.old_reward = .0
        self.alpha = .7
        self.gamma = .0
        self.epsilon = .0
        self.random_count = 0
        self.steps_total = 1
        self.steps_trial = 0
        self.rewards = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # Prepare for a new trip; reset any variables here, if required

        if self.steps_total == 1:
            print 'STATS,Steps,Total,Random,Epsilon,Alpha,Gamma,Rewards' 
        else:
            print 'STATS,{:2},{:4},{:2},{:.2f},{:.2f},{:.2f},{:3}'.format(
                    self.steps_trial, self.steps_total, self.random_count,
                    self.epsilon, self.alpha, self.gamma, self.rewards)

        self.old_state = None
        self.old_reward = .0
        self.random_count = 0
        self.steps_trial = 0
        self.rewards = 0
        self.epsilon = 2 / (2 + math.sqrt(self.steps_total))
        self.alpha = 10 / (10 + math.sqrt(self.steps_total))

    def getAction(self, state):
        action = None
        max_v = None
        # Use optimistic e-gready strategy. For unseen actions, set Q value between
        # 2-10 (this problem only)
        max_v_default = 5
        
        if random.random() < self.epsilon:
            # Realistic e-gready.
            # Use suggested waypoint instead of completely random action.
            action = state[4][1]
            max_v = max_v_default
            self.random_count += 1
        else:
            for _action in self.env.valid_actions:
                key = (state, _action)

                if key in self.q:
                    v = self.q[key]

                    if max_v is None or v > max_v:
                        max_v = v
                        action = _action
                else:
                    # Explore new action in current state. Use optimistic e-gready
                    # strategy by setting max long-term reward to above 0
                    action = _action
                    max_v = max_v_default
                    break

        return (action, max_v)

    def update(self, t):
        # Gather inputs.
        # From route planner, also displayed by simulator.
        self.next_waypoint = self.planner.next_waypoint() 
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.steps_total += 1
        self.steps_trial += 1

        # Update state
        state = (('oncoming', inputs['oncoming']),
                ('left', inputs['left']),
                ('right', inputs['right']),
                ('light', inputs['light']),
                ('next_waypoint', self.next_waypoint))
        # Update state in GUI
        self.state = state

        # Select action according to your policy
        action, max_v = self.getAction(state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewards += reward

        # Learn policy based on state, action, reward
        if self.old_state is not None:
            # Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * Q(s',a'))
            old_v = self.q.get(self.old_state, .0)
            new_v = self.old_reward + self.gamma * max_v
            self.q[self.old_state] = (1 - self.alpha) * old_v + self.alpha * new_v

        # Save current state
        self.old_state = (state, action)
        self.old_reward = reward

        print "deadline = {}, inputs = {}, waypoint = {}, action = {}, "\
                "reward = {}".format(
           deadline, inputs, self.next_waypoint, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    # Now simulate it.
    # Create simulator (uses pygame when display=True, if available).
    sim = Simulator(e, update_delay=.01, display=False)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=trials)  # run for a specified number of trials

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the
    # command-line
    
    # Print the final Q table.
    print '+'*100
    pprint(a.q)
    print '+'*100

if __name__ == '__main__':
    run()
