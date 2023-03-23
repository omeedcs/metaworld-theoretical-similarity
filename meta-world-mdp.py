    # S is a set of states.
    # A is a set of actions.
    # P is the state transition probability function: P(s'|s, a) represents the probability of transitioning to state s' from state s when taking action a.
    # R is the reward function: R(s, a, s') represents the immediate reward received after transitioning from state s to state s' due to action a.
    # γ is the discount factor, with a value between 0 and 1, which determines the importance of future rewards.

class MDPGraphNode:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class MDPGraph:
    def __init__(self):
        self.graph = {}
        self.states = []
        self.actions = []
        self.transitions = []
        self.rewards = []

    def add_node(self, node):
        if node.state not in self.states:
            self.states.append(node.state)
        if node.action not in self.actions:
            self.actions.append(node.action)
        self.transitions.append((node.state, node.action, node.next_state))
        self.rewards.append((node.state, node.action, node.reward))
        if node.state not in self.graph:
            self.graph[node.state] = {}
        if node.action not in self.graph[node.state]:
            self.graph[node.state][node.action] = []
        self.graph[node.state][node.action].append(node.next_state)

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_transitions(self):
        return self.transitions

    def get_rewards(self):
        return self.rewards

    def get_graph(self):
        return self.graph

    def get_next_states(self, state, action):
        return self.graph[state][action]

    def get_reward(self, state, action):
        return self.rewards[state][action]

    def get_transition_probability(self, state, action, next_state):
        return self.transitions[state][action][next_state]