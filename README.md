# metaworld-theoretical-similarity

### Categories of Similarity Metrics:

1. **************************************Model Based Metrics:************************************** based off structural similarities between MDP models. 
2. ****************************Performance Based Metrics:**************************** based off performance of the learning agents in source task and the target task. Achieved in two ways:
    1. Compare the resulting policies from learning in the source task and target task.
    2. Measuring the transfer gain (if it was positive or negative, and by how much). 
    3. Sometimes we can measure the similarity **************during************** the target task.

### Model-Based Metrics

1. **Transition and Reward Dynamics:** these require complete knowledge of the MDP models (source task and target task).
    1. State Abstraction: common practice is to aggregate states in order to obtain an abstract description of the problem. If a number of states are considered to be similar, they can be aggregated as a singular entity. Two common methods surveyed in the paper are *************bisimulation and homomorphism.*************
    2. Compliance: the probability of a sample <s, a, s’, r> in the target task being generated in the source task. ****hmm, why do we look at it backwards here??****
    3. MDP Graphs: construction of graphs that represent the transition and reward functions of both the source task and the target task. Once the graph is built, we can attempt to find structural similarities between tasks based on graphical similarity or graph matching algorithms. 
2. ************************Transitions************************
    1. Metrics used in this category use tuples in the form <s, a, s’> to measure the similarity between MDPs. The tuples model “behavioral dynamics” of the two MDPs to be compared, and attempt to find differences between them.
3. ****************Rewards****************
    1. Metrics used in this category use tuples in the form <s, a, r> to measure the similarity between MDPs. 
4. ************************************State and Actions:************************************ 
    1. Metrics used in this category use pairs in the form <s, a> for both the source task and target task to measure similarity between MDPs.
5. ****States:**** 
    1. Metrics used in this category use solely the state space both in the source and target task to measure the similarity between the MDPs. 
    2. Case Base Reasoning (CBR): another approach that uses a similarity function between states in the target task and the states stored in a case base corresponding to a previous source task.
    

### Performance Based Metrics

1. ******************************Transfer Gain:****************************** we measure the advantage gained by using the knowledge in one source task to speed the learning of another target task. The transfer experiment must be entirely or partially run before measuring the degree of similarity between tasks. We can look at total reward, asymptotic performance, jumpstart, etc. The higher the transfer gain, the greater the similarity between tasks.
    1. Offline Transfer Gain: the transfer gain is estimated as the difference in performance between the learning process with and without transfer. Computed once the learning processes are considered finished. 
    2. Online Transfer Gain: the transfer gain is estimated online, meaning at the same time that the policy in the target task is computed. This means we can decide online which is the closest task within a library of composed past tasks, so that the knowledge of the selected closest task can have a greater influence on learning about the policy in the new task.
2. ************************************Policy Similarity:************************************ these policies are based on the use of the learned value function V* or the action value function Q* (aka the behavioral policies π obtained in the source and target tasks. To compute these metrics, we require a full or partial learning of a policy before trying to find similarity. V* and Q* are value functions.
    1. Policy Values: observe the specific values of the Q* and V* functions correspondent to the source and target tasks, and see what degree of similarity they have.
    2. Policy Parameters: in reinforcement learning, V* and Q* is represented as parameter vector theta. In this metric, we compare the particular weights of the parameter vectors corresponding to the value functions of the source and target tasks to measure the similarity between them. This works with parametric representations of a policy, but not tabular representations.
        1. ************************Parametric:************************ *A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model*
        2. ********************Tabular:******************** refer to problems in which the state and actions spaces are small enough for approximate value functions to be represented as arrays and tables.
