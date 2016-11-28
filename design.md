Some notes on the architecture of the code


# Experiment

An experiment needs to have the following things

- environment
- agent(s)
    + Learning agents that get updated during the experiment
    + Needs to have:
        * learning algorithm
        * feature function
        * parameters (which may depend on current state, timestep, etc.)
- controller
    + The agent/policy that selects actions based on observations
    + Likely can be set up the same way as the learning agents
    + May have to 

- We can specify the following when running an experiment:
    * number of episodes
    * maximum number of steps
    * save-directory
        + base directory

I am inclined towards the Keras model, where the experiment can be run in a largely customizable fashion based on the initial setup and callback functions which trigger during each phase of the experiment.

# Agents

Learning agents are a useful abstraction to keep multiple variations of a common algorithm distinct.

- encapsulates the learning algorithm, function approx, and parameters
- parameters are functions
- param_funcs:
    - map context to parameters
    - key = name of parameter, value = function

# Policy

The policy can either be fixed in advance or can learn online. 

- map observations to actions
    - probably entails having a feature function
    - This likely means it should be an agent of some sort.
- should be able to compute probability of a particular action given an observation
    + This complicates things somewhat, but I can see this working via an appropriate `agent` implementation even when the `algo` inside the agent isn't quite set up to return action probabilities.
