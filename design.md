Some notes on the architecture of the code

# Summary

This codebase was developed for my thesis project, which explores different ways of estimating variance in reinforcement learning.

As such, there are a variety of algorithms to try on many different tasks, and there are complicating factors such as choosing the state representation (features) and evaluating general value functions, which means that things like the discount parameter may be different depending on the state.

Getting heterogenous algorithms to all work together regardless of environment, representation, or learning parameters is a difficult task, especially if you want the results to be reproducible.

To do this, we get pretty abstract.
We wrap the algorithms in "agents", which convert the raw state information into features and evaluate the state-dependent parameters as needed.

The information necessary for each agent is passed around as "contexts", which contain the information necessary for the state-dependent parameters to produce the correct values, which in turn is used in the context passed to the algorithms when updating.

The `Experiment` classes feed the raw context `(s, a, r, sp)` to the agents, which then get the needed information for the actual learning algorithms.
It has lead to a reasonably complicated code base, but it means that defining and running experiments is done reproducibly via simple configuration scripts.

There's probably a better architecture that I could've chosen, or perhaps a better programming model, but at this point I think I'll save rewriting everything for my next thesis.

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
    + May have to be able to specify action probabilities
    + Currently, we can perform learning within an experiment by passing the same control-capable agent as both the controller and one of the agents to be updated, but this is probably not ideal.

- We can specify the following when running an experiment:
    * number of episodes
    * maximum number of steps
    * save-directory
        + base directory

- The following things can be computed at runtime:
    + Code version (git hash, or the project's version number?)

I am inclined towards the Keras model, where the experiment can be run in a largely customizable fashion based on the initial setup and callback functions which trigger during each phase of the experiment.

I am further persuaded of making everything serializable in some sense, so the experiment and things that it requires (agents, control, environment, etc.) can be preserved/we know exactly what was run.
This may be excessive, however-- preserving the full state of everything may require too much space/computation, and may entail restrictive/excessive coding on the backend for everything we want to serialize.
I think therefore it might be better to start with a minimal amount of information and only store more if we find it necessary.

This should not pose a problem if all the necessary data can be re-computed.

## Experiment Data

There are some things that we can hope/expect to have access to, in general, for any experiment we wish to run.
There are other things that we will only have access to at particular times.
I am listing them below to help clarify what we expect to be able to compute or receive from the experiment-running code.

- Generally available
    - environment used for experiment
    - control/policy
    - learning agents
- Experiment start
    - current version
    - date/time when experiment was ran
    - type of experiment?
- Episode start
    + episode number
- Episode end
    + episode number
- Start-of-step:
    + step number
- End-of-step
    + step number
    + context
        * obs
        * obs_p
        * action
        * reward
        * done
    + update_info -- a list
        * agent's context
        * update result

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


# Strange Things / Edge Cases

## Episode Termination

- Normally, the episodic case can be unified with the continuous case via making some mild assumptions: that γ is zero or the feature vector is the zero vector, and that the reward is zero.
- In the OpenAI gym, we can't really get that same level of niceness when we wish to use state-dependent parameters.
    - Environments don't typically have a terminal state or examples of the terminal state. 
    - Furthermore, the next state following termination is usually just the initial state. 
- So it's not really possible to implement a well-defined terminal state of our own in general (although I am considering it because it's not really possible to learn from data when some information (like the episode abruptly "terminating") is not available.
- As for stop-gap measures, I've implemented a 'terminal step' in the experiment design class that ensures that when running out of time the context reflects that.
    - I've also added a `start_episode` method to most algorithms, that should zero out traces in case state-dependent parameters are not used. It gets called at the start of each episode in `LiveExperiment`.

# Data Analysis

- Data should probably be analyzed using `numpy`, `pandas` and `scipy`, with plotting done using either `matplotlib` or one of its wrappers (e.g. `seaborn`)
- Serialization probably will be handled by pickle, as the alternative is to write a custom serialization routine or use an existing library that doesn't quite fit our needs
    + Perhaps hdf5 might be better, because we can save metadata to it... 


