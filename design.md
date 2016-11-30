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
