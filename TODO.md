A list of work that needs to be done on the `varcompfa` package.

# High Priority

- [ ] A full example of the experiment pipeline
    + Define a learning agent, have it learn a policy, then freeze the policy
    + Use the frozen policy in a series of policy evaluation experiments
    + Performing full MC-rollouts from various start states sampled from the environment
    + Record and analyze the results
    + Make plots illustrating the results 
- [ ] Plotting functions
    + Need some boilerplate code that can produce graphs for publication
- [ ] Command line interface
- [ ] Add more documentation once the code is more or less finalized
- [ ] HTML documentation (Sphix+Napoleon?)
- [ ] Automatic versioning
- [ ] Add to the Makefile to automate generating the docs, updating version,...
- [ ] Set up scripts for running the experiments
    + This could be done as a Makefile, but may be better as a Python script
- [ ] Figure out a way to tag experiments/add more metadata to them so that their purpose is clear weeks/months afterwards.


## Test Coverage

Some parts of the code need test coverage soon, because the project is now at the stage where

- [ ] `Agent`
- [ ] `Experiment`
- [ ] `Callback`
- [ ] Various analysis functions (e.g., for computing the return)


## Generic Markov Decision Process Framework

I have ideas on how to make a generic MDP framework so that we don't have to implement a separate class for every MDP we wish to implement.
This would be helpful because there are MDPs defined in the papers we wish to examine that differ slightly from each other, so there would be a lot of copy-pasting, which is non-ideal.
Once it works we will have one single class to test, and so if we do end up defining a separate `gym` environment for each, we can at least proceed with confidence in the backend that they're running on. 

# Low Priority

- [ ] Improve the code for feature generation so that the DAG structure is more coherent
- [ ] Profile the code
- [ ] Add more test coverage to the overall code base
