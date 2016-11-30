A list of work that needs to be done on the `varcompfa` package.

# High Priority

- [ ] Binary vector tiling class
- [ ] A full example of the experiment pipeline
    + Define a learning agent, have it learn a policy, then freeze the policy
    + Use the frozen policy in a series of policy evaluation experiments
    + Performing full MC-rollouts from various start states sampled from the environment
    + Record and analyze the results
    + Make plots illustrating the results 
- [ ] Plotting functions
    + Need some boilerplate code that can produce graphs for publication
- [ ] Faster way of recording the results (`json_tricks` feels the strain) 
    + It seems faster when you use compression, however
    + However, even with compression we tend to see very large files...
- [ ] More efficient way of recording experiments
- [ ] Command line interface
- [ ] Add more documentation once the code is more or less finalized
- [x] Automatic versioning
- [ ] HTML documentation (Sphix+Napoleon?)
- [ ] Add to the Makefile to automate generating the docs, updating version,...
- [ ] Set up scripts for running the experiments
    + This could be done as a Makefile, but may be better as a Python script
- [ ] Figure out a way to tag experiments/add more metadata to them so that their purpose is clear weeks/months afterwards.
    + Currently using git-hash + versioning + timestamp
    + Consider adding the ability to add comments/tags to experiments
- [ ] Determine if it would be worthwhile to implement 'cascading' algorithms that can run online in an experiment class
    + It is manifestly *desirable* past a certain point, but it would be require a different setup to record these sorts of experiments in a modular way (as we have with callbacks)
    + Perhaps via some sort of `LoggingAgent`?
- [ ] Kuhn-triangulation / online representation refinement 


## Test Coverage

Some parts of the code need test coverage soon, because the project is now at the stage where

- [ ] `Agent`
- [ ] `Experiment`
    + Partially completed, need to wait for rest of API to settle down
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
- [ ] Uniform hashing tile coding
