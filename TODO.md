A list of work that needs to be done on the `varcompfa` package.

# High Priority

- [ ] History callback that can record parameter values
- [ ] Check serialization via `json_tricks`
- [ ] Progress widget using Blessings
- [ ] Figure out a way of making the agent class more efficient
    + Currently it computes the feature vector too many times
    + Caching the result somehow would definitely help (but `joblib` doesn't do this properly/efficiently)
- [ ] State-dependent action-probabilities for control algorithms
- [ ] Binary vector tiling class
- [ ] A full example of the experiment pipeline
    + Define a learning agent, have it learn a policy, then freeze the policy
    + Use the frozen policy in a series of policy evaluation experiments
    + Performing full MC-rollouts from various start states sampled from the environment
    + Record and analyze the results
    + Make plots illustrating the results 
- [ ] Plotting functions
    + Need some boilerplate code that can produce graphs for publication
- [ ] Faster way of recording the results (`json_tricks` feels the strain, `json` doesn't quite work, `pickle` is not ideal either) 
    + It seems faster when you use compression, however
    + However, even with compression we tend to see very large files...
- [ ] More efficient way of recording experiments
- [ ] Command line interface
- [ ] Add more documentation once the code is more or less finalized
- [ ] HTML documentation (Sphix+Napoleon?)
- [ ] Add to the Makefile to automate generating the docs, updating version,...
- [ ] Set up scripts for running the experiments
    + This could be done as a Makefile, but may be better as a Python script
- [ ] Add the ability to add comments/tags to experiments
- [ ] Kuhn-triangulation / online representation refinement 
- [ ] Parallelization of the code (via Joblib? IPython Kernels? Celery?)
- [ ] Finish adding a `get_config()` method to all classes or just use json_tricks?
    + Agents
    + Features
    + Environments(?)
    + Experiments 
- [x] Automatic versioning
- [x] Global logging setup (see `openai/gym` for an example)
- [x] Figure out a way to tag experiments/add more metadata to them so that their purpose is clear weeks/months afterwards.

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

- [ ] Reorganize namespace to increase clarity
    + 'features' --> 'ft'? Worthwhile?
- [ ] Improve the code for feature generation so that the DAG structure is more coherent
    + Remove 'child' attributes, compose as functions instead
- [ ] Profile the code
- [ ] Add more test coverage to the overall code base
- [ ] Uniform hashing tile coding
- [ ] Better progress bar
- [ ] Live dashboard
- [ ] Remote monitor
- [ ] Implement tile coding and other discretization/feature functions in C.
- [ ] Determine if it would be worthwhile to implement 'cascading' algorithms that can run online in an experiment class
    + It is manifestly *desirable* past a certain point, but it would be require a different setup to record these sorts of experiments in a modular way (as we have with callbacks)
    + Perhaps via some sort of `LoggingAgent`?


## Web Dashboard

- Implement the frontend using Javascript, either by hand or with one of the dashboard libraries (OpenMCT or similar)
    + https://github.com/christabor/flask_jsondash
- Have the data POST to the dashboard using `requests`
