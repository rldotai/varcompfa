A list of work that needs to be done on the `varcompfa` package.

# High Priority

- [ ] A full example of the experiment pipeline
    + Define a learning agent, have it learn a policy, then freeze the policy
    + Use the frozen policy in a series of policy evaluation experiments
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


# Low Priority

- [ ] Improve the code for feature generation so that the DAG structure is more coherent
- [ ] Profile the code
