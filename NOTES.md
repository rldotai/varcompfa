# Implementation notes and observations

## December 2, 2016

### Profiling Policy Evaluation Prototype

I currently implement the policy evaluation experiments with a design borrowed from the `Keras` library, which has a number of desirable aspects:

- It can support many different learning algorithms, each with their own parameters and feature representations, all running simultaneously.
- The experiment reports relevant information at the various phases of the experiment (e.g., at the start, during each episode, before/after each step...) using callbacks.
    + These callbacks can then be used to record the results of the experiment or display information about the experiment live. 
- The experiment manages a list of agents and updates them once per-step, and records the 'result' of the update (typically just the agent's TD-error)
- The agent encapsulates a learning algorithm, a feature function, and some parameters.
    + We do this via an `Agent` class, which wraps the learning algorithm, its feature function, and a dictionary of parameters.
    + The parameters can be functions of e.g., the current state, or they can be constant-valued if we don't need that sort of specificity.
    + The `Agent` class computes the relevant information and passes it to the learning algorithm as a dictionary (the algorithm's context) using the algo's `update()` method.
    + The learning algorithm then orders the parameters appropriately (made easier using a tiny bit of metaprogramming and passes them its `.learn()` method)


The only problem is that we sacrifice some performance for abstraction.

The current implementation does not assume that we use the same feature representation for every learning algorithm, or that each step is part of the same trajectory.

Thus, for each agent, we end up computing `x = phi(obs)` and `xp = phi(obs_p)` even when it is likely that on every step after the first, `x` could be set via the `xp` from the previous step.

Coupled with the fact that we also compute `x = phi(obs)` when selecting actions, we compute the feature function three times more than is necessary for an experiment where we train a single learning algorithm.
When training an algorithm for policy evaluation, we end up computing the features merely twice as often as necessary.

#### Should I switch designs?

No, I don't think so-- if we have a sufficient number of agents, then computing the features for action selection is not too punishing, and it's definitely possible that we might present the trajectories out of order (e.g., for training deep networks).

It's more that the hacked together experiment running code took advantage of a way of saving computation that might not always be justified.

#### Have you tried memoization?

Yes.

`joblib` promises that it will handle memoization for numpy arrays but after fiddling with the settings I found it to be either slower or dramatically slower as compared with just recomputing the features.

I then ventured to write my own memoization decorator, trying out various methods for custom-hashing of numpy arrays.
To get a baseline for the best case, we try:

- Using `dict` to contain the results
- Making a minimum of function calls
- Assuming that the input to memoize will always be a single array,...)
- Trying out the various hash functions in `hashlib` (MD5 is the best)
- Trying different methods of making the arrays hashable:
    + `np.ascontiguousarray(arr).view(np.uint8)`
    + `arr.tostring()`
    + `arr.tobytes()` (fastest or second fastest)
    + `pickle.dumps(arr)` (fastest or second fastest) 
- Variations on how the memoization stores previous results:
    + As a class with a dictionary
    + As a function where the dictionary is enclosed in the wrapper's scope


Tragically, it's still slower than just recomputing.
Making the arrays hashable kills us, as it incurs a lot of overhead from function calls.

We're in the uncomfortable territory where it's not worth it to memoize 
in general, but may be in particular cases.
For example, memoizing the output of converting a sequence of indices to a binary vector might be worthwhile, but memoizing something like tile-coding is probably a waste of time (since the input is continuous).

So, unfortunately, I can't just slap a memoization decorator everywhere and reap the speed boost.

#### What to do?

If it ends up being an issue I'm going to reimplement the feature functions in C, which may be a bit tedious because I haven't had to do that sort of thing in a while.

Furthermore, probabably going to investigate different methods for running large batch jobs in parallel.
