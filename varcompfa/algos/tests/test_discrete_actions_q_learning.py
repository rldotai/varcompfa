import numpy as np
import varcompfa as vcf


class TestDiscreteQ:
    def test_setup(self):
        for ns in range(1, 10):
            for na in range(2, 5):
                algo = vcf.algos.DiscreteQ(ns, na)
                assert(isinstance(algo, vcf.algos.LearningAlgorithm))

    def test_learn(self):
        na = 3
        for ns in range(1, 10):
            algo = vcf.algos.DiscreteQ(ns, na)

            for a in range(na):
                # Create fake but correctly typed data for testing the update
                x       = np.random.random(ns)  + 0.1
                r       = np.random.random()
                xp      = np.random.random(ns)
                alpha   = np.random.random()    + 0.1
                gm      = np.random.random()    + 0.1
                gm_p    = np.random.random()    + 0.1
                lm      = np.random.random()    + 0.1

                # Get the parameter values prior to the update
                w_0 = np.copy(algo.w)
                z_0 = np.copy(algo.z)

                # Perform the update
                algo.learn(x, a, r, xp, alpha, gm, gm_p, lm)

            # Check that something happened
            assert(np.any(algo.w != w_0))
            assert(np.any(algo.z != z_0))


    def test_update(self):
        """Test that updating the algorithm works (or at least doesn't fail)."""
        na = 4
        for ns in range(1, 10):
            algo = vcf.algos.DiscreteQ(ns, na)
            for a in range(na):
                params = dict(
                    x       = np.random.random(ns)  + 0.1,
                    a       = a,
                    r       = np.random.random(),
                    xp      = np.random.random(ns),
                    alpha   = np.random.random()    + 0.1,
                    gm      = np.random.random()    + 0.1,
                    gm_p    = np.random.random()    + 0.1,
                    lm      = np.random.random()    + 0.1,
                )

            # Get the parameter values prior to the update
            w_0 = np.copy(algo.w)
            z_0 = np.copy(algo.z)

            # Perform the update
            algo.update(params)

            # Check that something happened
            assert(np.any(algo.w != w_0))
            assert(np.any(algo.z != z_0))

    def test_get_params(self):
        """Test that `get_params()` returns the correct information."""
        ns = 10
        na = 7
        algo = vcf.algos.DiscreteQ(ns, na)
        params = algo.get_config()
        assert(isinstance(params, dict))
        assert(isinstance(params['num_features'], int))
        assert(isinstance(params['num_actions'], int))
        assert(isinstance(params['traces'], np.ndarray))
        assert(isinstance(params['weights'], np.ndarray))
        assert(params['num_features'] == ns)
        assert(params['num_actions'] == na)
        assert(np.shape(params['traces']) == (na, ns))
        assert(np.shape(params['weights']) == (na, ns))


    def test_from_config(self):
        """Test loading the algorithm from a config."""
        ns = 10
        na = 3
        algo_1 = vcf.algos.DiscreteQ(ns, na)
        cfg = algo_1.get_config()

        # Try to load from config
        algo_2 = vcf.algos.DiscreteQ.from_config(cfg)

        # Check that everything loaded correctly
        params = algo_2.get_config()
        assert(isinstance(params, dict))
        assert(isinstance(params['num_features'], int))
        assert(isinstance(params['num_actions'], int))
        assert(isinstance(params['traces'], np.ndarray))
        assert(isinstance(params['weights'], np.ndarray))
        assert(params['num_features'] == ns)
        assert(params['num_actions'] == na)
        assert(np.shape(params['traces']) == (na, ns))
        assert(np.shape(params['weights']) == (na, ns))
