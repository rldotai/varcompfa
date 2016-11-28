import numpy as np
import varcompfa as vcf


class TestTD:
    def test_setup(self):
        for ns in range(1, 10):
            algo = vcf.algos.TD(ns)
            assert(isinstance(algo, vcf.algos.LearningAlgorithm))

    def test_learn(self):
        for ns in range(1, 10):
            algo = vcf.algos.TD(ns)

            # Create fake but correctly typed data for testing the update
            x       = np.random.random(ns)  + 0.1
            xp      = np.random.random(ns)
            r       = np.random.random()
            alpha   = np.random.random()    + 0.1
            gm      = np.random.random()    + 0.1
            gm_p    = np.random.random()    + 0.1
            lm      = np.random.random()    + 0.1

            # Get the parameter values prior to the update
            w_0 = np.copy(algo.w)
            z_0 = np.copy(algo.z)

            # Perform the update
            algo.learn(x, r, xp, alpha, gm, gm_p, lm)

            # Check that something happened
            assert(np.any(algo.w != w_0))
            assert(np.any(algo.z != z_0))


    def test_update(self):
        for ns in range(1, 10):
            algo = vcf.algos.TD(ns)
            params = dict(
                x       = np.random.random(ns)  + 0.1,
                xp      = np.random.random(ns),
                r       = np.random.random(),
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
