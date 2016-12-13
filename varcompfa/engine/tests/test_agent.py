import numpy as np
import varcompfa as vcf


# TODO: Add more test cases
class TestAgent:
    def test_setup(self):
        # Set up the agent
        param_funcs = {
            'alpha' : 0.05,
            'gm'    : vcf.Constant(0.9999, 0),
            'gm_p'  : vcf.Constant(0.9999, 0),
            'lm'    : 0.1
        }
        phi   = vcf.BinaryVector(10)
        algo  = vcf.TD(len(phi))
        agent = vcf.Agent(algo, phi, param_funcs)

    def test_td_learn(self):
        pass

    def test_eval_params(self):
        # Test non-terminal case
        # Test terminal case
        pass

    def test_terminal_context(self):
        # Set up the agent
        param_funcs = {
            'alpha' : 0.05,
            'gm'    : vcf.Constant(0.9999, 0),
            'gm_p'  : vcf.Constant(0.9999, 0),
            'lm'    : 0.1
        }
        phi   = vcf.BinaryVector(10)
        algo  = vcf.TD(len(phi))
        agent = vcf.Agent(algo, phi, param_funcs)

        # No base context
        base_ctx = {}
        term_ctx = agent.terminal_context(base_ctx)
        assert(isinstance(term_ctx, dict))
        assert(term_ctx['done'] == True)
        assert(term_ctx['r'] == 0)
        assert(all(term_ctx['xp'] == 0))

        # Nonsense base context (should still be present)
        base_ctx = {'__'+str(i):i**2 for i in range(10)}
        term_ctx = agent.terminal_context(base_ctx)
        assert(isinstance(term_ctx, dict))
        assert(term_ctx['done'] == True)
        assert(term_ctx['r'] == 0)
        assert(all(term_ctx['xp'] == 0))
        assert(all(key in term_ctx for key in base_ctx.keys()))
        assert(term_ctx[key] == val for key, val in base_ctx.items())


    def test_act(self):
        pass

    def test_update(self):
        pass

    def test_get_td_error(self):
        pass

    def test_get_value(self):
        pass
