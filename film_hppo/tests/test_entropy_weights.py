"""Independent entropy coefficients: compatibility, real loss and gradients."""
from copy import deepcopy
import math
import unittest

import torch
from ding.config import compile_config
from film_rl.config import validate
from film_rl.hppo import build_configs, build_model
from film_rl.policy import FilmPPOPolicy
from fixtures import base_config
import test_selected_continuous_policy as selected_tests


class EntropyWeightTests(unittest.TestCase):
    def make_policy(self, discrete=None, continuous=None):
        cfg=base_config('.')
        cfg['training'].update(batch_size=4,epoch_per_collect=1,learning_rate=1e-4)
        if discrete is not None:cfg['training']['entropy_weight_discrete']=discrete
        if continuous is not None:cfg['training']['entropy_weight_continuous']=continuous
        main,create=build_configs(cfg);create.policy.type+='_command'
        compiled=compile_config(main,seed=0,auto=True,create_cfg=create,save_cfg=False)
        torch.manual_seed(42)
        return cfg,FilmPPOPolicy(compiled.policy,model=build_model(cfg,main),enable_field=['learn','collect'])

    def test_legacy_and_explicit_equal_weights_have_identical_updates(self):
        cfg,legacy=self.make_policy()
        _,explicit=self.make_policy(.001,.001)
        data=selected_tests.SelectedPolicyUpdateTests().samples(cfg,legacy,count=4)
        torch.manual_seed(867)
        log1=legacy.learn_mode.forward(deepcopy(data))[0]
        torch.manual_seed(867)
        log2=explicit.learn_mode.forward(deepcopy(data))[0]
        self.assertEqual(log1['total_loss'],log2['total_loss'])
        for k,v in legacy._model.state_dict().items():
            torch.testing.assert_close(v,explicit._model.state_dict()[k],rtol=0,atol=0)

    def test_independent_coefficients_enter_native_total_loss_and_monitors(self):
        cfg,policy=self.make_policy(.03,.0001)
        logs=policy.learn_mode.forward(selected_tests.SelectedPolicyUpdateTests().samples(cfg,policy,count=4))
        row=logs[0]
        expected=.03*row['entropy_discrete']+.0001*row['entropy_continuous']
        self.assertAlmostEqual(row['entropy_bonus'],expected,places=7)
        self.assertAlmostEqual(row['total_loss'],row['policy_loss']+.5*row['value_loss']-expected,places=6)
        self.assertTrue(all(math.isfinite(v) for v in row.values()))
        self.assertIn('entropy_discrete',policy._monitor_vars_learn())

    def test_zero_coefficients_do_not_give_gaussian_entropy_gradients_on_eos(self):
        cfg,policy=self.make_policy(.03,0.)
        data=selected_tests.SelectedPolicyUpdateTests().samples(cfg,policy,count=4,eos_only=True)
        before=policy._model.actor_head[1].log_sigma_param.detach().clone()
        policy.learn_mode.forward(data)
        torch.testing.assert_close(before,policy._model.actor_head[1].log_sigma_param,rtol=0,atol=0)

    def test_invalid_coefficients_rejected(self):
        for key in ('entropy_weight_discrete','entropy_weight_continuous'):
            for value in (True,-.01,float('nan'),float('inf'),'0.1'):
                cfg=base_config('.');cfg['training'][key]=value
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,key):validate(cfg)


if __name__=='__main__':unittest.main()
