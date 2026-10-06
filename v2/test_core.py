import unittest
import torch
from v2.core import *

class CoreTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(4); torch.set_num_threads(2)
    def test_schedule_boundary_and_sampling(self):
        s=Schedule(8); z=torch.randn(5,4); t=torch.full((5,),8); noisy,e=s.noise(z,t); a,b=s.coeff(t)
        clean,eps=s.decode_v(noisy,t,a*e-b*z)
        torch.testing.assert_close(clean,z); torch.testing.assert_close(eps,e)
        self.assertLess(float(s.abar[-1]),1.1e-6)
        m=Student(Network(4,16,3,2)); torch.testing.assert_close(m(z,torch.zeros(5,dtype=torch.long),s),z)
        for steps in (1,2,4,8): self.assertTrue(torch.isfinite(sample_latents(m,s,z,steps)).all())
        t,k=s.pairs(64); self.assertTrue(torch.equal(t-k,torch.ones_like(t)))
    def test_frozen_teacher_target_and_ema(self):
        s=Schedule(8); teacher=Network(4,16,3,2).eval(); teacher.requires_grad_(False)
        student=Student(Network(4,16,3,2)); target=frozen_copy(student); initial=[p.clone() for p in target.parameters()]
        loss=distillation_loss(student,target,teacher,torch.randn(8,4),s); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        with torch.no_grad():
            for p in student.parameters(): p.add_(1)
        update_ema(target,student,.5)
        for p,q in zip(target.parameters(),initial): torch.testing.assert_close(p,q+.5)
    def test_surrogate_and_entanglement(self):
        q=QuantumFeatures(3,2,False); surrogate=SeparableSurrogate(3,2); surrogate.load_state_dict({'weights':q.weights.detach()})
        x=torch.randn(7,6,requires_grad=True)
        torch.testing.assert_close(q(x),surrogate(x),atol=2e-6,rtol=2e-5)
        g1=torch.autograd.grad(q(x).sum(),(x,q.weights),retain_graph=True)
        g2=torch.autograd.grad(surrogate(x).sum(),(x,surrogate.weights))
        for a,b in zip(g1,g2): torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        entangled=QuantumFeatures(3,2,True); entangled.load_state_dict(q.state_dict())
        self.assertGreater(float((entangled(x)-q(x)).abs().max().detach()),1e-3)
        entangled(x).square().mean().backward()
        self.assertGreater(float(entangled.weights.grad[-1,:,1].abs().max()),1e-5)
    def test_timing_and_parameter_accounting(self):
        model=Student(Network(4,16,3,2)); schedule=Schedule(8)
        model(torch.randn(4,4),torch.ones(4,dtype=torch.long),schedule).sum().backward()
        counts=parameter_counts(model); self.assertEqual(counts['trainable'],counts['with_gradient'])
        _,stats=synchronized_time(lambda:torch.ones(2),'cpu',1,3)
        self.assertEqual(len(stats['seconds']),3); self.assertGreater(stats['median_seconds'],0)
if __name__=='__main__': unittest.main()
