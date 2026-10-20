"""Pilot components. v1 is retained unchanged for historical reproduction."""
import copy
import math
import time
import torch
from torch import nn

class Schedule:
    def __init__(self, steps=32, device='cpu'):
        if steps < 2:
            raise ValueError('steps must be >=2')
        self.steps = steps
        t = torch.linspace(0, 1, steps + 1, device=device)
        self.abar = torch.cos(t * math.pi / 2).square().clamp_min(1e-6)
        self.abar[0] = 1
    def coeff(self, t):
        a = self.abar[t].sqrt().unsqueeze(-1)
        b = (1-self.abar[t]).sqrt().unsqueeze(-1)
        return a, b
    def noise(self, z, t, eps=None):
        eps = torch.randn_like(z) if eps is None else eps
        a,b = self.coeff(t)
        return a*z+b*eps, eps
    def decode_v(self, z, t, v):
        a,b = self.coeff(t)
        return a*z-b*v, b*z+a*v
    def pairs(self, batch):
        t = torch.randint(1,self.steps+1,(batch,),device=self.abar.device)
        return t,t-1
    def ddim(self, z,t,s,v):
        clean,eps = self.decode_v(z,t,v)
        a,b = self.coeff(s)
        return a*clean+b*eps

class QuantumFeatures(nn.Module):
    """RY/RZ/CZ circuit with final basis mixing, exact statevector expectations.
    Separable control changes only entangling gates. No shot/hardware model.
    Data-independent gates are fused into one matrix per forward; `reference`
    keeps the gate-by-gate simulation for tests. With reupload, the encoding
    gates are repeated before every trainable layer.
    """
    def __init__(self, qubits=4, depth=2, entangle=True, reupload=False):
        super().__init__()
        if qubits < 2 or depth < 1:
            raise ValueError('qubits>=2 and depth>=1 required')
        self.qubits,self.depth,self.entangle,self.reupload = qubits,depth,entangle,reupload
        self.weights = nn.Parameter(torch.randn(depth,qubits,2)*0.1)
        basis = torch.arange(2**qubits)
        bits = ((basis[:,None] >> torch.arange(qubits-1,-1,-1))&1)
        self.register_buffer('signs', (1-2*bits).float())
        pairs=sorted({tuple(sorted((w,(w+1)%qubits))) for w in range(qubits)})
        self.register_buffer('cz', torch.stack([1-2*bits[:,a]*bits[:,b] for a,b in pairs]).prod(0).float())
    def gate(self, state, theta, wire, rz=False):
        batch = state.shape[0]
        x = state.reshape([batch]+[2]*self.qubits).movedim(wire+1,-1)
        if rz:
            g = torch.diag_embed(torch.stack([torch.exp(-.5j*theta),torch.exp(.5j*theta)],-1))
        else:
            c,s = torch.cos(theta/2),torch.sin(theta/2)
            g = torch.stack([c,-s,s,c],-1).reshape(batch,2,2).to(torch.complex64)
        x = torch.einsum('bij,bkj->bki',g,x.reshape(batch,-1,2))
        return x.reshape([batch]+[2]*self.qubits).movedim(-1,wire+1).reshape(batch,-1)
    def encode(self, state, x):
        for w in range(self.qubits):
            state=self.gate(state,x[:,w],w)
            state=self.gate(state,x[:,w+self.qubits],w,True)
        return state
    @staticmethod
    def single_qubit(ry, rz):
        # RZ(rz) @ RY(ry), transposed for the row-state convention (state @ matrix).
        c,s=torch.cos(ry/2),torch.sin(ry/2); m,p=torch.exp(-.5j*rz),torch.exp(.5j*rz)
        return torch.stack([torch.stack([c*m,s*p],-1),torch.stack([-s*m,c*p],-1)],-2)
    def kron(self, mats):
        out=mats[0]
        for m in mats[1:]: out=(out[:,None,:,None]*m[None,:,None,:]).reshape(out.shape[0]*2,-1)
        return out
    def layer_matrix(self, layer, device=None):
        # Row i is the layer applied to basis state i, so a batch of states maps as state @ matrix.
        mats=self.single_qubit(self.weights[layer,:,0],self.weights[layer,:,1])
        matrix=self.kron(list(mats))
        return matrix*self.cz if self.entangle else matrix
    def mixing_matrix(self, device=None):
        ry=torch.full((self.qubits,),math.pi/4,device=self.weights.device)
        return self.kron(list(self.single_qubit(ry,torch.zeros_like(ry))))
    def product_state(self, x):
        theta,phi=x[:,:self.qubits],x[:,self.qubits:]
        amps=torch.stack([torch.cos(theta/2)*torch.exp(-.5j*phi),torch.sin(theta/2)*torch.exp(.5j*phi)],-1)
        state=amps[:,0]
        for w in range(1,self.qubits): state=(state[:,:,None]*amps[:,w,None,:]).reshape(x.shape[0],-1)
        return state
    def forward(self, x):
        layers=[self.layer_matrix(l,x.device) for l in range(self.depth)]; mixing=self.mixing_matrix(x.device)
        if self.reupload:
            state=self.product_state(x)@layers[0]
            for layer in layers[1:]: state=self.encode(state,x)@layer
            state=state@mixing
        else:
            fused=mixing
            for layer in reversed(layers): fused=layer@fused
            state=self.product_state(x)@fused
        return state.abs().square()@self.signs
    def reference(self, x):
        state = torch.zeros(x.shape[0],2**self.qubits,device=x.device,dtype=torch.complex64)
        state[:,0] = 1
        for layer in range(self.depth):
            if layer==0 or self.reupload: state=self.encode(state,x)
            for w in range(self.qubits):
                state=self.gate(state,self.weights[layer,w,0].expand(x.shape[0]),w)
                state=self.gate(state,self.weights[layer,w,1].expand(x.shape[0]),w,True)
            if self.entangle:
                state=state*self.cz
        for w in range(self.qubits):
            state=self.gate(state,torch.full_like(x[:,0],math.pi/4),w)
        return state.abs().square()@self.signs

class SeparableSurrogate(nn.Module):
    """Exact algebraic substitute for QuantumFeatures(entangle=False), all depths, with or without reupload."""
    def __init__(self, qubits=4, depth=2, reupload=False):
        super().__init__()
        self.qubits,self.reupload=qubits,reupload
        self.weights=nn.Parameter(torch.randn(depth,qubits,2)*.1)
    @staticmethod
    def rotate(b,ry,rz):
        bx,by,bz=b
        cy,sy=torch.cos(ry),torch.sin(ry); bx,bz=cy*bx+sy*bz,-sy*bx+cy*bz
        cz,sz=torch.cos(rz),torch.sin(rz); bx,by=cz*bx-sz*by,sz*bx+cz*by
        return bx,by,bz
    def forward(self,x):
        q=self.qubits
        b=(torch.sin(x[:,:q])*torch.cos(x[:,q:]),torch.sin(x[:,:q])*torch.sin(x[:,q:]),torch.cos(x[:,:q]))
        for layer,weights in enumerate(self.weights):
            if layer and self.reupload: b=self.rotate(b,x[:,:q],x[:,q:])
            b=self.rotate(b,weights[:,0],weights[:,1])
        return (b[2]-b[0])/math.sqrt(2)

class ClassicalBranch(nn.Module):
    """Classical control for the circuit block: same input/output shapes and range, cross-feature interactions."""
    def __init__(self, qubits=4):
        super().__init__(); self.layer=nn.Sequential(nn.Linear(2*qubits,qubits),nn.Tanh())
    def forward(self,x): return self.layer(x)

class Network(nn.Module):
    """hidden=0 removes the MLP backbone, leaving only compressor -> branch -> readout."""
    def __init__(self, latent=16, hidden=64, qubits=4, depth=2, kind='hybrid', reupload=False):
        super().__init__()
        self.kind=kind
        if kind not in ('classical','hybrid','no_entanglement','surrogate','classical_branch'): raise ValueError(kind)
        if hidden==0 and kind=='classical': raise ValueError('classical network needs hidden>0')
        self.backbone=nn.Sequential(nn.Linear(latent+3,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,latent)) if hidden else None
        if kind != 'classical':
            self.compressor=nn.Sequential(nn.Linear(latent+3,2*qubits),nn.Tanh())
            self.quantum=(SeparableSurrogate(qubits,depth,reupload) if kind=='surrogate' else ClassicalBranch(qubits) if kind=='classical_branch'
                          else QuantumFeatures(qubits,depth,kind=='hybrid',reupload))
            self.readout=nn.Linear(qubits,latent)
    def branch(self,features):
        return self.readout(self.quantum(math.pi*self.compressor(features)))
    def forward(self,z,t,schedule):
        u=t.float()/schedule.steps
        features=torch.cat([z,u[:,None],torch.sin(math.pi*u)[:,None],torch.cos(math.pi*u)[:,None]],-1)
        y=self.backbone(features) if self.backbone is not None else 0
        if self.kind!='classical':
            y=y+self.branch(features)
        return y

class Student(nn.Module):
    def __init__(self, network):
        super().__init__(); self.network=network
    def forward(self,z,t,schedule):
        a,b=schedule.coeff(t)
        return a.square()*z+b*self.network(z,t,schedule)

def frozen_copy(model):
    target=copy.deepcopy(model).eval()
    target.requires_grad_(False)
    return target

@torch.no_grad()
def update_ema(target,online,decay):
    if not 0<=decay<1: raise ValueError('EMA decay must be in [0,1)')
    for p,q in zip(target.parameters(),online.parameters()): p.lerp_(q,1-decay)
    for p,q in zip(target.buffers(),online.buffers()): p.copy_(q)

def distillation_loss(student,target,teacher,z,schedule):
    t,s=schedule.pairs(z.shape[0]); noisy,_=schedule.noise(z,t)
    with torch.no_grad():
        nxt=schedule.ddim(noisy,t,s,teacher(noisy,t,schedule))
        endpoint=target(nxt,s,schedule)
    return (student(noisy,t,schedule)-endpoint).square().mean()

@torch.no_grad()
def sample_latents(model,schedule,noise,steps,teacher=False):
    if not 1<=steps<=schedule.steps: raise ValueError('invalid sampling steps')
    levels=torch.linspace(schedule.steps,1,steps,device=noise.device).round().long()
    z=noise.clone()
    for i,level in enumerate(levels):
        t=torch.full((z.shape[0],),int(level),device=z.device,dtype=torch.long)
        if teacher:
            s=torch.full_like(t,int(levels[i+1]) if i+1<len(levels) else 0)
            z=schedule.ddim(z,t,s,model(z,t,schedule))
        else:
            clean=model(z,t,schedule)
            if i+1<len(levels):
                s=torch.full_like(t,int(levels[i+1])); z,_=schedule.noise(clean,s)
            else: z=clean
    return z

def synchronized_time(fn,device,warmup=2,repeats=5):
    def sync():
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        elif str(device).startswith('mps'): torch.mps.synchronize()
    for _ in range(warmup): fn(); sync()
    measurements=[]
    for _ in range(repeats):
        sync(); start=time.perf_counter(); result=fn(); sync()
        measurements.append(time.perf_counter()-start)
    return result, {'seconds':measurements,'median_seconds':float(torch.tensor(measurements).median()),'warmup':warmup,'repeats':repeats}

def parameter_counts(model):
    return {'registered':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None)}
