import deepxde as dde
import torch
# dde.backend.set_default_backend("pytorch")

import matplotlib.pyplot as plt

import autograd.numpy as anp
from autograd import elementwise_grad

from scipy import integrate
from scipy.interpolate import interp1d

cmap = plt.get_cmap('tab10')


# FitzHugh-Nagumo model
params = {
    'I' : 0.5,
    'a' : 0.8,
    'b' : 0.7,
    'tau' : 12.5
}

ini = (0,0)

def dv_dt(x, p=params):
    v,w = x.T
    return v - v**3/3 - w + p['I']

def dw_dt(x, p=params):
    v,w = x.T
    return (v + p['a'] - p['b']*w)/p['tau']

def diff_eq(t, x, p=params):
    return [diff_eqs[i](x) for i in range(len(diff_eqs))]

def gen_truedata(func, t, ini, params):
    sol = integrate.solve_ivp(func, (min(t), max(t)), ini, t_eval=t, args=(params,))
    return sol.y.T

# Get PRC with DeepXDE
def ode_system(x, y):
    t  = x
    Z0 = y

    Z0dot = torch.stack([torch.autograd.grad(Z0[:, i], t, torch.ones_like(Z0[:, i]), create_graph=True)[0] for i in range(len(diff_eqs))], axis=1).squeeze()

    # elementwise dot product
    constraint = torch.sum(Z0 * F_xlc, dim=1) - 1

    # Elementwise vector*matrix product
    diff_eq = Z0dot + torch.bmm(J_T, Z0.unsqueeze(-1)).squeeze(-1)

    return [constraint, diff_eq]


diff_eqs = [dv_dt, dw_dt]
eq_names = ['v', 'w']


# Get a limit cycle
T = 1000  # in ms
dt = 0.001 # in ms
t_sim = anp.linspace(0,T,int(T/dt))

x = gen_truedata(diff_eq, t_sim, ini, params)

# get peaks
threshold = (max(x[:,0])-min(x[:,0]))/4+min(x[:,0])

crossed = False
cts = []
pks = []
v_max_cur = -100

for i,v_ in enumerate(x[:,0]):

    if (v_ > threshold):
        if not crossed:
            crossed = True
            v_max_cur = v_
            i_peak = i
        else:
            if v_ > v_max_cur:
                i_peak = i
                v_max_cur = v_

    else:
        if crossed:
            pks.append(i_peak)
            crossed = False

P0 = (pks[-1] - pks[-2])*dt
f0 = 1/P0

# limit cycle
x_lc = x[pks[-2]:pks[-1]]
t = t_sim[pks[-2]:pks[-1]] - t_sim[pks[-2]]

# Time-derive the limit cycle
F_xlc_ = anp.array(diff_eq(t,x_lc)).T

# get the Jacobian of the limit cycle
J_ = anp.stack([elementwise_grad(diff_eqs[i])(x_lc) for i in range(len(diff_eqs))],axis=1)

geom = dde.geometry.TimeDomain(t[0], t[-1])

n_train = 300
n_bounds = 2

data = dde.data.PDE(
    geom,
    ode_system,
    [],
    n_train,
    n_bounds,
    num_test=0
)
data.test_x = data.train_x # hack

f = interp1d(t, F_xlc_.T)

F_xlc = torch.tensor(f(data.train_x.squeeze()).swapaxes(0, -1), dtype=torch.float32)


# Jacobian 
J_T = anp.zeros((len(data.train_x),len(diff_eqs),len(diff_eqs)))
for ii,jj in enumerate(J_.T):
    for i,j in enumerate(jj):
        f = interp1d(t, j)
        J_T[:,ii,i] = f(data.train_x.squeeze())
J_T = torch.tensor(J_T, dtype=torch.float32)


# Train the network
layer_size =  layer_size = [1] + [64] * 5 + [len(diff_eqs)]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile("adam", lr=0.0001)

losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# evaluate Z0 along original LC
Z0 = model.predict(t.reshape(-1,1))

plt.figure(tight_layout=True)
for i,(z0,var) in enumerate(zip(Z0.T,eq_names)):
    plt.subplot(len(diff_eqs),1,i+1)
    plt.plot(t/P0,z0)
    plt.xlabel('phi')
    plt.ylabel('dphi/d%s'%var)
# plt.show()

# evaluate results
plt.plot(t[1:]/P0, anp.diff(Z0,axis=0)/dt,linewidth=5,c='y',label='dZ0/dt')
plt.plot(t/P0,-(J_.swapaxes(1,2) @ Z0[..., None])[..., 0],linestyle='--',c='k',label='-J.T*Z0')
plt.xlabel('phi')
plt.legend()
# plt.show()

F_xlc_ = torch.tensor(F_xlc_, dtype=torch.float32)
Z0 = torch.tensor(Z0, dtype=torch.float32)
dot_product_results = torch.sum(Z0 * F_xlc_, dim=1)

plt.plot(t / P0, dot_product_results.detach())
plt.xlabel('phi')
plt.ylabel('Z0 * W0')
# plt.show()

# Convert to torch script
# `model` is a DeepXDE model, and `net` is the underlying PyTorch model
net = model.net

# Put the underlying PyTorch model in evaluation mode
net.eval()

# Move the model to the CPU
net.to('cpu')

# Dummy input tensor
dummy_input = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)

# Trace the model with the dummy input
traced_script_module = torch.jit.trace(net, dummy_input)

# Save the traced model
traced_script_module.save("model/traced_model.pt")

