# Commented out IPython magic to ensure Python compatibility.
import numpy as onp
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu
from jax import config
from jax import lax
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

import scipy.io
from scipy.interpolate import griddata
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import least_squares

from scipy.integrate import odeint as scipy_odeint
from mpl_toolkits.mplot3d import Axes3D


# Define the neural net
def init_layer(key, d_in, d_out):
    k1, k2 = random.split(key)
    glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
    W = glorot_stddev * random.normal(k1, (d_in, d_out))
    b = np.zeros(d_out)
    return W, b


def MLP(layers, activation=relu):
    ''' Vanilla MLP'''

    def init(rng_key):
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply

# Define the model
def ResNet(layers, activation=relu):

    def init(rng_key):
        params = []
        keys = random.split(rng_key, len(layers))
        for i in range(len(layers) - 1):
            W = random.normal(keys[i], (layers[i], layers[i+1])) / np.sqrt(layers[i])
            b = np.zeros(layers[i+1])
            params.append((W, b))
        return params

    def apply(params, inputs):
        x = inputs
        for i in range(0, len(params) - 1, 2):
            W1, b1 = params[i]
            W2, b2 = params[i+1]
            h = activation(np.dot(x, W1) + b1)
            h = activation(np.dot(h, W2) + b2)
            x = x + h  
        if len(params) % 2 == 1:
            W, b = params[-1]
            outputs = np.dot(x, W) + b
        else:
            outputs = x
        return outputs

    return init, apply


# Define the model
class PINN:
    def __init__(self, layers, states0, t0, t1, tol):

        self.states0 = states0
        self.t0 = t0
        self.t1 = t1

        # Grid
        n_t = 300
        eps = 0.1 * self.t1
        self.t = np.linspace(self.t0, self.t1 + eps, n_t)

        self.M = np.triu(np.ones((n_t, n_t)), k=1).T
        self.tol = tol

        self.a = 3
        self.b = 2.7
        self.c = 4.7
        self.d = 2
        self.h = 9

        self.init, self.apply = ResNet(layers, activation=np.tanh)
        params = self.init(random.PRNGKey(1234))

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                           decay_steps=5000,
                                                                           decay_rate=0.9))
        self.opt_state = self.opt_init(params)
        _, self.unravel = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []

    def neural_net(self, params, t):
        t = np.stack([t])
        outputs = self.apply(params, t) * t
        x = outputs[0] + self.states0[0]
        y = outputs[1] + self.states0[1]
        z = outputs[2] + self.states0[2]
        return x, y, z

    def x_fn(self, params, t):
        x, _, _ = self.neural_net(params, t)
        return x

    def y_fn(self, params, t):
        _, y, _ = self.neural_net(params, t)
        return y

    def z_fn(self, params, t):
        _, _, z = self.neural_net(params, t)
        return z

    def residual_net(self, params, t):
        x, y, z = self.neural_net(params, t)
        x_t = grad(self.x_fn, argnums=1)(params, t)
        y_t = grad(self.y_fn, argnums=1)(params, t)
        z_t = grad(self.z_fn, argnums=1)(params, t)

        res_1 = x_t - y + self.a * x - self.b * y * z
        res_2 = y_t - self.c * y + x * z - z
        res_3 = z_t - self.d * x * y + self.h * z

        return res_1, res_2, res_3

    def loss_ics(self, params):
        # Compute forward pass
        x_pred, y_pred, z_pred = self.neural_net(params, self.t0)
        # Compute loss

        loss_x_ic = np.mean((self.states0[0] - x_pred) ** 2)
        loss_y_ic = np.mean((self.states0[1] - y_pred) ** 2)
        loss_z_ic = np.mean((self.states0[2] - z_pred) ** 2)
        return loss_x_ic + loss_y_ic + loss_z_ic

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, tol):
        r1_pred, r2_pred, r3_pred = vmap(self.residual_net, (None, 0))(params, self.t)
        W = lax.stop_gradient(np.exp(- tol * self.M @ (r1_pred ** 2 + r2_pred ** 2 + r3_pred ** 2)))
        return r1_pred, r2_pred, r3_pred, W
        


    @partial(jit, static_argnums=(0,))
    def loss_res(self, params):
        # Compute forward pass
        r1_pred, r2_pred, r3_pred, W = self.residuals_and_weights(params, self.tol)
        # Compute loss
        loss_res = np.mean(W * (r1_pred ** 2 + r2_pred ** 2 + r3_pred ** 2))
        return loss_res

    @partial(jit, static_argnums=(0,))
    def loss(self, params):
        loss_res = self.loss_res(params)
        loss = loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params)
        return self.opt_update(i, g, opt_state)

     # Optimize parameters in a loop
    def train(self, nIter=10000):
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state)

            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                loss_value = self.loss(params)
                loss_ics_value = self.loss_ics(params)
                loss_res_value = self.loss_res(params)
                _, _, _, W_value = self.residuals_and_weights(params, self.tol)
               

                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)

                pbar.set_postfix({'Loss': loss_value,
                                  'loss_ics': loss_ics_value,
                                  'loss_res': loss_res_value,
                                  'W_min': W_value.min()
                                 })

                if W_value.min() > 0.99:
                    break

    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, t_star):
        x_pred, y_pred, z_pred = vmap(self.neural_net, (None, 0))(params, t_star)
        return x_pred, y_pred, z_pred
    
    
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return y - a * x + b * y * z, c * y - x * z + z, d * x * y - h * z  # Derivatives


a = 3
b = 2.7
c = 4.7
d = 2
h = 9

state0 = [5.0, 0.0, -4.0]

T = 15
t_star = onp.arange(0, T, 0.01)
states = scipy_odeint(f, state0, t_star)


# Create PINNs model
t0 = 0.0
t1 = 0.5
tol = 0.1

tol_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
 
layers = [1, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 3]  




x_pred_list = []
y_pred_list = []
z_pred_list = []
params_list = []
losses_list = []

state0 = np.array([5.0, 0.0, -4.0])
t = np.arange(t0, t1, 0.01)
for k in range(int(T / t1)):
    # Initialize model
    print('Final Time: {}'.format((k + 1) * t1))
    model = PINN(layers, state0, t0, t1, tol)



    for tol in tol_list:
        model.tol = tol
        print('tol:', model.tol)
        # Train
        model.train(nIter=300000)
    
    

    params = model.get_params(model.opt_state)
    x_pred, y_pred, z_pred = model.predict_u(params, t)
    x0_pred, y0_pred, z0_pred = model.neural_net(params, model.t1)
    state0 = np.array([x0_pred, y0_pred, z0_pred])

    # Store predictions
    x_pred_list.append(x_pred)
    y_pred_list.append(y_pred)
    z_pred_list.append(z_pred)
    losses_list.append([model.loss_ics_log, model.loss_res_log])

    # Store params
    flat_params, _ = ravel_pytree(params)
    params_list.append(flat_params)

    np.save('x_pred_list.npy', x_pred_list)
    np.save('y_pred_list.npy', y_pred_list)
    np.save('z_pred_list.npy', z_pred_list)
    np.save('params_list.npy', params_list)


    # Error
    t_star = onp.arange(t0, (k + 1) * t1, 0.01)
    states = scipy_odeint(f, [5.0, 0.0, -4.0], t_star)

    x_preds = np.hstack(x_pred_list)
    y_preds = np.hstack(y_pred_list)
    z_preds = np.hstack(z_pred_list)

    error_x = np.linalg.norm(x_preds - states[:, 0]) / np.linalg.norm(states[:, 0])
    error_y = np.linalg.norm(y_preds - states[:, 1]) / np.linalg.norm(states[:, 1])
    error_z = np.linalg.norm(z_preds - states[:, 2]) / np.linalg.norm(states[:, 2])
    print('Relative l2 error x: {:.3e}'.format(error_x))
    print('Relative l2 error y: {:.3e}'.format(error_y))
    print('Relative l2 error z: {:.3e}'.format(error_z))
