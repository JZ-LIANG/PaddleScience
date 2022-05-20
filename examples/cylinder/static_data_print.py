# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.disable_static()


# load real data
def GetRealPhyInfo(time):
    real_data = np.load("flow_unsteady_re200/flow_re200_" + str(time) +
                        "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    return real_data[:, 0:7]


# define start time and time step
start_time = 100

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2), extent=(25, 8, 2), circle_center=cc, circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
print("ATTENTION! ####### The npoints must be same in your code! ########")
geo_disc = geo.discretize(npoints=40000, method="sampling")

# the real_cord need to be added in geo_disc
real_cord = GetRealPhyInfo(start_time)[:, 0:3]
geo_disc.user = real_cord

# N-S equation
pde = psci.pde.NavierStokes(
    nu=0.01,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[0.01, 0.01, 0.01, 0.01])

pde.set_time_interval([100.0, 110.0])

# boundary condition on left side: u=10, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=1.0, weight=1.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0, weight=1.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0, weight=1.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# pde discretization 
pde_disc = pde.discretize(
    time_method="implicit", time_step=1, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver parameter
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

current_interior = np.zeros(
    (len(pde_disc.geometry.interior), 3)).astype(np.float32)
solver.feed_data_interior_cur(current_interior)  # add u(n) interior
solver.feed_data_user_cur(GetRealPhyInfo(100)[:, 3:6])  # add u(n) user 
solver.feed_data_user_next(GetRealPhyInfo(101)[:, 3:7])  # add u(n+1) user

print("Please attntion the static input shape")
print("The interior shape is:")
print(geo_disc.interior.shape)
print("The real data shape is:")
print(geo_disc.user.shape)
print("The boundary shape is as follows:")
print(geo_disc.boundary['left'].shape)
print(geo_disc.boundary['right'].shape)
print(geo_disc.boundary['circle'].shape)

print("Please attntion the static label shape")
print("The label shape is:")
n = len(solver.labels)
for i in range(n):
    print(len(solver.labels[i]))
