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

import paddle
import six
import time
import numpy as np
import paddlescience as psci
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.static import global_scope
from paddle.fluid.incubate.ad_transform.primx import prim2orig, enable_prim, prim_enabled

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
enable_prim()


def GenInitPhyInfo(xyz):
    uvwp = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvwp[i][0] = 1.0
    return uvwp

def GetRealPhyInfo(time):
    use_real_data = False
    if use_real_data is True:
        xyzuvwp = np.load("csv/flow_re20_" + str(time) + "_xyzuvwp.npy") 
    else:
        xyzuvwp = np.ones((1000, 7)).astype(np.float32)
    return xyzuvwp

def compile_and_convert_back_to_program(program=None,
                                   feed=None,
                                   fetch_list=None,
                                   fetch_var_name='fetch',
                                   scope=None,
                                   use_prune=False,
                                   loss_name=None):
    def _add_fetch_ops(program, fetch_list, fetch_var_name):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        # append fetch_operators
        if not fluid.executor.has_fetch_operators(global_block, fetch_list,
                                                  fetch_var_name, 'fetch'):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types), (
                        "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})
        return tmp_program

    def _remove_fetch_ops(program):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == 'fetch':
                global_block._remove_op(idx)

        return tmp_program

    def _compile(program, loss_name=None):
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        exec_strategy.num_threads = 1

        compiled_program = paddle.static.CompiledProgram(
            program).with_data_parallel(
                loss_name=loss_name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

        return compiled_program

    if program is None:
        program = default_main_program()

    if scope is None:
        scope = global_scope()

    executor = paddle.static.Executor()

    fetch_list = executor._check_fetch_list(fetch_list)
    fetch_list, optimize_ops = executor._split_optimize_ops_in_fetch_list(
        fetch_list)

    if optimize_ops:
        raise ValueError("Unsupport to fetch optimize OP.")

    if use_prune:
        program = executor._prune_program(program, feed, fetch_list, optimize_ops)
        feed = executor._update_feed(program, feed)

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = _compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope, paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    #ir_graph.draw(save_path='./', name='compiled_graph')
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    #paddle.static.save(final_program, "final")
    return final_program

def init_algo():
    circle_center = (0.0, 0.0)
    circle_radius = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -0.5),
        extent=(25, 8, 0.5),
        circle_center=circle_center,
        circle_radius=circle_radius)

    geo.add_boundary(name="top", criteria=lambda x, y, z: z == 0.5)
    geo.add_boundary(name="down", criteria=lambda x, y, z: z == -0.5)
    geo.add_boundary(name="left", criteria=lambda x, y, z: x == -8)
    geo.add_boundary(name="right", criteria=lambda x, y, z: x == 25)
    geo.add_boundary(name="front", criteria=lambda x, y, z: y == -8)
    geo.add_boundary(name="back", criteria=lambda x, y, z: y == 8)
    geo.add_boundary(
        name="circle",
        criteria=lambda x, y, z: (x - circle_center[0])**2 + (y - circle_center[1])**2 == circle_radius**2
    )

    # N-S
    pde = psci.pde.NavierStokes(nu=0.05, rho=1.0, dim=3, time_dependent=True)

    # set bounday condition
    bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
    bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_top_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_down_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_right_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_front_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_front_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_front_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_back_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_back_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_back_w = psci.bc.Dirichlet('w', rhs=0.0)

    # TODO 3. circle boundry
    bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

    pde.add_geometry(geo)

    # add bounday and boundary condition
    pde.add_bc("top", bc_top_u, bc_top_v, bc_top_w)
    pde.add_bc("down", bc_down_u, bc_down_v, bc_down_w)
    pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
    pde.add_bc("right", bc_right_u, bc_right_v, bc_right_w)
    pde.add_bc("front", bc_front_u, bc_front_v, bc_front_w)
    pde.add_bc("back", bc_back_u, bc_back_v, bc_back_w)
    pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)
    # Discretization
    pde_disc = psci.discretize(
        pde,
        time_method="implicit",
        time_step=0.5,
        space_npoints=60000,
        space_method="sampling")

    # Get real data
    real_xyzuvwp = GetRealPhyInfo(0.5)
    real_xyz = real_xyzuvwp[:, 0:3]
    real_uvwp = real_xyzuvwp[:, 3:7]

    # load real physic data in geo
    pde_disc.geometry.data = real_xyz

    # Network
    # TODO: remove num_ins and num_outs
    net = psci.network.FCNet(
        num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

    # Loss, TO rename
    # bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
    loss = psci.loss.L2()

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)
    return algo, pde_disc

def slove_static():
    algo, pde_disc = init_algo()
    # create inputs/labels and its attributes
    inputs, inputs_attr = algo.create_inputs(pde_disc)
    labels, labels_attr = algo.create_labels(pde_disc)

    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program,
                                     startup_program):
        algo.net.make_network_static()
        inputs_var = []
        labels_var = []
        outputs_var = []
        # inputs
        for i in range(len(inputs)):
            #inputs
            input = paddle.static.data(
                name='input' + str(i),
                shape=inputs[i].shape,
                dtype='float32')
            input.stop_gradient = False
            inputs_var.append(input)

        for i in range(len(labels)):
            #labels
            if i in [0, 1, 2]:
                shape = (51982, )
            else:
                shape = (1000, )
            label = paddle.static.data(
                name='label' + str(i),
                shape=shape,
                dtype='float32')
            label.stop_gradient = False
            labels_var.append(label)
        
        for var in inputs_var:
            ret = algo.net.nn_func(var)
            outputs_var.append(ret)


        # bc loss
        bc_loss = 0.0
        for i, name_b in enumerate(inputs_attr["boundary"].keys()):
            # from outputs_var[1] to outputs_var[7]
            out_i = outputs_var[i+1]
            for j in range(len(pde_disc.bc[name_b])):
                rhs_b = labels_attr["bc"][name_b][j]["rhs"]
                bc_loss += paddle.norm((out_i[:, j]-rhs_b)*(out_i[:, j]-rhs_b)*1.0, p=1)

        # eq loss
        input_i = inputs_var[0] # (51982, 3)
        out_i = outputs_var[0] # (51982, 4)
        x = input_i[:, 0]
        y = input_i[:, 1]
        z = input_i[:, 2]
        u = out_i[:, 0]
        v = out_i[:, 1]
        w = out_i[:, 2]
        p = out_i[:, 3]
        u_n = labels_var[0]
        v_n = labels_var[1]
        w_n = labels_var[2]
        jac0, = paddle.static.gradients([u], [input_i]) # du/dx, du/dy, du/dz
        jac1, = paddle.static.gradients([v], [input_i]) # dv/dx, dv/dy, dv/dz
        jac2, = paddle.static.gradients([w], [input_i]) # dw/dx, dw/dy, dw/dz
        jac3, = paddle.static.gradients([p], [input_i]) # dp/dx, dp/dy, dp/dz
        hes0, = paddle.static.gradients([jac0[:, 0]], [input_i]) # du*du/dx*dx, du*du/dx*dy, du*du/dx*dz
        hes1, = paddle.static.gradients([jac0[:, 1]], [input_i]) # du*du/dy*dx, du*du/dy*dy, du*du/dy*dz
        hes2, = paddle.static.gradients([jac0[:, 2]], [input_i]) # du*du/dz*dx, du*du/dz*dy, du*du/dz*dz
        hes3, = paddle.static.gradients([jac1[:, 0]], [input_i]) # dv*dv/dx*dx, dv*dv/dx*dy, dv*dv/dx*dz
        hes4, = paddle.static.gradients([jac1[:, 1]], [input_i]) # dv*dv/dy*dx, dv*dv/dy*dy, dv*dv/dy*dz
        hes5, = paddle.static.gradients([jac1[:, 2]], [input_i]) # dv*dv/dz*dx, dv*dv/dz*dy, dv*dv/dz*dz
        hes6, = paddle.static.gradients([jac2[:, 0]], [input_i]) # dw*dw/dx*dx, dw*dw/dx*dy, dw*dw/dx*dz
        hes7, = paddle.static.gradients([jac2[:, 1]], [input_i]) # dw*dw/dy*dx, dw*dw/dy*dy, dw*dw/dy*dz
        hes8, = paddle.static.gradients([jac2[:, 2]], [input_i]) # dw*dw/dz*dx, dw*dw/dz*dy, dw*dw/dz*dz
        nu = 0.05
        rho = 1.0
        continuty = jac0[:, 0] + jac1[:, 1] + jac2[:, 2]
        # + 2.0*u(x, y, z) - 2.0*u_n(x, y, z)
        momentum_x = 2.0 * u - 2.0 * u_n + u * jac0[:, 0] + v * jac0[:, 1] + w * jac0[:, 2] - \
                    nu / rho * hes0[:, 0] - nu / rho * hes1[:, 1] - nu / rho * hes2[:, 2] + \
                    1.0 / rho * jac3[:, 0]
        momentum_y = 2.0 * v - 2.0 * v_n + u * jac1[:, 0] + v * jac1[:, 1] + w * jac1[:, 2] - \
                    nu / rho * hes3[:, 0] - nu / rho * hes4[:, 1] - nu / rho * hes5[:, 2] + \
                    1.0 / rho * jac3[:, 1]
        momentum_z = 2.0 * w - 2.0 * w_n + u * jac2[:, 0] + v * jac2[:, 1] + w * jac2[:, 2] - \
                    nu / rho * hes6[:, 0] - nu / rho * hes7[:, 1] - nu / rho * hes8[:, 2] + \
                    1.0 / rho * jac3[:, 2]
        eq_loss = paddle.norm(continuty, p=2)*paddle.norm(continuty, p=2) + \
                  paddle.norm(momentum_x, p=2)*paddle.norm(momentum_x, p=2) + \
                  paddle.norm(momentum_y, p=2)*paddle.norm(momentum_y, p=2) + \
                  paddle.norm(momentum_z, p=2)*paddle.norm(momentum_z, p=2)

        # data_loss
        input_i = inputs_var[8]
        out_i = outputs_var[8]
        label3 = labels_var[3]
        label4 = labels_var[4]
        label5 = labels_var[5]
        data_loss = paddle.norm(out_i[:, 0] - label3, p=2) + \
                    paddle.norm(out_i[:, 1] - label4, p=2) + \
                    paddle.norm(out_i[:, 2] - label5, p=2)

        # total_loss
        total_loss = paddle.sqrt(bc_loss) + paddle.sqrt(eq_loss) + paddle.sqrt(data_loss)

        paddle.fluid.optimizer.AdamOptimizer(0.001).minimize(total_loss)
        if prim_enabled():
            prim2orig(main_program.block(0))


    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    feeds = dict()
    for i in range(len(inputs)):
        feeds['input' + str(i)] = inputs[i]

    real_xyzuvwp = GetRealPhyInfo(0.5)
    real_xyz = real_xyzuvwp[:, 0:3]
    real_uvwp = real_xyzuvwp[:, 3:7]
    uvw = GenInitPhyInfo(pde_disc.geometry.interior)
    n = pde_disc.geometry.interior.shape[0]
    self_lables = algo.feed_labels_data_n(labels=labels, labels_attr=labels_attr, data_n=uvw)
    self_lables = algo.feed_labels_data(labels=self_lables, labels_attr=labels_attr, data=real_uvwp)
    for i in range(len(self_lables)):
        feeds['label' + str(i)] = self_lables[i]

    fetchs = [total_loss.name]
    for var in outputs_var:
        fetchs.append(var.name)

    main_program = compile_and_convert_back_to_program(main_program, feed=feeds, fetch_list=fetchs, use_prune=False)

    # Solver train t0 -> t1
    print("###################### start time=0.5 train task ############")
    for epoch in range(2):
        rslt = exe.run(main_program,
                      feed=feeds, 
                      fetch_list=fetchs)
        print("static epoch: " + str(epoch + 1), "loss: ", rslt[0])
    uvw_t1 = rslt[1: ]
    uvw_t1 = uvw_t1[0]
    uvw_t1 = np.array(uvw_t1)

    # Solver train t1 -> tn
    time_step = 9
    current_uvw = uvw_t1[:, 0:3]
    for i in range(time_step):
        if i == 3:
            begin_time = time.time()

        current_time = 0.5 + (i + 1) * 0.5
        print("###################### start time=%f train task ############" % current_time)
        self_lables = algo.feed_labels_data_n(labels=self_lables, labels_attr=labels_attr, data_n=current_uvw)
        real_xyzuvwp = GetRealPhyInfo(current_time)
        real_uvwp = real_xyzuvwp[:, 3:7]
        self_lables = algo.feed_labels_data(labels=self_lables, labels_attr=labels_attr, data=real_uvwp)
        
        for i in range(len(self_lables)):
            feeds['label' + str(i)] = self_lables[i]

        for epoch in range(2):
            rslt = exe.run(main_program,
                        feed=feeds, 
                        fetch_list=fetchs)
            print("static epoch: " + str(epoch + 1), "loss: ", rslt[0])
        next_uvwp = rslt[1: ]
        next_uvwp = next_uvwp[0]
        next_uvwp = np.array(next_uvwp)
        current_uvw = next_uvwp[:, 0:3]

    end_time = time.time()

    print(f"\n{time_step - 3} epoch time: {end_time - begin_time} s")

if __name__ == '__main__':
    slove_static()