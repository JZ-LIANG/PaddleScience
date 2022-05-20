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

import six
import os
import warnings
import numpy as np
import time
import paddlescience as psci
import paddle
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.static import global_scope
from paddle.incubate.autograd.primx import prim2orig
from paddle.incubate.autograd.utils import enable_prim, prim_enabled
# from paddle.fluid.incubate.ad_transform.primx import prim2orig, enable_prim, prim_enabled
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.partitioner import Partitioner 
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.utils import set_var_dist_attr 
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context, set_default_distributed_context
from gradient_merge_pass import parse_program

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
enable_prim()

# define start time and time step
start_time = 100
time_step = 1

def debug_program(main_program, path):
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    path += str(gpu_id)
    with open(path, "w+") as f:
        f.write(str(main_program))

def apply_gradient_merge_pass(main_program, startup_program, param_grads, k_step = 16, allreduce_in_update = True):
    with paddle.static.program_guard(main_program, startup_program):
        parse_program(main_program, startup_program, param_grads, k_steps, False, True)
        main_program._sync_with_cpp()
        debug_program(main_program, "./gm_program.txt.")
        debug_program(startup_program, "./gm_startup_program.txt.")

def set_init_dist_attr(serial_main_prog):

    # set init dp attr    
    default_dist_context = get_default_distributed_context()
    _global_parallel_strategy = "dp"
    _global_process_mesh = auto.ProcessMesh(list(range(paddle.distributed.get_world_size())))
    x_tensor = serial_main_prog.global_block().var("input0")
    bc_idx_tensor = serial_main_prog.global_block().var("label0")
    tensor_dist_attr = set_var_dist_attr(default_dist_context, x_tensor, [-1, -1], _global_process_mesh, mark_annotated=True)
    tensor_dist_attr = set_var_dist_attr(default_dist_context, bc_idx_tensor, [-1], _global_process_mesh, mark_annotated=True)

def init_comm():
    from paddle.distributed.auto_parallel.process_group import get_all_process_groups
    all_process_groups = get_all_process_groups()
    rank = paddle.distributed.get_rank()
    for process_group in all_process_groups:
        if rank not in process_group.ranks:
            continue
        process_group.instantiate()

def get_dist_prog(serial_main_prog, serial_startup_prog, params_grads):
    print("start auto parallel transform, wait ...")
    start_time_ = time.time()
    set_init_dist_attr(serial_main_prog)
    dist_context = DistributedContext(serial_main_prog, serial_startup_prog)

    # forward completion
    completer = Completer(dist_context)
    completer.complete_prim_annotation(serial_main_prog)
    set_default_distributed_context(dist_context)

    dist_context.block_state.parse_forward_blocks(serial_main_prog)
    # backward
    # completer.complete_backward_annotation(serial_main_prog)
    dist_context.block_state.parse_backward_blocks(serial_main_prog)
    dist_context.grads_params = dict()
    for p, g in params_grads:
        dist_context.grads_params[g.name] = p.name
        print(p.name, g.name)
    dist_context.synced_gradient = set()
    dist_context.data_parallel_group = list(range(paddle.distributed.get_world_size()))
    
    # parititoner
    rank = paddle.distributed.get_rank()
    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
    serial_main_prog, serial_startup_prog, params_grads)
    print("dist_context.synced_gradient: ", dist_context.synced_gradient)
    assert set(dist_context.grads_params.keys()) == dist_context.synced_gradient

    init_comm()
    print("auto parallel transform finish in {} sec.".format(time.time() - start_time_))
    return dist_main_prog, dist_startup_prog, dist_params_grads


def l2_norm_square(x, scale=None):
    if scale is None:
        l2_norm = paddle.norm(x, p=2)
    else:
        l2_norm = paddle.norm(x * scale, p=2) / scale
    return l2_norm * l2_norm


# load real data
def GetRealPhyInfo(time, need_cord=False, need_physic=False):
    real_data = np.load("./flow_unsteady_re200/flow_re200_" + str(time) +
                        "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    if need_cord is False and need_physic is False:
        print("Error: you need to get cord or get physic infomation")
        exit()
    elif need_cord is True and need_physic is True:
        return real_data
    elif need_cord is True and need_physic is False:
        return real_data[:, 0:3]
    elif need_cord is False and need_physic is True:
        return real_data[:, 3:7]
    else:
        pass


# get init physic infomation
def GenInitPhyInfo(xyz):
    uvw = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvw[i][0] = 1.0
    return uvw


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
        program = executor._prune_program(program, feed, fetch_list,
                                          optimize_ops)
        feed = executor._update_feed(program, feed)

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = _compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope,
                              paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    #ir_graph.draw(save_path='./', name='compiled_graph')
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    #paddle.static.save(final_program, "final")
    return final_program


def init_algo():

    cc = (0.0, 0.0)
    cr = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -2),
        extent=(25, 8, 2),
        circle_center=cc,
        circle_radius=cr)

    geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
    geo.add_boundary(
        name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
    geo.add_boundary(
        name="circle",
        criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4
    )

    # discretize geometry
    geo_disc = geo.discretize(npoints=200000, method="sampling")
    # the real_cord need to be added in geo_disc
    real_cord = GetRealPhyInfo(start_time, need_cord=True)
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
        time_method="implicit", time_step=time_step, geo_disc=geo_disc)

    # Network
    net = psci.network.FCNet(
        num_ins=3,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        activation='tanh')

    # Loss
    loss = psci.loss.L2(p=2)

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    return algo, pde_disc


def compute_eq_loss(inputs, outputs, labels_var):
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    u = outputs[:, 0]
    v = outputs[:, 1]
    w = outputs[:, 2]
    p = outputs[:, 3]
    u_n = labels_var[0]
    v_n = labels_var[1]
    w_n = labels_var[2]
    jac0, = paddle.static.gradients([u], [inputs])  # du/dx, du/dy, du/dz
    jac1, = paddle.static.gradients([v], [inputs])  # dv/dx, dv/dy, dv/dz
    jac2, = paddle.static.gradients([w], [inputs])  # dw/dx, dw/dy, dw/dz
    jac3, = paddle.static.gradients([p], [inputs])  # dp/dx, dp/dy, dp/dz
    hes0, = paddle.static.gradients(
        [jac0[:, 0]], [inputs])  # du*du/dx*dx, du*du/dx*dy, du*du/dx*dz
    hes1, = paddle.static.gradients(
        [jac0[:, 1]], [inputs])  # du*du/dy*dx, du*du/dy*dy, du*du/dy*dz
    hes2, = paddle.static.gradients(
        [jac0[:, 2]], [inputs])  # du*du/dz*dx, du*du/dz*dy, du*du/dz*dz
    hes3, = paddle.static.gradients(
        [jac1[:, 0]], [inputs])  # dv*dv/dx*dx, dv*dv/dx*dy, dv*dv/dx*dz
    hes4, = paddle.static.gradients(
        [jac1[:, 1]], [inputs])  # dv*dv/dy*dx, dv*dv/dy*dy, dv*dv/dy*dz
    hes5, = paddle.static.gradients(
        [jac1[:, 2]], [inputs])  # dv*dv/dz*dx, dv*dv/dz*dy, dv*dv/dz*dz
    hes6, = paddle.static.gradients(
        [jac2[:, 0]], [inputs])  # dw*dw/dx*dx, dw*dw/dx*dy, dw*dw/dx*dz
    hes7, = paddle.static.gradients(
        [jac2[:, 1]], [inputs])  # dw*dw/dy*dx, dw*dw/dy*dy, dw*dw/dy*dz
    hes8, = paddle.static.gradients(
        [jac2[:, 2]], [inputs])  # dw*dw/dz*dx, dw*dw/dz*dy, dw*dw/dz*dz

    nu = 0.01
    rho = 1.0
    dt = 1.0
    continuty = jac0[:, 0] + jac1[:, 1] + jac2[:, 2]
    # + u / dt - u_n / dt
    momentum_x = u / dt - u_n / dt + u * jac0[:, 0] + v * jac0[:, 1] + w * jac0[:, 2] - \
                nu / rho * hes0[:, 0] - nu / rho * hes1[:, 1] - nu / rho * hes2[:, 2] + \
                1.0 / rho * jac3[:, 0]
    momentum_y = v / dt - v_n / dt + u * jac1[:, 0] + v * jac1[:, 1] + w * jac1[:, 2] - \
                nu / rho * hes3[:, 0] - nu / rho * hes4[:, 1] - nu / rho * hes5[:, 2] + \
                1.0 / rho * jac3[:, 1]
    momentum_z = w / dt - w_n / dt + u * jac2[:, 0] + v * jac2[:, 1] + w * jac2[:, 2] - \
                nu / rho * hes6[:, 0] - nu / rho * hes7[:, 1] - nu / rho * hes8[:, 2] + \
                1.0 / rho * jac3[:, 2]

    rhs = 0
    wgt = np.sqrt(0.01)

    eq_loss = l2_norm_square((continuty - rhs)*wgt) + \
            l2_norm_square((momentum_x - rhs)*wgt) + \
            l2_norm_square((momentum_y - rhs)*wgt) + \
            l2_norm_square((momentum_z - rhs)*wgt)
    return eq_loss


def slove_static():
    algo, pde_disc = init_algo()

    # create inputs/labels and its attributes
    inputs, inputs_attr = algo.create_inputs(pde_disc)
    labels, labels_attr = algo.create_labels(pde_disc)

    # distributed info 
    nranks = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()

        # （lbsz, start_offset, end_offset(not include)）
    input_partition_meta = []
    for i in range(len(inputs)):
        gbsz = inputs[i].shape[0]
        lbsz = gbsz // nranks
        # last rank would contain more data
        start_idx = rank * lbsz
        end_idx = (rank + 1) * lbsz
        if rank == nranks - 1:
            lbsz += gbsz % nranks
            end_idx += gbsz % nranks
        input_partition_meta.append((lbsz, start_idx, end_idx))
    
    label_partition_meta = []
    for gbsz in [192093, 192093, 192093, 3415, 3415, 3415, 3415,3415, 3415, 3415]:
        lbsz = gbsz // nranks
        # last rank would contain more data
        start_idx = rank * lbsz
        end_idx = (rank + 1) * lbsz
        if rank == nranks - 1:
            lbsz += gbsz % nranks
            end_idx += gbsz % nranks
        label_partition_meta.append((lbsz, start_idx, end_idx))   
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        algo.net.make_network_static()
        inputs_var = []
        labels_var = []
        outputs_var = []

        # inputs
        for i in range(len(inputs)):
            #inputs
            # data parallel partition data 
            shape_ = list(inputs[i].shape)
            shape_[0] = input_partition_meta[i][0]
            input = paddle.static.data(
                name='input' + str(i), 
                shape=shape_, 
                dtype='float32')
            input.stop_gradient = False
            inputs_var.append(input)

        # labels
        for i in range(len(labels)):
            #labels
            # data parallel partition data 
            # # Hard code here for label shape. Shape may change when random seed changed 
            # if i in [0, 1, 2]:
            #     shape = (37174, )
            # else:
            #     shape = (3415, )
            shape = (label_partition_meta[i][0],)
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
        name2index = {'u': 0, 'v': 1, 'w': 2, 'p': 3}
        bc_loss = 0.0
        name_list = []
        for i, name_b in enumerate(inputs_attr["bc"].keys()):
            # from outputs_var[1] to outputs_var[3]
            out_el = outputs_var[i + 1]
            for j in range(len(pde_disc.bc[name_b])):
                rhs_b = labels_attr["bc"][name_b][j]["rhs"]
                wgt_b = labels_attr["bc"][name_b][j]["weight"]
                index = name2index.get(pde_disc.bc[name_b][j].name)

                # NOTE(lml): The new automatic differentiation mechanism splits the norm operator into more detailed operators, and the value range of the data used changes. We manually multiply by 10000 before calculating L2 norm, and then divide by 10000 to avoid the intermediate result from crossing the representation range of float32.
                bc_loss += l2_norm_square(
                    (out_el[:, index] - rhs_b) * np.sqrt(wgt_b), 10000)

        # inputs_var[0] eq loss
        output_var_0_eq_loss = compute_eq_loss(inputs_var[0], outputs_var[0],
                                               labels_var[0:3])

        # inputs_var[4] eq loss
        input_i = inputs_var[4]
        out_i = outputs_var[4]
        output_var_4_eq_loss = compute_eq_loss(inputs_var[4], outputs_var[4],
                                               labels_var[7:10])
        # data_loss
        data_loss = l2_norm_square(outputs_var[4][:, 0]-labels_var[3]) + \
                    l2_norm_square(outputs_var[4][:, 1]-labels_var[4]) + \
                    l2_norm_square(outputs_var[4][:, 2]-labels_var[5]) + \
                    l2_norm_square(outputs_var[4][:, 3]-labels_var[6])

        # total_loss
        total_loss = paddle.sqrt(bc_loss + output_var_0_eq_loss +
                                 output_var_4_eq_loss + data_loss)
        opt_ops, param_grads = paddle.optimizer.Adam(0.001).minimize(total_loss)
        debug_program(main_program, "./prim_program.txt.")

        if prim_enabled():
            if nranks > 1:
                main_program, startup_program, dist_params_grads = get_dist_prog(main_program, startup_program, param_grads)
                debug_program(main_program, "./auto_parallel_program.txt.")
            with paddle.static.program_guard(main_program, startup_program):
                prim2orig(main_program.block(0))
        debug_program(main_program, "./orign_program.txt.")

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id)
    exe = paddle.static.Executor(place)

    feeds = dict()
    for i in range(len(inputs)):
        # data parallel partition data
        if nranks > 1:
            start = input_partition_meta[i][1]
            end = input_partition_meta[i][2]
            feeds['input' + str(i)] = inputs[i][start: end]            
        else:
            feeds['input' + str(i)] = inputs[i]

    fetches = [total_loss.name]
    for var in outputs_var:
        fetches.append(var.name)

    main_program = compile_and_convert_back_to_program(
        main_program, feed=feeds, fetch_list=fetches, use_prune=True)
    debug_program(main_program, "./compiled_converted_program.txt.")

    # gradient merge
    # apply_gradient_merge_pass(main_program, startup_program, param_grads, k_step = 16, allreduce_in_update = True)

    exe.run(startup_program)
    # num_epoch in train
    train_epoch = 150

    # Solver time: (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    num_time_step = 1
    current_interior = np.zeros(
        (len(pde_disc.geometry.interior), 3)).astype(np.float32)
    current_user = GetRealPhyInfo(start_time, need_physic=True)[:, 0:3]
    for i in range(num_time_step):
        next_time = start_time + (i + 1) * time_step
        print("############# train next time=%f train task ############" %
              next_time)
        self_lables = algo.feed_data_interior_cur(labels, labels_attr,
                                                  current_interior)
        self_lables = algo.feed_data_user_cur(self_lables, labels_attr,
                                              current_user)
        self_lables = algo.feed_data_user_next(
            self_lables,
            labels_attr,
            GetRealPhyInfo(
                next_time, need_physic=True))
        for j in range(len(self_lables)):
            if nranks > 1:
                start = label_partition_meta[j][1]
                end = label_partition_meta[j][2]
                feeds['label' + str(j)] = self_lables[j][start: end]            
            else:    
                feeds['label' + str(j)] = self_lables[j]
            # feeds['label' + str(j)] = self_lables[j]

        for k in range(train_epoch):
            if  k == 50 :
                start_time__ = time.time()
            if k == 100:
                duration = time.time() - start_time__
                print("avg time from 50 - 100 epoch is {}".format(duration / 50.0))
            out = exe.run(main_program, feed=feeds, fetch_list=fetches)
            if k == 101:
                print("avg time from 50 - 100 epoch is {}".format(duration / 50.0))
                break

            print("autograd epoch: " + str(k + 1), "    loss:", out[0])
        next_uvwp = out[1:]
        # # Save vtk
        # file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(next_time)
        # psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)

        # next_info -> current_info
        next_interior = np.array(next_uvwp[0])
        next_user = np.array(next_uvwp[-1])
        current_interior = next_interior[:, 0:3]
        current_user = next_user[:, 0:3]


if __name__ == '__main__':
    slove_static()
