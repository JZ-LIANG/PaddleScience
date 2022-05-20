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

import sys
sys.path.append("/work/somecode/Science/manual/PaddleScience")
import os
import paddle
import six
import time
import warnings
import paddle.compat as cpt
import paddlescience as psci
import numpy as np
import time
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
import paddle
from paddle.incubate.autograd import Hessian
from paddle.static import global_scope
from transform import program_transform
from transform import dead_code_elimination
from transform import fuse_shape_fill_constant
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.partitioner import Partitioner 
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context, set_default_distributed_context
from paddle.distributed.auto_parallel.utils import set_var_dist_attr 
import paddle.distributed.auto_parallel as auto
import argparse

paddle.enable_static()
paddle.seed(1234)
np.random.seed(1234)

os.environ['FLAGS_USE_STANDALONE_EXECUTOR'] = 'False'
warnings.warn("FLAGS_USE_STANDALONE_EXECUTOR is disabled.")

np.set_printoptions(
    suppress=True,
    precision=6,
    formatter={'float': '{:0.6f}'.format},
    threshold=np.inf,
    linewidth=1000)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def compile(program, loss_name=None):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program

def compile_and_convert_back_to_program(program=None,
                                   fetch_list=None,
                                   fetch_var_name='fetch',
                                   scope=None,
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

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope, paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    #ir_graph.draw(save_path='./', name='compiled_graph')
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    #paddle.static.save(final_program, "final")
    return final_program


# Analytical solution
def LaplaceRecSolution(x, y, k=1.0):
    if (k == 0.0):
        return x * y
    else:
        return np.cos(k * x) * np.cosh(k * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i] = LaplaceRecSolution(xy[i][0], xy[i][1])
    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]
    return [sol, bc_value]

def set_init_dist_attr(serial_main_prog):
    # set init dp attr    
    default_dist_context = get_default_distributed_context()
    _global_parallel_strategy = "dp"
    _global_process_mesh = auto.ProcessMesh([0, 1])
    x_tensor = serial_main_prog.global_block().var("x")
    bc_idx_tensor = serial_main_prog.global_block().var("bc_idx")
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

def p_norm_allreduce(prog, dist_context):
    block = prog.global_block()
    ops = block.ops
    for i , op in enumerate(ops):
        if op.type == 'reduce_p':
            if ops[i + 1].type == "sqrt_p":
                var_name = op.output_arg_names[0]

                from paddle.distributed.auto_parallel.process_group import new_process_group
                from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY
                from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute
                group_ranks = list(range(paddle.distributed.get_world_size()))
                dp_group = new_process_group(group_ranks)

                allreduce_op = block._insert_op(
                    i + 1,
                    type='c_allreduce_sum',
                    inputs={'X': [var_name]},
                    outputs={'Out': [var_name]},
                    attrs={
                        'ring_id': dp_group.id,
                        'use_calc_stream': True,
                        OP_ROLE_KEY: OpRole.Backward
                    })

                var = block.var(var_name)
                tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
                op_attr = OperatorDistributedAttribute()
                op_attr.process_mesh = tensor_dist_attr.process_mesh
                op_attr.set_output_dims_mapping(var.name, tensor_dist_attr.dims_mapping)
                op_attr.set_input_dims_mapping(var.name, tensor_dist_attr.dims_mapping)
                dist_context.set_op_dist_attr_for_program(op, op_attr)

    block._sync_with_cpp()


def get_dist_prog(serial_main_prog, serial_startup_prog, params_grads, args):

    # dist_main_program = paddle.static.Program()
    # dist_startup_program = paddle.static.Program()
    dist_context = DistributedContext(serial_main_prog, serial_startup_prog)

    # forward completion
    completer = Completer(dist_context)
    completer.complete_forward_annotation(serial_main_prog)
    set_default_distributed_context(dist_context)

    dist_context.block_state.parse_forward_blocks(serial_main_prog)
    # backward
    # completer.complete_backward_annotation(serial_main_prog)
    dist_context.block_state.parse_backward_blocks(serial_main_prog)
    dist_context.grads_params = dict()
    for p, g in params_grads:
        dist_context.grads_params[g.name] = p.name
    dist_context.synced_gradient = set()
    dist_context.global_process_mesh = auto.ProcessMesh([0, 1])

    # parititoner
    rank = paddle.distributed.get_rank()
    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
    serial_main_prog, serial_startup_prog, params_grads)
    assert set(dist_context.grads_params.keys()) == dist_context.synced_gradient

    # insert p_norm allreduce
    if args.norm_allreduce:
        p_norm_allreduce(dist_main_prog, dist_context)

    return dist_main_prog, dist_startup_prog, dist_params_grads

def data_parallel_split_data(geo, pdes):
    nranks = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    print("nranks: ", nranks)
    print("rank: ", rank)

    # original data
    space_domain = geo.get_space_domain()
    bc_index = geo.get_bc_index()
    bc_value = pdes.bc_value

    # split inner and bc
    inner_domain = []
    bc_domain = []
    for index in range(len(space_domain)):
        if index in bc_index:
            i = list(bc_index).index(index)
            value = bc_value[i]
            bc_domain.append([space_domain[index], value])
        else:
            inner_domain.append(space_domain[index])


    len_inner = len(inner_domain)
    len_bc = len(bc_domain)
    assert len_inner % nranks == 0
    assert len_bc % nranks == 0

    # split and concat
    local_inputs = []
    local_bc_idx = []
    local_bc_value = []
    inner_sz = len_inner // nranks
    bc_sz = len_bc // nranks
    bc_idx = [b[0] for b in bc_domain]
    bc_val = [b[1] for b in bc_domain]

    i = rank
    local_inputs.extend(inner_domain[inner_sz*i : inner_sz*(i+1)])
    local_inputs.extend(bc_idx[bc_sz*i : bc_sz*(i+1)])
    local_bc_idx.extend(range(inner_sz, inner_sz + bc_sz))
    local_bc_value.extend(bc_val[bc_sz*i : bc_sz*(i+1)])


    local_inputs = np.array(local_inputs)
    local_bc_idx = np.array(local_bc_idx)
    local_bc_value = np.array(local_bc_value)

    return local_inputs, local_bc_idx, local_bc_value

def main(args):
    # Geometry
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

    # PDE Laplace
    pdes = psci.pde.Laplace2D()

    # Discretization
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(10, 10))

    # bc value
    golden, bc_value = GenSolution(geo.get_space_domain(), geo.get_bc_index())
    pdes.set_bc_value(bc_value=bc_value)
    local_x, local_bc_idx, local_bc_v = data_parallel_split_data(geo, pdes)
    nranks = paddle.distributed.get_world_size()

    psci.visu.save_vtk(geo, golden, 'golden_laplace_2d')
    np.save('./golden_laplace_2d.npy', golden)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id)
    exe = paddle.static.Executor(place)

    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        inputs = paddle.static.data(
            name='x', shape=[geo.get_domain_size() // nranks, 2], dtype='float32')
        inputs.stop_gradient = False
        # Network
        net = psci.network.FCNetStatic(
            num_ins=2,
            num_outs=1,
            num_layers=5,
            hidden_size=20,
            dtype='float32',
            activation='tanh')

        outputs = net.nn_func(inputs)

        # eq_loss
        hes = Hessian(net.nn_func, inputs, is_batched=True)
        eq_loss = paddle.norm(hes[:, 0, 0] + hes[:, 1, 1], p=2)

        # bc_loss
        bc_index = paddle.static.data(name='bc_idx', shape=[36 // nranks], dtype='int32')
        bc_value = paddle.static.data(name='bc_v', shape=[36 // nranks, 1], dtype='float32')
        bc_u = paddle.index_select(outputs, bc_index)
        bc_diff = bc_u - bc_value
        bc_loss = paddle.norm(bc_diff, p=2)
        loss = eq_loss + bc_loss
        optimize_ops, params_grads = paddle.optimizer.Adam(learning_rate=0.001).minimize(loss)

    output_dir=str(args.output_dir)
    set_default_distributed_context(None)
    with open(output_dir + "/train_program.txt", "w+") as f:
        f.write(str(train_program))
    with open(output_dir + "/startup_program.txt", "w+") as f:
        f.write(str(startup_program))
    new_program = program_transform(train_program)
    dead_code_elimination(new_program)
    fuse_shape_fill_constant(new_program)


    with open(output_dir + "/transform_program.txt", "w+") as f:
        f.write(str(new_program))

    set_init_dist_attr(new_program)
    new_program, startup_program, dist_params_grads = get_dist_prog(new_program, startup_program, params_grads, args)
    set_default_distributed_context(None)
    with open(output_dir + "/dist_main_program.txt", "w+") as f:
        f.write(str(new_program))
    with open(output_dir + "/dist_startup_program.txt", "w+") as f:
        f.write(str(startup_program))

    num_epoch = args.num_epoch
    convert_back_to_program = True
    if convert_back_to_program:
        compiled_program = compile_and_convert_back_to_program(
            new_program, fetch_list=[loss.name])
    else:
        compiled_program = compile(new_program, loss.name)
    train_program = compile(train_program, loss.name)
    # with open(output_dir + "/dist_compiled_program.txt", "w+") as f:
    #     f.write(str(compiled_program))

    init_comm()
    exe.run(startup_program)
    begin = time.time()

    if os.getenv('FLAGS_use_cinn') == "1":
        for i in range(num_epoch):
            if i == 10:
                paddle.device.cuda.synchronize()
                begin = time.time()
                print("begin With CINN at ", begin)

            # loss_d = exe.run(compiled_program,
            #                  feed={
            #                      'x': geo.get_space_domain().astype(np.float32),
            #                      'bc_idx': geo.bc_index.astype(np.int32),
            #                      'bc_v': pdes.bc_value
            #                  },
            #                  fetch_list=[loss.name])
            loss_d = exe.run(compiled_program,
                            feed={
                                'x': local_x.astype(np.float32),
                                'bc_idx': local_bc_idx.astype(np.int32),
                                'bc_v': local_bc_v
                            },
                            fetch_list=[loss.name], 
                            use_program_cache=True)
            print('base num_epoch: ', i, '/', num_epoch, ' loss: ', loss_d[0][0])

        end = time.time()
        print("[With CINN] 2000 epoch(10~2010) time: ", end - begin, " s")
    else:
        for i in range(num_epoch):
            if i == 10:
                paddle.device.cuda.synchronize()
                begin = time.time()
                print("begin Without CINN at ", begin)

            loss_d, eq_loss_d, bc_loss_d = exe.run(
                train_program,
                feed={
                    'x': geo.get_space_domain().astype(np.float32),
                    'bc_idx': geo.bc_index.astype(np.int32),
                    'bc_v': pdes.bc_value
                },
                fetch_list=[loss.name, eq_loss.name, bc_loss.name])
            print('num_epoch: ', i, '/', num_epoch, ' loss: ', loss_d[0])

        end = time.time()
        print("[Base + Without CINN] 2000 epoch(10~2010) time: ", end - begin, " s")

    rslt = exe.run(train_program,
                feed={
                    'x': geo.get_space_domain().astype(np.float32),
                    'bc_idx': geo.bc_index.astype(np.int32),
                    'bc_v': pdes.bc_value
                },
                fetch_list=[outputs.name, ])[0]

    psci.visu.save_vtk(geo, rslt, 'rslt_laplace_2d')
    np.save('./rslt_laplace_2d.npy', rslt)

    # Calculate diff and l2 relative error
    diff = rslt - golden
    psci.visu.save_vtk(geo, diff, 'diff_laplace_2d')
    np.save('./diff_laplace_2d.npy', diff)
    root_square_error = np.linalg.norm(diff, ord=2)
    mean_square_error = root_square_error * root_square_error / geo.get_domain_size(
    )
    print('mean_sqeare_error: ', mean_square_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training logs and checkpoints will be written."
    )
    parser.add_argument(
        "--norm_allreduce",
        type=str2bool,
        nargs='?',
        const=False,
        help="insert allreduce op when calculate the loss")
    parser.add_argument(
        "--num_epoch",
        default=1,
        type=int,
        help="num_epoch.", )

    args = parser.parse_args()
    main(args)


