import numpy as np
from warnings import warn

from smaug.core import node_pb2, types_pb2
from smaug.python import global_vars
from smaug.python.tensor import Tensor
from smaug.python.ops import activation_ops, array_ops, common

def to_padding_type(padding):
  if padding == "same":
    return types_pb2.SamePadding
  elif padding == "valid":
    return types_pb2.ValidPadding
  else:
    return types_pb2.UnknownPadding

# TODO attack simulation only: OPAtt (operator attack) helpers. The attack runs
# at the operator-creating stage, so whatever it rewrites here is what gets
# serialized and executed by the backend. See global_vars for the spec list.

def _opatt_active(spec):
  """True if OPAtt is enabled and configured with this exact spec."""
  return (global_vars.enable_operator_attack
          and global_vars.operator_attack_spec == spec)

def _opatt_substitute_activation(activation, op_name):
  """Operator substitution: swap the fused activation for a different one."""
  substitutions = {
      "relu-identity": None,
      "relu-sigmoid": "sigmoid",
      "relu-tanh": "tanh",
  }
  if activation is None or not global_vars.enable_operator_attack:
    return activation
  spec = global_vars.operator_attack_spec
  if spec not in substitutions:
    return activation
  replacement = substitutions[spec]
  print("[OPAtt/%s] %s: activation %s -> %s" %
        (spec, op_name, activation, replacement or "identity"))
  return replacement

def _opatt_corrupt_weights(weight_tensor, op_name, spec):
  """Parameter corruption: zero a random fraction of the weight elements."""
  if not _opatt_active(spec) or weight_tensor.tensor_data is None:
    return
  data = weight_tensor.tensor_data
  rng = np.random.RandomState(global_vars.operator_attack_seed)
  mask = rng.random_sample(size=data.shape) < global_vars.operator_attack_ratio
  data[mask] = 0
  print("[OPAtt/%s] %s: zeroed %d/%d (%.1f%%) weight elements." %
        (spec, op_name, int(mask.sum()), data.size,
         100.0 * int(mask.sum()) / data.size))

def convolution(
    input_tensor, filter_tensor, stride, padding, activation=None,
    activation_params=None, name="conv"):
  """Compute a 3D Convolution given 4D `input_tensor` and `filter_tensor`.

  Args:
    input_tensor: A 4D `Tensor`.
    filter_tensor: A 4D `Tensor`.
    stride: A list of two integers: [row_stride, col_stride].
    padding: A string from: `same`, `valid`. The zero padding options.
    activation: A string representing the activation function (optional).
    activation_params: kwargs for the activation function (optional).
    name: Operator name (optional).
  """
  def compute_output_dim(input_dim, weight_dim, stride, padding):
    pad = 0
    if to_padding_type(padding) == types_pb2.SamePadding:
      pad = weight_dim - 1
    return (input_dim - weight_dim + pad) // stride + 1

  activation = _opatt_substitute_activation(activation, name)
  _opatt_corrupt_weights(filter_tensor, name, "conv-weight")

  input_tensor, filter_tensor = array_ops.check_and_add_layout_transform(
      name=name, op=types_pb2.Convolution3d,
      input_tensors=[input_tensor, filter_tensor])

  row_idx = 2 if input_tensor.shape.layout == types_pb2.NCHW else 1
  col_idx = 3 if input_tensor.shape.layout == types_pb2.NCHW else 2
  chan_idx = 1 if input_tensor.shape.layout == types_pb2.NCHW else 3
  assert input_tensor.dims(chan_idx) == filter_tensor.dims(chan_idx), (
      "The weights must have the same number of channels as the inputs.")
  output_rows = compute_output_dim(input_tensor.shape.dims[row_idx],
                                   filter_tensor.shape.dims[row_idx], stride[0],
                                   padding)
  output_cols = compute_output_dim(input_tensor.shape.dims[col_idx],
                                   filter_tensor.shape.dims[col_idx], stride[1],
                                   padding)
  output_layout = input_tensor.shape.layout
  if output_layout == types_pb2.NCHW:
    output_tensor_dims = [
        input_tensor.shape.dims[0], filter_tensor.shape.dims[0], output_rows,
        output_cols
    ]
  elif output_layout == types_pb2.NHWC:
    output_tensor_dims = [
        input_tensor.shape.dims[0], output_rows, output_cols,
        filter_tensor.shape.dims[0]
    ]
  else:
    assert False, "Unsupported output layout!"
  params = node_pb2.Params()
  params.conv_params.padding = to_padding_type(padding)
  params.conv_params.stride.extend(stride)
  if activation is not None:
    params.act_params.CopyFrom(
        activation_ops.to_proto(activation, activation_params))
  return common.add_node(
      name=name, op=types_pb2.Convolution3d,
      input_tensors=[input_tensor, filter_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=output_layout, params=params)[0]

def batch_norm(
    input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor,
    activation=None, activation_params=None, name="batch_norm"):
  """Perform batch normalization.

  Args:
    input_tensor: A 2D or 4D `Tensor`.
    mean_tensor: Mean parameter.
    var_tensor: Variance parameter. For performance reasons, this is
      precomputed as 1/sqrt(variance + eps).
    gamma_tensor: Gamma parameter.
    beta_tensor: Beta parameter.
    activation/activation_params: Activation function to use (optional).
    name: Operator name (optional).
  """
  activation = _opatt_substitute_activation(activation, name)
  assert (len(mean_tensor.shape.dims) == 2 and len(var_tensor.shape.dims) == 2
          and len(gamma_tensor.shape.dims) == 2
          and len(beta_tensor.shape.dims) == 2)
  # If the batch norm is after a FC layer, then the input/output tensors should
  # be in NC. Otherwise, the batch norm is after a convolution layer, and we
  # check backend_layouts for expected input/output layouts and do layout
  # transformation if needed.
  post_fc = False
  if len(input_tensor.shape.dims) == 2:
    post_fc = True

  if not post_fc:
    input_tensor = array_ops.check_and_add_layout_transform(
        name=name, op=types_pb2.BatchNorm, input_tensors=[input_tensor])[0]

  output_layout = types_pb2.UnknownLayout
  output_layout = types_pb2.NC if post_fc else input_tensor.shape.layout
  params = node_pb2.Params()
  if activation is not None:
    params.act_params.CopyFrom(
        activation_ops.to_proto(activation, activation_params))
  return common.add_node(
      name=name, op=types_pb2.BatchNorm, input_tensors=[
          input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor
      ], output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=output_layout, params=params)[0]

def max_pool(input_tensor, pool_size, stride, name="max_pool"):
  """Compute max pooling.

  Args:
    input_tensor: A 4D `Tensor`.
    pool_size: A list of two integers: [pool_rows, pool_cols].
    stride: A list of two integers: [row_stride, col_stride].
    name: Operator name (optional).
  """
  def compute_output_dim(input_dim, pool_size, stride):
    return (input_dim - pool_size) // stride + 1

  op_type = types_pb2.MaxPooling
  if _opatt_active("maxpool-avgpool"):
    op_type = types_pb2.AveragePooling
    print("[OPAtt/maxpool-avgpool] %s: MaxPooling -> AveragePooling" % name)

  input_tensor = array_ops.check_and_add_layout_transform(
      name=name, op=op_type, input_tensors=[input_tensor])[0]

  row_idx = 2 if input_tensor.shape.layout == types_pb2.NCHW else 1
  col_idx = 3 if input_tensor.shape.layout == types_pb2.NCHW else 2
  output_rows = compute_output_dim(input_tensor.shape.dims[row_idx],
                                   pool_size[0], stride[0])
  output_cols = compute_output_dim(input_tensor.shape.dims[col_idx],
                                   pool_size[1], stride[1])
  output_layout = input_tensor.shape.layout
  if output_layout == types_pb2.NCHW:
    output_tensor_dims = [
        input_tensor.shape.dims[0], input_tensor.shape.dims[1], output_rows,
        output_cols
    ]
  else:
    output_tensor_dims = [
        input_tensor.shape.dims[0], output_rows, output_cols,
        input_tensor.shape.dims[3]
    ]
  params = node_pb2.Params()
  params.pool_params.stride.extend(stride)
  params.pool_params.pool_size.extend(pool_size)
  return common.add_node(
      name=name, op=op_type, input_tensors=[input_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=output_layout, params=params)[0]

def mat_mul(
    input_tensor, weight_tensor, activation=None, activation_params=None,
    name="mat_mul"):
  """Compute a matrix multiplication for `input_tensor` and `weight_tensor`.

  Args:
    input_tensor: A 2D `Tensor`. Shaped as `NC`, where `N` is batch size and `C`
      is number of channels.
    weight_tensor: A 2D `Tensor`. Shaped as `NC` or `CN`, where `N` is number of
      neurons and `C` is the same as in `input_tensor`.
    activation/activation_params: Activation function to use (optional).
    name: Operator name (optional).
  """
  activation = _opatt_substitute_activation(activation, name)
  _opatt_corrupt_weights(weight_tensor, name, "gemm-weight")

  input_tensor, weight_tensor = array_ops.check_and_add_layout_transform(
      name=name, op=types_pb2.InnerProduct,
      input_tensors=[input_tensor, weight_tensor])

  weight_layout = weight_tensor.shape.layout
  actIdx = 1 if weight_layout == types_pb2.NC else 0
  neuronIdx = 0 if weight_layout == types_pb2.NC else 1
  assert (len(input_tensor.shape.dims) == 2
          and len(weight_tensor.shape.dims) == 2
          and input_tensor.shape.dims[1] == weight_tensor.shape.dims[actIdx])
  output_tensor_dims = [
      input_tensor.shape.dims[0], weight_tensor.shape.dims[neuronIdx]
  ]
  params = node_pb2.Params()
  if activation is not None:
    params.act_params.CopyFrom(
        activation_ops.to_proto(activation, activation_params))
  return common.add_node(
      name=name, op=types_pb2.InnerProduct,
      input_tensors=[input_tensor, weight_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=types_pb2.NC, params=params)[0]
