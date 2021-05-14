"""
Logging util
@Author: penhe@microsoft.com
"""

""" Utils for torch jit tracing customer operators/functions
"""
import os
import torch

def traceable(cls):
  class _Function(object):
    @staticmethod
    def apply(*args):
      if torch.onnx.is_in_onnx_export():
        return cls.forward(_Function, *args)
      else:
        return cls.apply(*args)
    
    @staticmethod
    def save_for_backward(*args):
      pass
  return _Function


class TraceMode():
  """ Trace context used when tracing modules contains customer operators/Functions
  """
  def __enter__(self):
    os.environ['JIT_TRACE'] = 'True'
    return self

  def __exit__(self, exp_value, exp_type, trace):
    del os.environ['JIT_TRACE']

