import numpy as np

def dump_nparray(array, filename):
  #array_file = open(filename, 'w')
  array_file = open(filename, 'w')
  np.uint32(array.ndim).tofile(array_file)
  #for d in xrange(array.ndim):
  for d in range(array.ndim):
    np.uint32(array.shape[d]).tofile(array_file)
  array.tofile(array_file)
  array_file.close()


def load_nparray(filename, array_dtype):
  array_file = open(filename, 'r')
  n_dim = np.fromfile(array_file, dtype = np.uint32, count = 1)[0]
  shape = []
  #for d in xrange(n_dim):
  for d in range(n_dim):
    shape.append(np.fromfile(array_file, dtype = np.uint32, count = 1)[0])
  array_data = np.fromfile(array_file, dtype = array_dtype)
  array_file.close()
  return np.reshape(array_data, shape)