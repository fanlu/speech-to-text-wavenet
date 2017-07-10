import sugartensor as tf

num_blocks = 3  # dilated blocks
num_dim = 128  # latent dimension


#
# logit calculating graph using atrous convolution
#
def get_logit(x, voca_size):
  regularizer = None
  # residual block
  def res_block(tensor, size, rate, block, dim=num_dim):

    with tf.sg_context(name='block_%d_%d' % (block, rate)):
      # filter convolution
      conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter', regularizer='l2')

      # gate convolution
      conv_gate = tensor.sg_aconv1d(size=size, rate=rate, act='sigmoid', bn=True, name='conv_gate', regularizer=regularizer)

      # output by gate multiplying
      out = conv_filter * conv_gate

      # final output
      out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out', regularizer=regularizer)

      # residual and skip output
      return out + tensor, out

  # expand dimension
  with tf.sg_context(name='front'):
    z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in', regularizer=regularizer)

  # dilated conv block loop
  skip = 0  # skip connections
  for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
      z, s = res_block(z, size=7, rate=r, block=i)
      skip += s

  # final logit layers
  with tf.sg_context(name='logit'):
    logit = (skip
             .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1', regularizer=regularizer)
             .sg_conv1d(size=1, dim=voca_size, name='conv_2', regularizer=regularizer))

  return logit
