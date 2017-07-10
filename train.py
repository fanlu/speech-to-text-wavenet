import sugartensor as tf
from data import SpeechCorpus
from data_ch import voca_size
from model import *

__author__ = 'namju.kim@kakaobrain.com'

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 16  # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# mfcc feature of audio
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
# target sentence label
labels = tf.split(data.label, tf.sg_gpus(), axis=0)
# sequence length except zero-padding
seq_len = []
for input_ in inputs:
  seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))


# parallel loss tower
@tf.sg_parallel
def get_loss(opt):
  # encode audio feature
  logit = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
  for i in tf.get_collection("regularization_losses"):
    print(i)
  print('--------------------')

  train_list = tf.trainable_variables()

  var_list = tf.global_variables()
  real_var_list = []
  for item in var_list:
    # print(item)
    if 'W' in item.name:
      real_var_list.append(item)

  loss = logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])
  # print(loss)
  # tf.add_to_collection("losses", loss)
  # losses = tf.get_collection("losses")
  # losses += tf.get_collection("regularization_losses")
  # for i in tf.get_collection("losses"):
  #   print(i.name)
  # print('++++++++++++++++++++')
  # total_loss = tf.add_n(losses, name='total_loss')
  # for item in real_var_list:
  #   loss += 0.03 * tf.nn.l2_loss(item)

  # for i in tf.get_collection("regularization_losses"):
  #   loss += 0.03 * i

  regular_loss = tf.sg_regularizer_loss(0.03)
  loss += regular_loss
  return loss


#
# train
#
tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=data.num_batch, max_ep=50)
