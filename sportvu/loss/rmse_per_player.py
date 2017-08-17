import tensorflow as tf

class RMSEPerPlayer(object):
	def __init__(self):
		super(RMSEPerPlayer, self).__init__()
	

	def reshape_per_player(self, inp):
		n,t,d = inp.shape
		return tf.reshape(inp,[n.value,t.value,d.value//2,2])
	def build_tf_loss(self, pred, y_):
		"""
		both pred, y_ shape (N, T, 2*#players)
		"""
		return tf.reduce_mean(tf.pow(tf.reduce_sum(tf.pow(self.reshape_per_player(pred) - self.reshape_per_player(y_), 2), axis=-1),.5))