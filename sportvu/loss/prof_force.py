"OBSOLETE...just keeping it for potential future references"

# import tensorflow as tf
# from ..loss import *

# class ProfForce(object):
#     def __init__(self):
#         super(ProfForce, self).__init__()
#         """
#         specify some configs    
#             for the discriminator
#             for the complimentary loss
#             for weights between losses
#         """
#         ### WARNING: hardcoding some config for now
#         import yaml
#         self.config = yaml.load(open('loss/config/prof_force_ex.yaml', 'rb'))
#         ###
#         model_config = self.config['model_config']
#         self.tf_disc_step = tf.Variable(0, name='disc_step')
#         self.tf_disc_learning_rate = tf.placeholder(tf.float32, [], name='disc_lr') 
#         with tf.variable_scope("loss_model") as vs:
#             self.disc_net = eval(model_config['class_name'])(model_config['model_config'])
#             self.disc_net.build()
#             self.disc_net_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]


#     def build_tf_loss(self, pred, y_):
#         """
#         use any model,
#         use only the last timestep with a binary output,
#         ....
#         what I need 
#         [
#         1. a model, an optimizer for update discriminator
#         2. another loss (like RMSE or whatever)
#         ]
#         """
#         self.disc_loss = ....
#         self.disc_step = optimize_loss(self.disc_loss, self.tf_disc_step, self.tf_disc_learning_rate,
#                                optimizer=lambda lr: tf.train.RMSPropOptimizer(lr),
#                                clip_gradients=0.01, variables=self.disc_net_variables)


        
#     def update_loss(self, sess, loss_update_kwargs):
#         """
#         (called after train_step)
#         update discriminator here
#         """
#         return {'disc_loss':sess.run(self.disc_step, feed_dict=feed_dict)}