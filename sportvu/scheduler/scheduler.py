import numpy as np

class Scheduler(object):
    def __init__(self, config):
        self._schedulers = {}
        self.max_iters = config['max_iters']
        for k_output in config['schedulers']:
            self._schedulers[k_output] = eval(config['schedulers'][k_output]['class'])(config['schedulers'][k_output]['kwargs'], self.max_iters)

        self.curr_iter = 0
    def update_iteration(self, iter=None):
        if iter:
            self.curr_iter = iter
        else:
            self.curr_iter += 1
    def get(self, k_output):
        return self._schedulers[k_output].get(self.curr_iter)

class ConstantScheduler(object):
    def __init__(self, config, max_iters):
        self.config = config
        self.max_iters = max_iters
        self.type = config['type']
        self.init_value = config['init_value']
        self.value = self.init_value
        if 'clamp' in config:
            self.clamp = config['clamp']
        else:
            self.clamp = None

    def get(self, iter):
        return self.value

    def cast(self, value):
        if self.clamp:
            value = min(max(value,self.clamp[0]), self.clamp[1])
        if self.type == 'float':
            return value
        elif self.type == 'int':
            return int(value)
        else:
            raise ValueError("Type not implemented")

class ConstantStepMultiplyScheduler(ConstantScheduler):
    def __init__(self, config, max_iters):
        super(ConstantStepMultiplyScheduler,self).__init__(config,max_iters)
        self.every_N_step =  config['every_N_step']
        self.factor = config['factor']
        self.last_time_update_iter = 0

    def get(self, iter):
        if iter % self.every_N_step ==0 and iter > self.last_time_update_iter:
            self.last_time_update_iter = iter
            self.value *= self.factor
        return self.cast(self.value)


class ConstantStepIncrementScheduler(ConstantScheduler):
    def __init__(self, config, max_iters):
        super(ConstantStepIncrementScheduler, self).__init__(config, max_iters)
        self.every_N_step =  config['every_N_step']
        self.increment = config['increment']
        self.last_time_update_iter = 0

    def get(self, iter):
        if iter % self.every_N_step ==0 and iter > self.last_time_update_iter:
            self.last_time_update_iter = iter
            self.value -= self.increment
        return self.cast(self.value)


# class LinearScheduler(ConstantScheduler):
#     def __init__(self, config, max_iters):
#         super(LinearScheduler, self).__init__(config, max_iters)
#
#
#     def get(self, iter):
#         if iter % self.every_N_step == 0 and iter > self.last_time_update_iter:
#             self.last_time_update_iter = iter
#             self.value -= self.increment
#         if self.type == 'float':
#             return self.value
#         elif self.type == 'int':
#             return int(self.value)
#         else:
#             raise ValueError("Type not implemented")