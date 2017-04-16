from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
import yaml
from sportvu.train import train


# Write a function like this called 'main'
def main(job_id, params):
    # print 'Anything printed here will end up in the output directory for job #%d'% job_id
    # print params
    # base hyperparams
    f_data_config = '../../../sportvu/data/config/rev3_1-hard-bmf-25x25.yaml'
    f_model_config = '../../../sportvu/model/config/conv2d-3layers-25x25-tuned.yaml'
    #
    data_config = yaml.load(open(f_data_config, 'rb'))
    model_config = yaml.load(open(f_model_config, 'rb'))
    model_name = os.path.basename(f_model_config).split('.')[0]
    data_name = os.path.basename(f_data_config).split('.')[0]
    exp_name = '%s-X-%s-%i' % (model_name, data_name, job_id)
    fold_index = 0
    init_lr = params['lr'][0]
    max_iter = params['max_iter'][0]
    best_acc_delay = 5000
    # modify model hyperparam
    c = params['crop_size_radius'][0] * 2 + 1
    data_config['extractor_config']['crop_size'] = [c, c]
    data_config['extractor_config'][
        'tfa_jitter_radius'] = params['tfa_jitter_radius'][0]
    data_config['extractor_config']['jitter'] = [
        params['jitter'][0], params['jitter'][0]]
    # modify data hyperparam
    model_config['model_config']['d1'] = c
    model_config['model_config']['d2'] = c
    c1, c2, c3 = params['channel_sizes']
    model_config['model_config']['conv_layers'] = [
        [5, 5, 3, c1], [5, 5, c1, c2], [5, 5, c2, c3]]
    model_config['model_config']['fc_layers'] = list(params['fc_sizes']) + [2]
    model_config['model_config']['keep_prob'] = params['keep_prob'][0]
    model_config['model_config']['bn'] = params['bn'][0]
    model_config['model_config']['pool'] = params['pool'][0]

    ret = 10
    # try:
    ret = train(data_config, model_config, exp_name,
                fold_index, init_lr, max_iter, best_acc_delay)
    # except:
    #     pass
    return {'main':float(ret)}
