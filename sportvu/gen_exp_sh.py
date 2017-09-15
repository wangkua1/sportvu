import os

out_file = '/home/ethanf/gobi5/ethanf/projects/sportvu/sportvu/run_exp_ethanf.sh'

data_folder = ['data/config/rnn_baseline_loc-loc.yaml','data/config/rnn_baseline_vel-vel.yaml','data/config/rnn_baseline_loc-vel.yaml',
               'data/config/rnn_location_full_enc.yaml']

loss_folder = ['model/config/gauss_full','model/config/gauss_diag']
loss_functions = ['FullGaussNLL','DiagGaussNLL']
os.chdir('/home/ethanf/gobi5/ethanf/projects/sportvu/sportvu')




with open(out_file,'w') as f:
    f.write('#!/bin/bash\n')
    for data in data_folder:
        for i in range(2):
            loss = loss_functions[i]
            folder = loss_folder[i]
            files = os.listdir(loss_folder[i])
            for file in files:
                exp = 'python train-seq2seq.py 0 {} {}/{} {}'.format(data, folder, file, loss)
                f.write('srun --gres=gpu:1 -c 2 -l -p gpuc {} &\n'.format(exp))

