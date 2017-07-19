from __future__ import print_function, division
import os
import argparse
import time
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import tensorboard_logger
from sportvu.data.dataset import SeqDataset
from sportvu.data.extractor import  EthanSeqExtractor
from sportvu.vis.Event import Event, EventException


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)' )
parser.add_argument('--iter', type=int, default=300000, help='number of training iterations')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CPU training')
parser.add_argument('--seed', type=int, default=1, help='seed for reproducible experiments')
parser.add_argument('--val-interval', type=int, default=1000, help='Interval between validation steps')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr-step', type=int, default=50000, help='Decay lr after step iterations')
parser.add_argument('--lr-decay', type=float, default=0.5, help='Decay lr by decay factor')
parser.add_argument('--log-dir', default='auto', help='Folder to save log into (tensorboard logger), empty strong to not save, auto for automatically generated path')
parser.add_argument('--save-file', default='model.pkl', help='File name to save trained model, saves under log dir')
parser.add_argument('--data-config', default='/home/ethanf/projects/sportvu/sportvu/data/config/ethanf_seq.yaml', help='yaml with data configuration')


min_val_loss = np.inf
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.log = bool(args.log_dir)
if args.log_dir is 'auto':
    if os.environ['HOME'] == '/home/ethanf':  ## ethan local
        args.log_dir = os.path.join('/home/ethanf/gobi5/ethanf/log/nba/rnn',time.strftime("%Y.%m.%d_%H:%M"))
    elif os.environ['HOME'] == '/u/ethanf':  ## ethan guppy
        args.log_dir = os.path.join('/ais/gobi5/ethanf/log/nba/rnn',time.strftime("%Y.%m.%d_%H:%M"))
    else:
        assert(1==0)
# if not os.path.isdir(args.log_dir):
#     os.mkdir(args.log_dir)
# if args.save_file:
#     args.save_file = os.path.join(args.log_dir,args.save_file)
#
# if args.log:
#     tensorboard_logger.configure(args.log_dir, flush_secs=2)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



class LSTM_NBA_AR(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_NBA_AR, self).__init__()
        self.hidden_dim = hidden_dim
        self.location_dim = 11*2
        self.state_embeddings = nn.Linear(self.location_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2location_mean = nn.Linear(hidden_dim, self.location_dim)
        self.hidden2location_std_sq = nn.Linear(hidden_dim, self.location_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, seq):
        embeds = self.state_embeddings(seq)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(seq), 1, -1), self.hidden)
        mean = self.hidden2location_mean(lstm_out.view(len(seq), -1))
        std = self.hidden2location_std(lstm_out.view(len(seq), -1))
        return (mean,std)

dataset = SeqDataset(args.data_config)
extractor = EthanSeqExtractor(args.data_config)

events = [dataset.propose_Ta(train=True,return_Event=True) for i in range(5)]
inputs, labels, masks = extractor.extract_batch(events)
print(inputs.shape)



model = LSTM_NBA_AR(50,100)
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)