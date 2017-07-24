from __future__ import print_function, division
import os
import argparse
import time
import yaml
import pickle
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import tensorboard_logger
import sportvu
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
parser.add_argument('--labels-format', default='velocity', help='')


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
print(args)
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
if args.save_file:
    args.save_file = os.path.join(args.log_dir,args.save_file)

if args.log:
    tensorboard_logger.configure(args.log_dir, flush_secs=2)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_data(dataset, extractor, batch_size, train=True, normalize=True,labels_format='velocity'):
    events = [dataset.propose_Ta(train=train, return_Event=True) for i in range(batch_size)]
    inputs, labels, masks = extractor.extract_batch(events)
    shape = inputs.shape
    if normalize:
        inputs[:, :, :, 0] = (inputs[:, :, :, 0] - 50.) / 50.
        inputs[:, :, :, 1] = (inputs[:, :, :, 1] - 25.) / 25.
        labels[:, :, :, 0] = (labels[:, :, :, 0] - 50.) / 50.
        labels[:, :, :, 1] = (labels[:, :, :, 1] - 25.) / 25.

    inputs = inputs.transpose(2,0,1,3).reshape(shape[2], shape[0],-1).transpose(0,2,1)
    labels = labels.transpose(2,0,1,3).reshape(shape[2], shape[0],-1).transpose(0,2,1)
    masks = masks.transpose(0,1)
    if labels_format == 'location':
        pass
    elif labels_format == 'velocity':
        labels = labels-inputs
    if train:
        return Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels)), Variable(torch.from_numpy(masks))
    else:
        return Variable(torch.from_numpy(inputs),volatile=True), Variable(torch.from_numpy(labels),volatile=True), Variable(torch.from_numpy(masks),volatile=True)

def diagonal_gauss_nll(labels, mean, std_sq, masks):
    x = (torch.pow(mean - labels, 2) / (2 * std_sq) - 0.5 * torch.log(std_sq))
    return (x.mean(1).view(x.size(0), -1)*masks).sum()/masks.sum()


class LSTM_NBA_AR(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, eps = 1e-6):
        super(LSTM_NBA_AR, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.location_dim = 11*2
        self.state_embeddings = nn.Conv1d(self.location_dim, embedding_dim,kernel_size=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2location_mean = nn.Linear(hidden_dim, self.location_dim)
        self.hidden2location_std_sq = nn.Linear(hidden_dim, self.location_dim)


    def forward(self, seq):
        h0 = Variable(torch.zeros(1, seq.size(2), self.hidden_dim)).double()
        c0 = Variable(torch.zeros(1, seq.size(2), self.hidden_dim)).double()
        # h0 = Variable(torch.zeros(1, seq.size(2), self.hidden_dim))
        # c0 = Variable(torch.zeros(1, seq.size(2), self.hidden_dim))
        if args.cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        embeds = self.state_embeddings(seq)
        lstm_out, (h,c) = self.lstm(
            embeds.transpose(1,2), (h0,c0))
        mean = self.hidden2location_mean(lstm_out.view(-1, self.hidden_dim))
        std_sq = torch.pow(self.hidden2location_std_sq(lstm_out.view(-1, self.hidden_dim)),2)+self.eps

        mean = mean.view(lstm_out.size(0),lstm_out.size(1),-1).transpose(1,2)
        std_sq = std_sq.view(lstm_out.size(0),lstm_out.size(1),-1).transpose(1,2)
        return mean, std_sq

dataset = SeqDataset(args.data_config)
extractor = EthanSeqExtractor(args.data_config)





model = LSTM_NBA_AR(50,100)
if args.cuda:
    model.cuda()
    model.double()

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)


for counter in range(args.iter):
    inputs, labels, masks = get_data(dataset, extractor, args.batch_size, labels_format=args.labels_format)
    if args.cuda:
        inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
    optimizer.zero_grad()
    mean, std_sq = model.forward(inputs)
    loss = diagonal_gauss_nll(labels, mean, std_sq, masks)
    l2_dist = torch.pow(mean-labels,2)
    l2_dist = (l2_dist.mean(1).view(l2_dist.size(0), -1) * masks).sum() / masks.sum()
    if args.log:
        loss_log = loss.cpu().data.numpy()[0]
        l2_dist_log = l2_dist.cpu().data.numpy()[0]
        tensorboard_logger.log_value('train_loss', loss_log,step=counter)
        tensorboard_logger.log_value('train_l2_dist', l2_dist_log, step=counter)
    print('Iter {} train loss {}, l2_dist {}'.format(counter, loss_log, l2_dist_log))
    loss.backward()
    optimizer.step()
    if counter % args.lr_step == 0 and counter>0:
        lr *= args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if counter % args.val_interval == 0 and counter>0:

        model.eval()
        val_loss = 0
        val_dist = 0
        val_iter = 0
        for i in range(100):
            inputs, labels, masks = get_data(dataset, extractor, args.batch_size,train=False, labels_format=args.labels_format)
            if args.cuda:
                inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            mean, std_sq = model.forward(inputs)
            loss = diagonal_gauss_nll(labels, mean, std_sq, masks)
            l2_dist = torch.pow(mean-labels,2)
            l2_dist = (l2_dist.mean(1).view(l2_dist.size(0), -1) * masks).sum() / masks.sum()
            val_loss += loss
            val_dist += l2_dist
            val_iter += 1
        if args.log:
            val_loss_log = val_loss.cpu().data.numpy()[0]/val_iter
            val_l2_dist_log = val_dist.cpu().data.numpy()[0]/val_iter
            tensorboard_logger.log_value('val_loss', loss_log, step=counter)
            tensorboard_logger.log_value('val_l2_dist', l2_dist_log, step=counter)

        model.train()
        print('Iter {} val loss {}, l2_dist {}'.format(counter, val_loss_log, val_l2_dist_log))
        if args.save_file and (val_loss_log)<min_val_loss:
            min_val_loss = val_loss_log
            torch.save(model.state_dict(),args.save_file)


