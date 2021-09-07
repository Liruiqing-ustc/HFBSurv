from __future__ import print_function
import torch
import adabound
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



class SALMON(nn.Module):
    def __init__(self, in_size, output_dim, hidden=16):
        super(SALMON, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_size, hidden), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(hidden, output_dim),nn.Sigmoid())
    def forward(self, x):
        code = self.encoder(x)
        pred = self.classifier(code)

        return pred,code

class GDBFN(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.1):
        super(GDBFN, self).__init__()

        skip_dim =in_size*2

        self.gene_gene = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())
        self.path_path = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())
        self.gene_path = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())

        encoder1 = nn.Sequential(nn.Linear((skip_dim+hidden*3), 500), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(500,256), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1,encoder2,encoder3,encoder4)
        self.classifier = nn.Sequential(nn.Linear(in_size,output_dim), nn.Sigmoid())


    def forward(self, x_gene,x_path):
        o1 = x_gene.squeeze(1)
        o2 = x_path.squeeze(1)
        o11 = torch.bmm(o1.unsqueeze(2), o1.unsqueeze(1)).flatten(start_dim=1)
        o22 = torch.bmm(o2.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        inter = self.gene_path(o12)
        intra_gene = self.gene_gene(o11)
        intra_path = self.path_path(o22)
        fusion = torch.cat((o1,o2,inter,intra_gene,intra_path),1)
        code = self.encoder(fusion)
        out =  self.classifier(code)

        return out, code


class Unimodal1(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.3):
        super(Unimodal1, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size,256), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(p=dropout))
        encoder5 = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1, encoder2,encoder3, encoder4,encoder5)
        self.classifier = nn.Sequential(nn.Linear(hidden, output_dim),nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x):

        code = self.encoder(x)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out,code

class Unimodal(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.3):
        super(Unimodal, self).__init__()

        encoder1 = nn.Sequential(nn.Linear(in_size, hidden), nn.Tanh(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Dropout(p=0.15))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3)
        self.classifier = nn.Sequential(nn.Linear(32, output_dim), nn.Sigmoid())

    def forward(self, x):
        code = self.encoder(x.squeeze(1))
        out = self.classifier(code)

        return out,code
















class Bimodal(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.3):
        super(Bimodal, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size*2, 400), nn.ReLU(),nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(400,256), nn.ReLU(),nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),nn.Dropout(p=dropout))
        encoder5 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),nn.Dropout(p=dropout))
        encoder6 = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4,encoder5,encoder6)
        self.classifier = nn.Sequential(nn.Linear(hidden, output_dim), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    def forward(self, x1,x2):
        fusion = torch.cat((x1,x2),2)
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out, code

class Trimodal(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.5):
        super(Trimodal, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size * 3, 500), nn.ReLU(), nn.Dropout(P=0.3))
        encoder2 = nn.Sequential(nn.Linear(500, 256), nn.ReLU(), nn.Dropout(P=0.3))
        encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.3))
        encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=0.3))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(in_size, output_dim), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x1,x2,x3):
        fusion = torch.cat((x1,x2,x3),2)
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift

        return out, code

