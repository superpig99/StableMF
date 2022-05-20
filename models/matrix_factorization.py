import torch
import torch.nn as nn
from six import iteritems
from collections import defaultdict
from .pretrain import pretrain_SVD

__all__ = ['SVD']

class SVD(nn.Module):
    def __init__(self, train_loader, rating_scale = (1, 5), pretrained=True, args=None,
                 n_factors=100, biased=False, init_mean=0, init_std_dev=.1):

        super(SVD, self).__init__()

        self.rating_scale = rating_scale

        self.n_factors = args.n_factors
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev

        self.n_feature = args.n_feature
        self.gpu = args.gpu        
        
        self.register_buffer('pre_features', torch.zeros(self.n_feature, 2 * self.n_factors))
        self.register_buffer('pre_weight1', torch.ones(self.n_feature, 1))

        # for different levels
        # if args.n_levels > 1:
        #     self.register_buffer('pre_features_2', torch.zeros(args.n_feature, args.feature_dim))
        #     self.register_buffer('pre_weight1_2', torch.ones(args.n_feature, 1))
        # if args.n_levels > 2:
        #     self.register_buffer('pre_features_3', torch.zeros(args.n_feature, args.feature_dim))
        #     self.register_buffer('pre_weight1_3', torch.ones(args.n_feature, 1))
        # if args.n_levels > 3:
        #     self.register_buffer('pre_features_4', torch.zeros(args.n_feature, args.feature_dim))
        #     self.register_buffer('pre_weight1_4', torch.ones(args.n_feature, 1))
        # if args.n_levels > 4:
        #     print('WARNING: THE NUMBER OF LEVELS CAN NOT BE BIGGER THAN 4')

        self.raw2inner_id_users, self.raw2inner_id_items, self.ur, self.ir, self.n_ratings, self.global_mean = self.construct_inner_trainset(train_loader.dataset)
        self.n_users = len(self.ur)      # number of users
        self.n_items = len(self.ir)      # number of items
        
        if pretrained:
            pu, qi, bu, bi = self.pretrain(verbose=False)
            self.pu = nn.Parameter(pu)
            self.qi = nn.Parameter(qi)       
            self.bu = nn.Parameter(bu)
            self.bi = nn.Parameter(bi)
        else:
            self.pu = nn.Parameter(torch.normal(self.init_mean, self.init_std_dev, (self.n_users, self.n_factors)))
            self.qi = nn.Parameter(torch.normal(self.init_mean, self.init_std_dev, (self.n_items, self.n_factors)))       
            self.bu = nn.Parameter(torch.zeros(self.n_users))
            self.bi = nn.Parameter(torch.zeros(self.n_items))



    def construct_inner_trainset(self, raw_trainset):
        n_ratings = len(raw_trainset)

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)
        all_r = torch.zeros(n_ratings)

        # user raw id, item raw id, translated rating
        for i, (urid_irid, r) in enumerate(raw_trainset):
            urid = urid_irid[0]
            irid = urid_irid[1]
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))
            all_r[i] = r

        global_mean = torch.mean(all_r)
        return raw2inner_id_users, raw2inner_id_items, ur, ir, n_ratings, global_mean

    def forward(self, user_item, clip=True):
        batch_size = user_item.size()[0]
        predict = self.global_mean * torch.ones(batch_size)
        flatten_features = torch.zeros(self.n_feature, 2 * self.n_factors)   
        if self.gpu is not None:
            flatten_features = flatten_features.cuda(self.gpu, non_blocking=True)
            predict = predict.cuda(self.gpu, non_blocking=True)

        if self.biased:
            for i in range(0, batch_size):
                ruid = int(user_item[i][0])
                riid = int(user_item[i][1])
                try:
                    iuid = self.raw2inner_id_users[ruid]
                    predict[i] = predict [i] + self.bu[iuid]
                except KeyError:
                    iuid = None
                try:
                    iiid = self.raw2inner_id_items[riid]
                    predict[i] = predict [i] + self.bi[iiid]
                except KeyError:
                    iiid = None
                if iuid and iiid:
                    pu = self.pu[iuid]
                    qi = self.qi[iiid]
                    flatten_features[i] = torch.cat((pu, qi))
                    predict[i] = predict[i] + torch.sum(torch.mul(pu, qi))
        else:
            for i in range(0, batch_size):
                ruid = int(user_item[i][0])
                riid = int(user_item[i][1])
                try:
                    iuid = self.raw2inner_id_users[ruid]
                except KeyError:
                    iuid = None
                try:
                    iiid = self.raw2inner_id_items[riid]
                except KeyError:
                    iiid = None
                if iuid and iiid:
                    pu = self.pu[iuid]
                    qi = self.qi[iiid]
                    flatten_features[i] = torch.cat((pu, qi))
                    # flatten_features[i] = qi
                    predict[i] = torch.sum(torch.mul(pu, qi))

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.rating_scale
            predict = torch.min(predict, higher_bound * torch.ones_like(predict))
            predict = torch.max(predict, lower_bound * torch.ones_like(predict))
        return predict, flatten_features   # output.shape is torch.Size([pairs])

    def pretrain(self, n_epochs=20, verbose=True,
                 lr_all=.005, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_all=.02, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None):
        pu, qi, bu, bi = pretrain_SVD(self, n_epochs=n_epochs, verbose=verbose,
                                      lr_all=lr_all, lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi,
                                      reg_all=reg_all, reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi)
        pu = torch.from_numpy(pu).to(torch.float32)
        qi = torch.from_numpy(qi).to(torch.float32)
        bu = torch.from_numpy(bu).to(torch.float32)
        bi = torch.from_numpy(bi).to(torch.float32)
        return pu, qi, bu, bi