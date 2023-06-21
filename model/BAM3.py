import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set
# from timm.models.layers import to_2tuple, trunc_normal_

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H_x, W_x, y, H_y, W_y):
        B_x, N_x, C_x = x.shape
        B_y, N_y, C_y = y.shape
        assert B_x == 1 or B_y == 1
        # assert B_y == 1
        q_x = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        q_y = self.q(y).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)
        # print("q_x.shape={}, q_y.shape={}".format(q_x.shape, q_y.shape))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
                x_ = self.sr(x_).reshape(B_x, C_x, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

                y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
                y_ = self.sr(y_).reshape(B_y, C_y, -1).permute(0, 2, 1)
                y_ = self.norm(y_)
                kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv_x = self.kv(x).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
                kv_y = self.kv(y).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
            x_ = self.sr(self.pool(x_)).reshape(B_x, C_x, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

            y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
            y_ = self.sr(self.pool(y_)).reshape(B_y, C_y, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            y_ = self.act(y_)
            kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)

        k_x, v_x = kv_x[0], kv_x[1]
        k_y, v_y = kv_y[0], kv_y[1]
        # print("k_x.shape={}, k_y.shape={}".format(k_x.shape, k_y.shape))
        # print("v_x.shape={}, v_y.shape={}".format(v_x.shape, v_y.shape))

        if B_x == 1:
            k_y_avg = k_y.mean(0, True)
            v_y_avg = v_y.mean(0, True)
            k_cat_x = torch.cat((k_x, k_y_avg), dim=2)
            v_cat_x = torch.cat((v_x, v_y_avg), dim=2)
        elif B_y == 1:
            k_y_ext = k_y.repeat(B_x, 1, 1, 1)
            v_y_ext = v_y.repeat(B_x, 1, 1, 1)
            k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
            v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        # print("k_cat.shape={}, v_cat.shape={}".format(k_cat.shape, v_cat.shape))

        attn_x = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        x = (attn_x @ v_cat_x).transpose(1, 2).reshape(B_x, N_x, C_x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if B_x == 1:
            k_x_ext = k_x.repeat(B_y, 1, 1, 1)
            v_x_ext = v_x.repeat(B_y, 1, 1, 1)
            k_cat_y = torch.cat((k_x_ext, k_y), dim=2)
            v_cat_y = torch.cat((v_x_ext, v_y), dim=2)
        elif B_y == 1:
            k_x_avg = k_x.mean(0, True)
            v_x_avg = v_x.mean(0, True)
            k_cat_y = torch.cat((k_x_avg, k_y), dim=2)
            v_cat_y = torch.cat((v_x_avg, v_y), dim=2)

        attn_y = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)
        attn_y = self.attn_drop(attn_y)

        y = (attn_y @ v_cat_y).transpose(1, 2).reshape(B_y, N_y, C_y)
        y = self.proj(y)
        y = self.proj_drop(y)

        return x, y #torch.cat((x, y), dim=1)


class single_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,  attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, y):
        B_x, C_x, H_x, W_x = x.shape
        B_y, C_y,H_y, W_y= y.shape #N_y = shot*4
        N_x = H_x*W_x
        N_y = H_y*W_y
        x = x.reshape(B_x, C_x, N_x).permute(0,2,1)
        y = y.reshape(B_y, C_y, N_y).permute(0,2,1)

        q_x = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        # q_y = self.q(y).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)


        # kv_x = self.kv(x).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_y = self.kv(y).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)

        # k_x, v_x = kv_x[0], kv_x[1]  # B_x, self.num_heads, N_x, C_x // self.num_heads
        k_y, v_y = kv_y[0], kv_y[1]

        # if B_x == 1:
        #     k_y_avg = k_y.mean(0, True)
        #     v_y_avg = v_y.mean(0, True)
        #     k_cat_x = torch.cat((k_x, k_y_avg), dim=2)
        #     v_cat_x = torch.cat((v_x, v_y_avg), dim=2)
        # elif B_y == 1:
        #     k_y_ext = k_y.repeat(B_x, 1, 1, 1)
        #     v_y_ext = v_y.repeat(B_x, 1, 1, 1)
        #     k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
        #     v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        # print("k_cat.shape={}, v_cat.shape={}".format(k_cat.shape, v_cat.shape))

        attn_x = (q_x @ k_y.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        x = (attn_x @ v_y).transpose(1, 2).reshape(B_x, N_x, C_x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0,2,1).reshape(B_x, C_x, H_x, W_x)
   
        return x

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.attention1 = single_Attention(dim=256)
        self.attention2 = single_Attention(dim=256)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        elif self.dataset == 'iSAID':
            self.base_classes = 10

        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)               
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512           
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                {'params': model.kshot_rw.parameters()},
                {'params': model.attention1.parameters()},
                {'params': model.attention2.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},        
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},      
                {'params': model.attention1.parameters()},
                {'params': model.attention2.parameters()},          
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False


    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = [] 
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(eval('supp_feat_' + self.low_fea_id))

        # Support Feature1
        supp_pro_list1 = []
        final_supp_list1 = []
        mask_list1 = []
        s_y1 = torch.rot90(s_y,1,[2,3])
        s_x1 = torch.rot90(s_x,1,[3,4])
        for i in range(self.shot):
            mask = (s_y1[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list1.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x1[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list1.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro1 = Weighted_GAP(supp_feat, mask)
            supp_pro_list1.append(supp_pro1)

        # Support Feature2
        supp_pro_list2 = []
        final_supp_list2 = []
        mask_list2 = []
        s_y2 = torch.rot90(s_y,2,[2,3])
        s_x2 = torch.rot90(s_x,2,[3,4])
        for i in range(self.shot):
            mask = (s_y2[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list2.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x2[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list2.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro2 = Weighted_GAP(supp_feat, mask)
            supp_pro_list2.append(supp_pro2)

        # Support Feature3
        supp_pro_list3 = []
        final_supp_list3 = []
        mask_list3 = []
        s_y3 = torch.rot90(s_y,3,[2,3])
        s_x3 = torch.rot90(s_x,3,[3,4])
        for i in range(self.shot):
            mask = (s_y3[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list3.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x3[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list3.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_pro3 = Weighted_GAP(supp_feat, mask)
            supp_pro_list3.append(supp_pro3)

        # K-Shot Reweighting
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1,True) # [bs, 1, 1, 1]            

        # Prior Similarity Mask
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1,True)

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro1 = torch.cat(supp_pro_list1, 2)  # [bs, 256, shot, 1]
        supp_pro2 = torch.cat(supp_pro_list2, 2)  # [bs, 256, shot, 1]
        supp_pro3 = torch.cat(supp_pro_list3, 2)  # [bs, 256, shot, 1]

        supp_pro_all = torch.cat([supp_pro,supp_pro1,supp_pro2,supp_pro3],2) # [bs, 256, 4*shot, 1]
        concat_feat = self.attention1(query_feat,supp_pro_all)
        supp_pro = (weight_soft.permute(0,2,1,3) * supp_pro).sum(2,True)
        support_feat_attention = self.attention2(supp_pro_all, query_feat)
        support_feat_attention = support_feat_attention.mean(dim=2,keepdim=True).expand_as(query_feat)
        # Tile & Cat
        # concat_feat = supp_pro.expand_as(query_feat)
        # merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)   # 256+256+1
        merge_feat = torch.cat([support_feat_attention, concat_feat, corr_query_mask], 1)   # 256+256+1
        merge_feat = self.init_merge(merge_feat)

        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out = self.cls_meta(query_meta)
        
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:,1:,:,:]                            # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out, meta_out, base_out

