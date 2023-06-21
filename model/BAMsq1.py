import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2
import os
import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set
# from timm.models.layers import to_2tuple, trunc_normal_
import cv2
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
        self.r_merge1 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.r_merge1.weight = nn.Parameter(torch.tensor([[0.25],[0.25],[0.25],[0.25]]).reshape_as(self.r_merge1.weight))
        self.r_merge2 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.r_merge2.weight = nn.Parameter(torch.tensor([[0.25],[0.25],[0.25],[0.25]]).reshape_as(self.r_merge2.weight))


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
                {'params': model.r_merge1.parameters()},      
                {'params': model.r_merge2.parameters()},                                      
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
                {'params': model.r_merge1.parameters()},      
                {'params': model.r_merge2.parameters()},            
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

        # Query Feature1
        x1 = torch.rot90(x,1,[2,3])
        with torch.no_grad():
            query_feat1_0 = self.layer0(x1)
            query_feat1_1 = self.layer1(query_feat1_0)
            query_feat1_2 = self.layer2(query_feat1_1)
            query_feat1_3 = self.layer3(query_feat1_2)
            query_feat1_4 = self.layer4(query_feat1_3)
            if self.vgg:
                query_feat1_2 = F.interpolate(query_feat1_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat1 = torch.cat([query_feat1_3, query_feat1_2], 1)
        query_feat1 = self.down_query(query_feat1)

        # Query Feature2
        x2 = torch.rot90(x,2,[2,3])
        with torch.no_grad():
            query_feat2_0 = self.layer0(x2)
            query_feat2_1 = self.layer1(query_feat2_0)
            query_feat2_2 = self.layer2(query_feat2_1)
            query_feat2_3 = self.layer3(query_feat2_2)
            query_feat2_4 = self.layer4(query_feat2_3)
            if self.vgg:
                query_feat2_2 = F.interpolate(query_feat2_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat2 = torch.cat([query_feat2_3, query_feat2_2], 1)
        query_feat2 = self.down_query(query_feat2)

        # Query Feature3
        x3 = torch.rot90(x,3,[2,3])
        with torch.no_grad():
            query_feat3_0 = self.layer0(x3)
            query_feat3_1 = self.layer1(query_feat3_0)
            query_feat3_2 = self.layer2(query_feat3_1)
            query_feat3_3 = self.layer3(query_feat3_2)
            query_feat3_4 = self.layer4(query_feat3_3)
            if self.vgg:
                query_feat3_2 = F.interpolate(query_feat3_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat3 = torch.cat([query_feat3_3, query_feat3_2], 1)
        query_feat3 = self.down_query(query_feat3)

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

        # K-Shot Reweighting1
        que_gram = get_gram_matrix(eval('query_feat1_' + self.low_fea_id)) # [bs, C, C] in (0,1)
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
            weight_soft1 = torch.softmax(weight, 1)
        else:
            weight_soft1 = torch.ones_like(est_val_total)
        est_val1 = (weight_soft1 * est_val_total).sum(1,True) # [bs, 1, 1, 1]   

        # K-Shot Reweighting2
        que_gram = get_gram_matrix(eval('query_feat2_' + self.low_fea_id)) # [bs, C, C] in (0,1)
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
            weight_soft2 = torch.softmax(weight, 1)
        else:
            weight_soft2 = torch.ones_like(est_val_total)
        est_val2 = (weight_soft2 * est_val_total).sum(1,True) # [bs, 1, 1, 1]

        # K-Shot Reweighting2
        que_gram = get_gram_matrix(eval('query_feat3_' + self.low_fea_id)) # [bs, C, C] in (0,1)
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
            weight_soft3 = torch.softmax(weight, 1)
        else:
            weight_soft3 = torch.ones_like(est_val_total)
        est_val3 = (weight_soft3 * est_val_total).sum(1,True) # [bs, 1, 1, 1]

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

        corr_query_mask1 = torch.rot90(corr_query_mask,1,[2,3])
        corr_query_mask2 = torch.rot90(corr_query_mask,2,[2,3])
        corr_query_mask3 = torch.rot90(corr_query_mask,3,[2,3])

        # Support Prototype
        supp_pro = torch.cat(supp_pro_list, 2)  # [bs, 256, shot, 1]
        supp_pro1 = torch.cat(supp_pro_list1, 2)  # [bs, 256, shot, 1]
        supp_pro2 = torch.cat(supp_pro_list2, 2)  # [bs, 256, shot, 1]
        supp_pro3 = torch.cat(supp_pro_list3, 2)  # [bs, 256, shot, 1]

        supp_pro_all = torch.cat([supp_pro,supp_pro1,supp_pro2,supp_pro3],2) # [bs, 256, 4*shot, 1]
        supp_pro_all = supp_pro_all.squeeze(-1)
        normalize_supp_pro_all = F.normalize(supp_pro_all, dim=1)# [bs, 256, 4*shot]
        bq,cq,hq,wq = query_feat.shape
        normalize_query_feat = F.normalize(query_feat.reshape(bq,cq,hq*wq), dim=1)
        cosine_matrix = torch.bmm(normalize_supp_pro_all.permute(0,2,1), normalize_query_feat)# [bs, 4*shot, hq*wq]
        cosine_matrix = F.softmax(cosine_matrix,dim=1)
        concat_feat = torch.bmm(supp_pro_all,cosine_matrix)# [bs, 256, hq*wq]
        concat_feat = concat_feat.reshape(bq,cq,hq,wq)

        # Tile & Cat
        # concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask1], 1)   # 256+256+1
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


        # concat_feat1 = self.attention(query_feat1,supp_pro_all.squeeze(-1))
        normalize_query_feat = F.normalize(query_feat1.reshape(bq,cq,hq*wq), dim=1)
        cosine_matrix = torch.bmm(normalize_supp_pro_all.permute(0,2,1), normalize_query_feat)# [bs, 4*shot, hq*wq]
        cosine_matrix = F.softmax(cosine_matrix,dim=1)
        concat_feat1 = torch.bmm(supp_pro_all,cosine_matrix)# [bs, 256, hq*wq]
        concat_feat1 = concat_feat1.reshape(bq,cq,hq,wq)
        # Tile & Cat
        # concat_feat = supp_pro.expand_as(query_feat)
        merge_feat1 = torch.cat([query_feat1, concat_feat1, corr_query_mask1], 1)   # 256+256+1
        merge_feat1 = self.init_merge(merge_feat1)
        # Base and Meta
        base_out1 = self.learner_base(query_feat1_4)
        query_meta = self.ASPP_meta(merge_feat1)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out1 = self.cls_meta(query_meta)
        meta_out_soft = meta_out1.softmax(1)
        base_out_soft = base_out1.softmax(1)
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
        est_map = est_val1.expand_as(meta_map_fg)
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]
        final_out1 = torch.cat([merge_bg, meta_map_fg], dim=1)

        base_out1 = torch.rot90(base_out1,-1,[2,3])
        final_out1= torch.rot90(final_out1,-1,[2,3])


        # concat_feat2 = self.attention(query_feat2,supp_pro_all.squeeze(-1))
        # Tile & Cat
        # concat_feat = supp_pro.expand_as(query_feat)
        normalize_query_feat = F.normalize(query_feat2.reshape(bq,cq,hq*wq), dim=1)
        cosine_matrix = torch.bmm(normalize_supp_pro_all.permute(0,2,1), normalize_query_feat)# [bs, 4*shot, hq*wq]
        cosine_matrix = F.softmax(cosine_matrix,dim=1)
        concat_feat2 = torch.bmm(supp_pro_all,cosine_matrix)# [bs, 256, hq*wq]
        concat_feat2 = concat_feat2.reshape(bq,cq,hq,wq)
        merge_feat2 = torch.cat([query_feat2, concat_feat2, corr_query_mask2], 1)   # 256+256+1
        merge_feat2 = self.init_merge(merge_feat2)
        # Base and Meta
        base_out2 = self.learner_base(query_feat2_4)
        query_meta = self.ASPP_meta(merge_feat2)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out2 = self.cls_meta(query_meta)
        meta_out_soft = meta_out2.softmax(1)
        base_out_soft = base_out2.softmax(1)
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
        est_map = est_val2.expand_as(meta_map_fg)
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]
        final_out2 = torch.cat([merge_bg, meta_map_fg], dim=1)

        base_out2 = torch.rot90(base_out2,-2,[2,3])
        final_out2= torch.rot90(final_out2,-2,[2,3])

        # concat_feat3 = self.attention(query_feat3,supp_pro_all.squeeze(-1))
        # Tile & Cat
        # concat_feat = supp_pro.expand_as(query_feat)
        normalize_query_feat = F.normalize(query_feat3.reshape(bq,cq,hq*wq), dim=1)
        cosine_matrix = torch.bmm(normalize_supp_pro_all.permute(0,2,1), normalize_query_feat)# [bs, 4*shot, hq*wq]
        cosine_matrix = F.softmax(cosine_matrix,dim=1)
        concat_feat3 = torch.bmm(supp_pro_all,cosine_matrix)# [bs, 256, hq*wq]
        concat_feat3 = concat_feat3.reshape(bq,cq,hq,wq)
        merge_feat3 = torch.cat([query_feat3, concat_feat3, corr_query_mask3], 1)   # 256+256+1
        merge_feat3 = self.init_merge(merge_feat3)
        # Base and Meta
        base_out3 = self.learner_base(query_feat3_4)
        query_meta = self.ASPP_meta(merge_feat3)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out3 = self.cls_meta(query_meta)
        meta_out_soft = meta_out3.softmax(1)
        base_out_soft = base_out3.softmax(1)
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
        est_map = est_val3.expand_as(meta_map_fg)
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]
        final_out3 = torch.cat([merge_bg, meta_map_fg], dim=1)

        base_out3 = torch.rot90(base_out3,-3,[2,3])
        final_out3= torch.rot90(final_out3,-3,[2,3])


        # query_merge1_outb = self.r1_merge1(torch.cat([final_out[:,0:1,:,:] , final_out1[:,0:1,:,:] ], 1))
        # query_merge1_outf = self.r1_merge2(torch.cat([final_out[:,1:,:,:] , final_out1[:,1:,:,:] ], 1))
        # query_merge1_aout = torch.cat([query_merge1_outb, query_merge1_outf], 1)
        # query_merge2_outb = self.r2_merge1(torch.cat([final_out2[:,0:1,:,:] , final_out3[:,0:1,:,:] ], 1))
        # query_merge2_outf = self.r2_merge2(torch.cat([final_out2[:,1:,:,:] , final_out3[:,1:,:,:] ], 1))
        # query_merge2_aout = torch.cat([query_merge2_outb, query_merge2_outf], 1)
        # query_merge3_outb = self.r3_merge1(torch.cat([query_merge1_aout[:,0:1,:,:] , query_merge2_aout[:,0:1,:,:] ], 1))
        # query_merge3_outf = self.r3_merge2(torch.cat([query_merge1_aout[:,1:,:,:] , query_merge2_aout[:,1:,:,:] ], 1))
        # query_merge_aout = torch.cat([query_merge3_outb, query_merge3_outf], 1)

        query_merge_outb = self.r_merge1(torch.cat([final_out[:,0:1,:,:], final_out1[:,0:1,:,:], final_out2[:,0:1,:,:], final_out3[:,0:1,:,:] ] , 1))
        query_merge_outf = self.r_merge2(torch.cat([final_out[:,1:,:,:] , final_out1[:,1:,:,:], final_out2[:,1:,:,:],final_out3[:,1:,:,:], ], 1))
        query_merge_aout = torch.cat([query_merge_outb, query_merge_outf], 1)


        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

            meta_out1 = F.interpolate(meta_out1, size=(h, w), mode='bilinear', align_corners=True)
            base_out1 = F.interpolate(base_out1, size=(h, w), mode='bilinear', align_corners=True)
            final_out1 = F.interpolate(final_out1, size=(h, w), mode='bilinear', align_corners=True)

            meta_out2 = F.interpolate(meta_out2, size=(h, w), mode='bilinear', align_corners=True)
            base_out2 = F.interpolate(base_out2, size=(h, w), mode='bilinear', align_corners=True)
            final_out2 = F.interpolate(final_out2, size=(h, w), mode='bilinear', align_corners=True)

            meta_out3 = F.interpolate(meta_out3, size=(h, w), mode='bilinear', align_corners=True)
            base_out3 = F.interpolate(base_out3, size=(h, w), mode='bilinear', align_corners=True)
            final_out3 = F.interpolate(final_out3, size=(h, w), mode='bilinear', align_corners=True)

            query_merge_aout = F.interpolate(query_merge_aout, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())

            main_loss1 = self.criterion(final_out1, y_m.long())
            aux_loss11 = self.criterion(meta_out1, y_m.long())
            aux_loss21 = self.criterion(base_out1, y_b.long())

            main_loss2 = self.criterion(final_out2, y_m.long())
            aux_loss12 = self.criterion(meta_out2, y_m.long())
            aux_loss22 = self.criterion(base_out2, y_b.long())

            main_loss3 = self.criterion(final_out3, y_m.long())
            aux_loss13= self.criterion(meta_out3, y_m.long())
            aux_loss23 = self.criterion(base_out3, y_b.long())

            main_lossa = self.criterion(query_merge_aout, y_m.long())

            main_loss =  main_lossa + 0.25*main_loss+0.25*main_loss1+0.25*main_loss2+0.25*main_loss3 
            aux_loss1 = 0.25*aux_loss1+0.25*aux_loss11+0.25*aux_loss12+0.25*aux_loss13
            aux_loss2 =  0.25*aux_loss2+0.25*aux_loss21+0.25*aux_loss22+0.25*aux_loss23

            return query_merge_aout.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return query_merge_aout, meta_out, base_out

