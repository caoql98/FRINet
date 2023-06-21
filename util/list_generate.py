import cv2
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
import os
# from .util import get_train_val_set, check_makedirs
def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if args.split == 3: 
            sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))                    
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))   

        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81)) 
                sub_val_list = list(range(1, 21))

    elif args.data_set == 'iSAID':
        class_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        if args.split == 2:
            sub_list = list(range(1, 11))  #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 16)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 16)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

                
    return sub_list, sub_val_list

def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
      
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])

        # line_split0 = line.strip(line[-4:])
        # line_split1 = line.strip(line[-4:]).strip('_instance_color_RGB.png') + '.png'
        # image_name = os.path.join(data_root, line_split0)
        # label_name = os.path.join(data_root, line_split1)

        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 0:      
                            new_label_class.append(c)     
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 0:      
                        new_label_class.append(c)            

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list

mode  = 'val'
split = 0

data_root = '/data/caoql2022/iSAID_5i/iSAID_5i'
# root_path = 'D:/dataset/iSAID_5i/iSAID_patches/train/images/'
# data_path = 'D:/dataset/iSAID_5i/iSAID_patches/val/semantic_png'
if mode == 'train':
    data_list_path = '/data/caoql2022/iSAID_5i/iSAID_5i/train.txt'
elif mode == 'val':
    data_list_path = '/data/caoql2022/iSAID_5i/iSAID_5i/val.txt'

# with open(data_list_path, 'w') as f:
    # data_name_list = os.listdir(data_path)
    # for index in tqdm(range(len(data_name_list))):
    #     label =  data_name_list[index]
    #     img =  label.strip('_instance_color_RGB.png') + '.png'
    #     f.write('val/images/' +img + ' ')
    #     f.write('val/semantic_png/'+label + '\n')

if split == 2:
    sub_list = list(range(1, 11))  #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
    sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
elif split == 1:
    sub_list = list(range(1, 6)) + list(range(11, 16)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
    sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
elif split == 0:
    sub_list = list(range(6, 16)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    sub_val_list = list(range(1, 6)) #[1,2,3,4,5]
if mode == 'train':
    data_list, sub_class_file_list = make_dataset(split, data_root, data_list_path, sub_list, True)
elif mode == 'val':
    data_list, sub_class_file_list = make_dataset(split, data_root, data_list_path, sub_val_list, False)

fss_list_root = '/code/BAM-main-re/lists/{}/fss_list/{}/'.format('iSAID', mode)
check_makedirs(fss_list_root)
fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

# Write FSS Data
with open(fss_data_list_path, 'w') as f:
    for item in data_list:
        img, label = item
        f.write(img + ' ')
        f.write(label + '\n')
with open(fss_sub_class_file_list_path, 'w') as f:
    f.write(str(sub_class_file_list))

print('end')
