import cv2
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
# from .util import get_train_val_set, check_makedirs
import os
# Get the annotations of base categories
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
# root_path
# ├── BAM/
# │   ├── util/
# │   ├── config/
# │   ├── model/
# │   ├── README.md
# │   ├── train.py
# │   ├── train_base.py
# │   └── test.py
# └── data/
#     ├── base_annotation/   # the scripts to create THIS folder
#     │   ├── pascal/
#     │   │   ├── train/   
#     │   │   │   ├── 0/     # annotations of PASCAL-5^0
#     │   │   │   ├── 1/
#     │   │   │   ├── 2/
#     │   │   │   └── 3/
#     │   │   └── val/      
#     │   └── coco/          # the same file structure for COCO
#     ├── VOCdevkit2012/
#     └── MSCOCO2014/

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.data_set = 'iSAID'  # pascal coco
args.use_split_coco = True
args.mode = 'val'       # train val
args.split = 0           # 0 1 2 
if args.data_set == 'pascal':
    num_classes = 20
elif args.data_set == 'coco':
    num_classes = 80
elif args.data_set == 'iSAID':
    num_classes = 15

root_path = '/code/BAM-main-re'
data_path = osp.join(root_path, 'base_annotation/')
save_path = osp.join(data_path, args.data_set, args.mode, str(args.split))
check_makedirs(save_path)

# get class list
sub_list, sub_val_list = get_train_val_set(args)

# get data_list
fss_list_root = root_path + '/lists/{}/fss_list/{}/'.format(args.data_set, args.mode)
fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(args.split)
with open(fss_data_list_path, 'r') as f:
    f_str = f.readlines()
data_list = []
for line in f_str:
    img, mask = line.split(' ')
    data_list.append((img, mask.strip()))

# Start Processing
for index in tqdm(range(len(data_list))):
    image_path, label_path = data_list[index]
    # image_path, label_path = root_path + image_path[3:], root_path+ label_path[3:]  # 
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label_tmp = label.copy()

    for cls in range(1,num_classes+1):
        select_pix = np.where(label_tmp == cls)
        if cls in sub_list:
            label[select_pix[0],select_pix[1]] = sub_list.index(cls) + 1
        else:
            label[select_pix[0],select_pix[1]] = 0

    # for pix in np.nditer(label, op_flags=['readwrite']):
    #     if pix == 255:
    #         pass
    #     elif pix not in sub_list: 
    #         pix[...] = 0
    #     else:
    #         pix[...] = sub_list.index(pix) + 1
    
    save_item_path = osp.join(save_path, label_path.split('/')[-1])
    cv2.imwrite(save_item_path, label)


print('end')