import argparse
import logging
import os
import random
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn

from lib.networks import PVT_GCASCADE, MERIT_GCASCADE

from trainer import trainer_lits
from torchsummaryX import summary
from ptflops import get_model_complexity_info
from utils.utils import random_split_array

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str,
                    default='PVT', help='Name of encoder: PVT or MERIT')
parser.add_argument('--skip_aggregation', type=str,
                    default='additive', help='Type of skip-aggregation: additive or concatenation')
parser.add_argument('--root_path', type=str,
                    default='./data/lits/', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu') #6
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input') #256
parser.add_argument('--seed', type=int,
                    default=32, help='random seed')
parser.add_argument('--log_interval', type=int,
                    default=50, help='Interval for training logging')
parser.add_argument('--save_interval', type=int,
                    default=50, help='Interval for saving model')
parser.add_argument('--val_log_interval', type=int,
                    default=10, help='Interval for validation set evaluation logging')
parser.add_argument('--is_liver', action='store_true',
                    default=0, help='add for liver, remove for tumor')


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    dataset_name = 'Lits'
    dataset_config = {
        'Lits': {
            'root_path': args.root_path,
            'num_classes': args.num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = True
    args.exp = 'PVT_GCASCADE_MUTATION_w3_7_Run1_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'PVT_GCASCADE_MUTATION_w3_7_Run1')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    #current_time = time.strftime("%H%M%S")
    #print("The current time is", current_time)
    #snapshot_path = snapshot_path +'_t'+current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    if args.encoder=='PVT':
        net = PVT_GCASCADE(n_class=args.num_classes, img_size=args.img_size, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    elif args.encoder=='MERIT':
        net = MERIT_GCASCADE(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    else:
        print('Implementation not found for this encoder. Exiting!')
        sys.exit()

    print('Model %s created' % (args.encoder+'-GCASCADE: '))

    net = net.cuda()
   
    macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    

    #This is for generating train-test-split
    if (args.is_liver):
      organ = "liver"
    else:
      organ = "cancer"

    #Splitting Dataset
    original = []
    for i in range(131) :
      original.append(i)

    train, test, val = random_split_array(original,(0.8,0.1,0.1))
    print("SEED: ",args.seed)
    print("Training Folders - ")
    print(train)
    print("Testing Folders - ")
    print(test)
    print("Validation Folders - ")
    print(val)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_val = []
    Y_val = []

    scan_list = os.listdir(args.root_path)
    scan_list.sort()
    for i in scan_list:
      num = int(i.split("_")[-1])
      path = os.path.join(args.root_path,i)
      imgpath = os.path.join(path,"images")
      maskpath = os.path.join(path,"masks")
      piclist = os.listdir(imgpath)
      if num in train:
        for j in piclist:
            X_train.append(os.path.join(imgpath,j))
            Y_train.append(os.path.join(os.path.join(maskpath,organ),j))
      elif num in test:
        for j in piclist:
            X_test.append(os.path.join(imgpath,j))
            Y_test.append(os.path.join(os.path.join(maskpath,organ),j))
      else:
        for j in piclist:
            X_val.append(os.path.join(imgpath,j))
            Y_val.append(os.path.join(os.path.join(maskpath,organ),j))
    print(len(X_train))     
    print("Train length : ", len(X_train))
    print("Test length : ", len(X_test))
    print("Val length : ", len(X_val))
    
    trainer = {'Lits': trainer_lits,}
    print("Snapshot path: ", snapshot_path)
    trainer[dataset_name](args, net, snapshot_path, X_train, Y_train, X_val, Y_val)
