import argparse
import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader


from utils.dataset_lits import LITSTestDataset, LITSDataset
from utils.utils import test_single_volume, random_split_array, val_single_volume, test_lits_single

from lib.networks import PVT_GCASCADE, MERIT_GCASCADE

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str,
                    default='PVT', help='Name of encoder: PVT or MERIT')
parser.add_argument('--skip_aggregation', type=str,
                    default='additive', help='Type of skip-aggregation: additive or concatenation')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--test_save_dir', type=str, default='test_predictions', help='saving prediction as nii!')
parser.add_argument('--seed', type=int, default=32, help='random seed')
parser.add_argument('--is_liver', action='store_true',
                    default=0, help='add for liver, remove for tumor')
parser.add_argument('--root_path', type=str,
                    default='./data/lits/', help='root dir for data')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=2, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--test_log_interval', type=int,
                    default=50, help='Interval for testing set evaluation logging')

args = parser.parse_args()

#WILL MODIFY ONCE INFERENCE FUNCTION WORKS IN TRAINING SCRIPT

def inference_lits(args, model, db_test, test_save_path=None):
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], None
        metric_i = test_lits_single(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        if (i_batch%args.test_log_interval == 0):
            logging.info('idx %d , mean dice %f , mean hd95 %f , mean jacard %f' % (i_batch, np.mean(metric_list, axis=0)[0]/(i_batch + 1), np.mean(metric_list, axis=0)[1]/(i_batch + 1), np.mean(metric_list, axis=0)[2]/(i_batch + 1)))
    metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f ' % (performance, mean_hd95, mean_jacard))
    return "Testing Finished!"


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
    
    if args.encoder=='PVT':
        net = PVT_GCASCADE(n_class=args.num_classes, img_size=args.img_size, k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    elif args.encoder=='MERIT':
        net = MERIT_GCASCADE(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), k=11, padding=5, conv='mr', gcb_act='gelu', skip_aggregation=args.skip_aggregation)
    else:
        print('Implementation not found for this encoder. Exiting!')
        sys.exit()
        
    snapshot = os.path.join(snapshot_path, 'best.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    test_save_path = None#This is for generating train-test-split
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
    
    print("Snapshot path: ", snapshot_path)
    net = net.cuda()
    db_test = LITSTestDataset(X_test,Y_test,transform=None)
    inference_lits(args, net, db_test, test_save_path)


