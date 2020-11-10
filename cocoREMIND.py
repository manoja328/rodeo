import pickle
import utils
from tqdm import tqdm
import numpy as np
import random
import h5py
import time
import argparse
import os
import torch.nn as nn
import torch
from engine import _get_iou_types
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from frcnn_mod import   FastRCNNPredictor
from train_bettercoco import get_model_FRCNN
import math
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from  torch.utils.tensorboard import SummaryWriter
import os.path as osp

# min_size=800, max_size=1333
def batch_images(images, size_divisible=32):
    # concatenate
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    stride = size_divisible
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).zero_()
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs

def get_trainable_params(classifier, start_lr):
    trainable_params = []
    for k, v in classifier.named_parameters():
        trainable_params.append({'params': v, 'lr': start_lr})
    return trainable_params

def set_bn_eval(m):
#    classname = m.__class__.__name__
#    if classname.find('BatchNorm') != -1:
#      m.eval()      
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()


@torch.no_grad()
def evaluate_withpq(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    test_data_pkl = h5py.File('cocoresnet_imagenet_features/backbone.7.0_test_reconstructed.h5', 'r')
    #test_data_pkl = h5py.File('cocoresnet_imagenet_features/backbone.7.0_test.h5', 'r')

    for images, targets in tqdm(data_loader,desc=header):
        images = list(image for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        image_id = targets[0]['image_id'].item()
        quantized_x = test_data_pkl[str(image_id)][()]
        quantized_x = torch.from_numpy(quantized_x)
        imagepq = quantized_x.to(device)

        torch.cuda.synchronize()
        model_time = time.time()
     
        #print ("----",image_id,"----",imagepq.shape)
        outputs = model(images, imagepq, targets)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_evaluator.summarize_per_category()
    torch.set_num_threads(n_threads)

    test_data_pkl.close()
    return coco_evaluator


from voc_loader import imgtransform,pil_loader
class DataIndices():
    def __init__(self, indices, root='./datasets/'):
        self.indices = indices

    def __len__(self):    
        return len(self.indices)

    def __getitem__(self, index):  
        image_id = int(self.indices[index])
        image_path = "/home/manoj/train2014/COCO_train2014_{:012d}.jpg".format(image_id)
        img = pil_loader(image_path)
        return image_id, imgtransform(img)


def fit_one_incremental_batch(model, indices, optimizer):
    model.train()
    #print ("Freezing Batch Norm layers..")
    model.apply(set_bn_eval)
    # train set
    train_data_pkl = h5py.File('cocoresnet_imagenet_features/backbone.7.0_trainval_reconstructed.h5', 'r')
    #train_data_pkl = h5py.File('cocoresnet_imagenet_features/backbone.7.0_trainval.h5', 'r')       
    #makes a data loader out of image_ids
    curr_ds = DataIndices(indices)   
    curr_loader = torch.utils.data.DataLoader(curr_ds, batch_size= 1,
                                              shuffle = True, num_workers= 2)
    
    start_time = time.time()  
    avgloss = []
    for batch,(image_ids,images) in enumerate(curr_loader):
        #get features as respective reconstructions   
        image_ids = image_ids.tolist()
        for image_id in image_ids:
            quantized_x = train_data_pkl[str(image_id)][()]
            quantized_x = torch.from_numpy(quantized_x)
        imagepq = quantized_x.to(device)
        info = {}        
        optimizer.zero_grad()
        #access from the buffer
        targets = [ coco_buffer.buffer[image_id] for image_id in image_ids]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]     
        images = list(image for image in images)     
        loss_dict = model(images, imagepq, targets)
        losses = sum(loss for loss in loss_dict.values())
        avgloss.append(losses.item())
        if torch.isnan(losses):
            print ("Nan encountered!!! ",image_ids)
        losses.backward()
        optimizer.step()
        
        for name in loss_dict:
            info[name] = loss_dict[name].item()
        info['image_id'] = image_ids
        #print (info)
        
    train_data_pkl.close()
    return sum(avgloss) / len(avgloss) 
    
import copy, pickle
from collections import defaultdict, Counter
import torch.nn.functional as F
class MemoryBuffer():
    def __init__(self,dataset,method='unique'):
        self.dataset = dataset
        self.method = method
        self.MAX_BUFFER_SIZE = 17668
        self.dist_each = torch.zeros(self.MAX_BUFFER_SIZE,80)
        self.index2key = [0]*self.MAX_BUFFER_SIZE
        self._initbuffer()


    def get_criterion(self,target):
        #return len(target["labels"])
        return len(target["labels"].unique())

    def save_stats(self,fname):

        cnt = defaultdict(int)
        for e in self.buffer.values():
            labels = e['labels'].tolist()
            for l in labels:
                cnt[l] +=1

        s = [ self.buffer, self.buffer, cnt ]
        with open(fname,'wb') as ff:
            pickle.dump( [self.buffer, self.criterion] ,ff)
        print ("stats saved ...:",fname)


    def add_withreplace(self,target):
        nclasses = 80
        uniform = torch.ones(1,nclasses) / nclasses
        #sum by total to make a p dist
        div = F.kl_div(uniform,self.dist_each,reduction='none').sum(dim=-1)
        index = torch.argmin(div).item()

        image_id =  target['image_id'].item()
        minkey  = self.index2key[index]
        
        self.criterion.pop(minkey)
        self.buffer.pop(minkey)
        
        self.buffer[image_id] = copy.deepcopy(target)
        self.criterion[image_id] = self.get_criterion(target)
        cnt = Counter(target["labels"].tolist())
        for c in cnt:
            self.dist_each[index][c-1] = cnt[c]
        self.dist_each[index]  /=len(target["labels"])
        self.index2key[index] = image_id

    # def add_withreplace(self,target):
    #     #minkey = min( self.criterion , key = self.criterion.get)
    #     minkey = random.sample(self.buffer.keys(),k = 1)[0]
    #     self.criterion.pop(minkey)
    #     self.buffer.pop(minkey)
    #     image_id = int(target['image_id'].item())
    #     self.buffer[image_id] = copy.deepcopy(target)
    #     self.criterion[image_id] = self.get_criterion(target)

    def add(self,target):
        image_id = int(target['image_id'].item())
        self.buffer[image_id] = copy.deepcopy(target)
        self.criterion[image_id] = self.get_criterion(target)
        
    def _initbuffer(self):  
        self.buffer = {}
        self.criterion = {}
        print ("Populating INIT buffer of size.....", self.MAX_BUFFER_SIZE)
        indexes = np.random.choice(range(len(dataset)),self.MAX_BUFFER_SIZE,replace=False)
        for index in tqdm(indexes):
            image, target = self.dataset[index]
            image_id = target['image_id'].item() 
            target["labels"] = target["labels"].long()
            self.buffer[image_id] = copy.deepcopy(target)
            self.criterion[image_id] = self.get_criterion(self.buffer[image_id])
            if len(self.buffer) == self.MAX_BUFFER_SIZE:
                break

        for idx, image_id in enumerate(self.buffer):
            labels = self.buffer[image_id]["labels"].tolist()
            cnt = Counter(labels)
            for c in cnt:
                self.dist_each[idx][c-1] = cnt[c]
            self.dist_each[idx]  /=len(labels)
            self.index2key[idx] = image_id

        assert len(self.dist_each) == len(self.buffer) , "Buffer and dist mismatch!!"


    def replay(self, n):
        #sample always return unique elements
        return random.sample(self.buffer.keys(),k = n)


def get_proposals(incriter, dataset_name='voc'):        
    with open("iter{}_models_incr_voc/info.pkl".format(incriter),"rb") as f:
        info = pickle.load(f)    
    return info

  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, default='iter0_models_incr_coco/chkpt9.pth')
    parser.add_argument('--features_save_dir', type=str, default='resnet_imagenet_features')
    parser.add_argument('--replay', type=int, default = 4)
    parser.add_argument('--bs', type=int, default=2)
    args = parser.parse_args()
    return args

class ResNet50_StartAt_Layer4_1(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        #get last block , remove 0th
        self.chopped = core_model.backbone[-1][1:]
        
    def forward(self, x):
        return  self.chopped(x)

def get_rundir(dirs): 
    if not osp.exists(dirs):
        return osp.join('log','run_%02d' % 0) 
    previous_runs = os.listdir(dirs)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    
    return osp.join('log','run_%02d' % run_number) 


# setup log data writer
RUNDIR = get_rundir('log')


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.backends.cudnn.deterministic = False   
    
    
    from train_bettercoco import half_incr,dpr_to_normal
    
        
    print('\nBeginning streaming training...')       
    args = get_args()  
    print (args)
    model = get_model_FRCNN(num_classes = 41)
    if os.path.exists(args.ckpt_file):
        print ("Reusing last checkpoint from phase:",args.ckpt_file)
        load_tbs = utils.load_checkpoint(args.ckpt_file)
        model.load_state_dict(dpr_to_normal(load_tbs['state_dict']))
    else:
        print (args.ckpt_file, "  half checkpoint not found ....")
        exit(1)

        
#%%    
    #chopped_backbone = get_chopped(core_model)
    chopped_backbone = ResNet50_StartAt_Layer4_1(model)
    model.backbone = chopped_backbone
    print (model)
    model.to(device) 

    datasets = half_incr()

    writer = SummaryWriter(log_dir = RUNDIR)
    
    for incriter,(num_classes, dataset, dataset_test) in enumerate(datasets):      
        
        if incriter == 0:
            coco_buffer = MemoryBuffer(dataset)
                   
        else:
            #get fc data from previous model
            fc_data =  model.roi_heads.box_predictor.state_dict()
            new_box_predictor = FastRCNNPredictor(1024,num_classes)            
            for key in fc_data:
                ndim = fc_data[key].data.ndim
                s = fc_data[key].shape
                if ndim == 1:
                    new_box_predictor.state_dict()[key].data[:s[0]] = fc_data[key].detach()
                else:
                    new_box_predictor.state_dict()[key].data[:s[0],:s[1]] = fc_data[key].detach()
            new_box_predictor = new_box_predictor.to(device)
            model.roi_heads.box_predictor = new_box_predictor


        # if incriter + 40 <= 60: # lr change here 
        #     lr = 0.001
        # elif incriter + 40 > 60:
        #     lr = 0.0001
               
        trainable_params = get_trainable_params(model, start_lr = 0.001 )
        optimizer = torch.optim.SGD(trainable_params,momentum = 0.9, nesterov=True)              
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size= 1 , shuffle = False,
            num_workers = 2,collate_fn=utils.collate_fn)
        
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size= 1, shuffle=False,
            num_workers= 2,collate_fn=utils.collate_fn)
        
        
        if incriter == 0:
            print ("evaluating base init performance....")
            #evaluate_withpq(model, data_loader_test, device = device)  
            coco_buffer.save_stats("{}.pkl".format(incriter+40))
            continue
     
        print(" ------------------------------", incriter)
           
        for phaseiter,(images, targets) in tqdm(enumerate(data_loader),total=len(data_loader)): 
            
            #since streaming .. only 1 entry always            
            image_id = targets[0]['image_id'].item()  
                        
            if image_id in coco_buffer.buffer:
                #update the annotations for that image_id
                old_target = coco_buffer.buffer[image_id]
                for key in old_target:
                    #coz there is only one image_id
                    if key == 'image_id' or key == 'size':
                        continue
                    #append to previous box and their labesl
                    
                    val = old_target[key].tolist() + targets[0][key].tolist()
                    old_target[key] = torch.Tensor(val)
                    if key == 'labels':
                        old_target[key] = old_target[key].long()
            else: #if imageid not there
                #put the new sample in buffer
                coco_buffer.add_withreplace(targets[0])
                # coco_buffer.add(targets[0])
                #add new image-annotations too

            # if incriter + 40 <=90:
            #     continue
            X = [image_id]            
            replay_samples = coco_buffer.replay(n = args.replay - 1)
            X_and_replay_samples =  X + replay_samples
            #print (incriter,phaseiter)
            loss = fit_one_incremental_batch(model,list(X_and_replay_samples), optimizer)
            writer.add_scalar("losses_{}".format(incriter+40), loss,   phaseiter)

        coco_buffer.save_stats("{}.pkl".format(incriter+ 40))   

        if incriter + 40 in {50,60,70,80}: 
            tbs = {'phase': incriter,
                   'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict()}
            
            MODELDIR ="iter_replay{}_{}_coco_{}".format(args.replay,incriter,'remind')            
            chkptname = os.path.join(MODELDIR,"chkpt.pth")
            utils.save_checkpoint(tbs,checkpoint = chkptname)   

            # evaluate on the test dataset
            evaluate_withpq(model, data_loader_test, device = device)  

    