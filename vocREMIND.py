import pickle
import utils
from tqdm import tqdm
import random
import h5py
import time
import argparse
import os
import os.path as osp
import torch.nn as nn
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
from torch.utils.data import TensorDataset
from engine import _get_iou_types
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from data_pardigm import data_dict
from  torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.transform import resize_boxes
from frcnn_mod import ModifiedFasterRCNN , FastRCNNPredictor
from train_better import get_model_FRCNN
import math
from PIL import Image

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

    test_data_pkl = h5py.File('resnet_imagenet_features/backbone.7.0_test_reconstructed.h5', 'r')
    #test_data_pkl = h5py.File('resnet_imagenet_features/backbone.7.0_test.h5', 'r')


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
        self.root = root + 'voc/VOCdevkit/VOC2007/'
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.indices = indices


    def __len__(self):    
        return len(self.indices)

    def __getitem__(self, index):  
        image_id = int(self.indices[index])
        image_path = os.path.join(self.image_dir, '{0:06d}.jpg'.format(image_id))
        img = pil_loader(image_path)
        return image_id, imgtransform(img)


def fit_one_incremental_batch(model, indices, optimizer):
    model.train()
    #print ("Freezing Batch Norm layers..")
    model.apply(set_bn_eval)
    # train set
    #train_data_pkl = h5py.File('resnet_imagenet_features/backbone.7.0_trainval.h5', 'r')
    train_data_pkl = h5py.File('resnet_imagenet_features/backbone.7.0_trainval_reconstructed.h5', 'r')       
    #makes a data loader out of image_ids
    curr_ds = DataIndices(indices)   
    curr_loader = torch.utils.data.DataLoader(curr_ds, batch_size= 1,
                                              shuffle = True, num_workers= 2)
    
    start_time = time.time()  
    for batch,(image_ids,images) in enumerate(curr_loader):
        #get features as respective reconstructions   
               
#        imagepq = []
        image_ids = image_ids.tolist()
        for image_id in image_ids:
            quantized_x = train_data_pkl[str(image_id)][()]
            quantized_x = torch.from_numpy(quantized_x)
#            print (quantized_x.shape)
#            if quantized_x.ndim == 4:
#                quantized_x = quantized_x.squeeze(0)
#            imagepq.append(quantized_x)   
      
        imagepq = quantized_x.to(device)
#        images_res = batch_images(imagepq) 
#        images_res = images_res.to(device)

        info = {}        
        #print ("----",image_ids,"----")

        optimizer.zero_grad()
        #access from the buffer
        targets = [ voc_buffer[image_id] for image_id in image_ids]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]     
        images = list(image for image in images)     
        loss_dict = model(images, imagepq, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        for name in loss_dict:
            info[name] = loss_dict[name].item()
        info['image_id'] = image_ids
        #print (info)
        
    train_data_pkl.close()
    

def get_initbuffer(dataset):  
    voc_buffer = {}
    print ("Populating INIT buffer.....")           
    for image , target in dataset:
        image_id = target['image_id'].item() 
        voc_buffer[image_id] = target
        target["labels"] = target["labels"].long()
    return voc_buffer


def rehearsalSampler(buffer, n):
    return random.sample(voc_buffer.keys(),k = n)


def get_proposals(incriter, dataset_name='voc'):        
    with open("iter{}_models_incr_voc/info.pkl".format(incriter),"rb") as f:
        info = pickle.load(f)    
    return info

  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, default='iter0_models_incr_voc/chkpt24.pth')
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

#from collections import OrderedDict
#class ResNet50_StartAt_Layer4_2(nn.Module):
#    def __init__(self, model):
#        super().__init__()
#        #get last block , remove 0th
#        last_block = model.backbone[-1][1:]
#        odict = OrderedDict(list(last_block[0].named_children())[2:])
#        chopped_a = nn.Sequential(odict)        
#        chopped_b = last_block[1]
#        
#        chopped =  OrderedDict({'1':chopped_a,'2':chopped_b})
#        self.chopped = nn.Sequential(chopped)
#        
#        #get fc data from previous model
#        old_dict  =  last_block.state_dict()    
#        new_dict = self.chopped.state_dict()
#        for key in new_dict:       
#            new_dict[key].data = old_dict[key].data.detach()
#        
#    def forward(self, x):
#        return self.chopped(x)

#from collections import OrderedDict
#import copy
#class ResNet50_StartAt_Layer4_2(nn.Module):
#    def __init__(self, model):
#        super().__init__()
#        #get last block , remove 0th
#        last_block = model.backbone[-1][1:]        
#        copied = copy.deepcopy(last_block)
#        copied[0].conv1 =  nn.Identity()
#        copied[0].bn1 = nn.Identity()
#        self.chopped = copied
#
#    def forward(self, x):
#        return self.chopped(x)


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)
    torch.backends.cudnn.deterministic = False   
        
    print('\nBeginning streaming training...')       
    args = get_args()  
    print (args)
    model = get_model_FRCNN(num_classes = 11)
    if os.path.exists(args.ckpt_file):
        print ("Reusing last checkpoint from phase:",args.ckpt_file)
        load_tbs = utils.load_checkpoint(args.ckpt_file)
        model.load_state_dict(load_tbs['state_dict'])
          
#%%    
    #chopped_backbone = get_chopped(core_model)
    chopped_backbone = ResNet50_StartAt_Layer4_1(model)
    model.backbone = chopped_backbone
    print (model)
    model.to(device)                                  
    datasets = data_dict['incr_voc']()
    for incriter,(num_classes, dataset, dataset_test) in enumerate(datasets):      
        
        if incriter == 0:
            voc_buffer = get_initbuffer(dataset)
                   
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
            print ("evaluating base inint performance....")
            evaluate_withpq(model, data_loader_test, device = device)  
            continue
        
        print(" ------------------------------", incriter)
           
        for phaseiter,(images, targets) in tqdm(enumerate(data_loader),total=len(data_loader)): 
            #since streaming .. only 1 entry always            
            image_id = targets[0]['image_id'].item()  
            
            if image_id in voc_buffer:
                #update the annotations for that image_id
                old_target = voc_buffer[image_id]
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
                voc_buffer[image_id] = targets[0]

            X = [image_id]
            
            replay_samples= rehearsalSampler(voc_buffer, n = args.replay - 1)

            X_and_replay_samples =  X + replay_samples
            
            #print (incriter,phaseiter)
            fit_one_incremental_batch(model,X_and_replay_samples, optimizer)
                 
        # evaluate on the test dataset
        evaluate_withpq(model, data_loader_test, device = device)  
        






