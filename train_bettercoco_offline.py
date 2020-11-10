# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

#benchmark reference on VOC
#https://github.com/jwyang/faster-rcnn.pytorch

import os
import torch
import torch.nn as nn
import torchvision
from engine import evaluate
import utils
import transforms as T
from tqdm import tqdm
from  torch.utils.tensorboard import SummaryWriter
import pickle
from torchvision.models.detection.transform import resize_boxes
from opt import parse_args

from frcnn_mod import ModifiedFasterRCNN , FastRCNNPredictor
import os.path as osp
#%%

def get_transform(istrain=False):
     transforms = []
     transforms.append(T.ToTensor())
     if istrain:
         transforms.append(T.RandomHorizontalFlip(0.5))
     return T.Compose(transforms)

class BoxHead(nn.Module):
    def __init__(self, vgg):
        super(BoxHead, self).__init__()
        self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.in_features = 4096 # feature out from mlp

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class FakeRegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        print (" ----- Using fake region proposal boxes -----")
        with open("datasets/edboxes_coco2014trainval_2000.pkl","rb") as f:
            self.edgeboxes = pickle.load(f)


    def forward(self, images, features, targets=None):
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        num_images = len(images.tensors)
        device = images.tensors.device
                    
        proposals = []
        for idx in range(num_images):
            image_id = int(targets[idx]['image_id'].item())
            orig_size = targets[idx]["size"]
            new_size = images.image_sizes[idx]
            box = self.edgeboxes[image_id]
            box = torch.Tensor(box).float()
            box = resize_boxes(box,orig_size,new_size)
            box = box.to(device)
            proposals.append(box)

        boxes = proposals
        losses = {}
        return boxes, losses



def get_model_FRCNN(num_classes):

    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(res50_model.children())[:-2])
    backbone.out_channels = 2048
    
#    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#    backbone.out_channels = 1280   
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here    

       
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = None
    
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    

    
    model = ModifiedFasterRCNN(backbone, num_classes,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler)
   
    model.rpn = FakeRegionProposalNetwork()
    
    return  model


def get_distillinfo(model,dl):   
    save = {}
    print ("dumping info ......")
    model.eval()
    with torch.no_grad():
        for ii, (images, targets) in tqdm(enumerate(dl),total=len(dl)):   
           images = list(image.to(device) for image in images)
           targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           for image,target in zip(images,targets):
               image_id = '{0:06d}'.format(target['image_id'].item())
               info = model.get_data128([image], [target]) 
               save[image_id] = info  
        return save

def save_distillinfo(obj,file):    
    dirn = os.path.dirname(file)
    if not os.path.exists(dirn):
        os.mkdir(dirn)
    with open(file,"wb") as f:
        pickle.dump(obj,f)


from coco_loader import COCOLoader
def getds(split='val2014'):
    DATASETS_ROOT = './datasets'
    root = '/home/manoj/%s' % (split)
    annFile = '%s/coco/annotations/instances_%s.json' % (DATASETS_ROOT,split)  
    return root,annFile


def get_classes():
    base  =  [ list(range(0,41)) ]
    one_class =  base + [ [i] for i in range(41,81)]
    fivepts =  base + [[*range(41,51)]] + [[*range(51,61)]] +\
                  [[*range(61,71)]] + [[*range(71,81)]]
                  
        #return base + [ [i] for i in range(41,81)] #incremental
    two =  base + [[*range(41,81)]] #offlien half and last half                 
    return one_class             
                

def half_incr():        
    classes = get_classes()
    included = []
    num_classes = 0
    for c in classes:
        included += c
        num_classes = len(included)
        print ("-------------------------------------------")
        print ("{} classes: {}".format(num_classes,c))
        root,annFile = getds('train2014')
        dataset =    COCOLoader(root,annFile,included = c)
        root,annFile = getds('val2014')       
        dataset_test = COCOLoader(root,annFile,included = included)
            
        print('data prepared, train data: {}'.format(len(dataset)))
        print('data prepared, test data: {}'.format(len(dataset_test)))
    
        yield num_classes, dataset, dataset_test   

def offline():
        
    classes =  [[*range(0,81)]] #offlien half and last half       
    included = []
    num_classes = 0
       
    for c in classes:
        included += c
        num_classes = len(included)
        print ("-------------------------------------------")
        print ("{} classes: {}".format(num_classes,c))
        root,annFile = getds('train2014')
        dataset =    COCOLoader(root,annFile,included = included)
        root,annFile = getds('val2014')       
        dataset_test = COCOLoader(root,annFile,included = included)
            
        print('data prepared, train data: {}'.format(len(dataset)))
        print('data prepared, test data: {}'.format(len(dataset_test)))
    
        yield num_classes, dataset, dataset_test   
    

    
def load_distillinfo(file):    
    print ("loading distill infos....")
    with open(file,"rb") as f:
        return pickle.load(f)
        
def set_bn_eval(m):
#    classname = m.__class__.__name__
#    if classname.find('BatchNorm') != -1:
#      m.eval()      
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
 
def get_rundir(dirs): 
    if not osp.exists(dirs):
        return osp.join('log','run_%02d' % 0) 
    previous_runs = os.listdir(dirs)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    
    return osp.join('log','run_%02d' % run_number) 



def dpr_to_normal(state_dict):
    #data parallel adds module. at the state dict names so torch.load gives error
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
    
        
#%%    
if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    args = parse_args()    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=False   

    # setup log data writer
    RUNDIR = get_rundir('log')
    writer = SummaryWriter(log_dir = RUNDIR)
    
    data_dict = ['incr_coco','incr_finetune_coco','offline_coco']

                  
    args.dataset = 'COCO'
    args.exp = 'COCO'
    args.dpr == 'offline_coco_full'
    datasets = offline()
        
    num_epochs = 10   
    
    for incriter,(num_classes, dataset, dataset_test) in enumerate(datasets):
        
        print ("-----------------Iteration: -----------------",incriter)
        
        MODELDIR ="iter{}_models_{}".format(incriter,args.dpr)  
        finalchkpt = osp.join(MODELDIR,"chkpt{}.pth".format(num_epochs-1))
        

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size = 2, shuffle=True,
            num_workers=args.workers,collate_fn=utils.collate_fn)
    
    
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size = 2, shuffle=False,
            num_workers=args.workers,collate_fn=utils.collate_fn)
    
    
        lr = args.lr
        # get the model using our helper function
        model = get_model_FRCNN(num_classes)
        model = torch.nn.DataParallel(model)
        # move model to the right device
        model.to(device)
        #make sure to say its base
        model.base = True             

                 
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr = lr,momentum = 0.9,
                                    weight_decay = 0.00005, nesterov=True)
        
        
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=30,
                                                       gamma=0.1)
    

    #%%    
        iters_per_epoch = int( len(data_loader) / data_loader.batch_size)
        freezebn = True
        if freezebn:
            print ("[[.... Freezing Batch Norm layers....]]")
        for epoch in range(num_epochs):
            
            model.train()
            if freezebn: model.apply(set_bn_eval)
            warm_lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1. / 1000
                warmup_iters = min(1000, len(data_loader) - 1)
                warm_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
            loss_epoch = {}
            header = 'Phase[{}] Epoch: [{}/{}]'.format(incriter,epoch,num_epochs)
            loss_name = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
            for ii, (images, targets) in tqdm(enumerate(data_loader),total=len(data_loader),desc = header):  
               optimizer.zero_grad()
               images = list(image.to(device) for image in images)
               targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
               # training
               loss_dict = model(images, targets)
               losses = sum(loss for loss in loss_dict.values())
               losses.backward()
               optimizer.step()
               if warm_lr_scheduler is not None:
                    warm_lr_scheduler.step()
               info = {}
               for name in loss_dict:
                   info[name] = loss_dict[name].item()
                   
               writer.add_scalars("losses", info, epoch * iters_per_epoch + ii)
               
               
            if 'incr' in args.dpr:
                if incriter+40 in {50,60,70,80}:
                    evaluate(model, data_loader_test, device=device)  
            else:
                if incriter == 0: #save all checkpoints
                    # Save weights
                    tbs = {'epoch': epoch,
                           'state_dict': model.state_dict(),
                           'optim_dict': optimizer.state_dict()}
                    
                    chkptname = osp.join(MODELDIR,"chkpt{}.pth".format(epoch))
                    print(chkptname)
                    utils.save_checkpoint(tbs,checkpoint = chkptname)
                    
                if (epoch + 1 ) % 4 == 0  or epoch + 1 == num_epochs:    
                    # evaluate on the test dataset
                    evaluate(model, data_loader_test, device=device)   

                      
            lr_scheduler.step()        
        # Save weights
        tbs = {'epoch': epoch,
               'state_dict': model.state_dict(),
               'optim_dict': optimizer.state_dict()}
        
        chkptname = osp.join(MODELDIR,"chkpt{}.pth".format(epoch))
        utils.save_checkpoint(tbs,checkpoint = chkptname)   
        
                      
    writer.close()                
         

    
