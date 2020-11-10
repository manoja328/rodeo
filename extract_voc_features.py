import os
import numpy as np
import  utils
import h5py
import argparse
from REMIND.retrieve_any_layer import ModelWrapper
import torch
from tqdm import tqdm
from voc_loader import VOC
from train_better import get_model_FRCNN
from engine import evaluate

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, default='iter0_models_incr_voc/chkpt24.pth')
    parser.add_argument('--features_save_dir', type=str, default='resnet_imagenet_features')
    parser.add_argument('--extract_features_from', type=str,default='backbone.7.0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=2048)
    parser.add_argument('--num_feats', type=int, default=7)
    args = parser.parse_args()
    return args


#def extract_features(model, data_loader, h5_file_full_path, data_len=None, num_channels=512, num_feats=7):
#    if data_len is None:
#        data_len = len(data_loader.dataset)  
#    h5_file = h5py.File(h5_file_full_path, 'w')
##    h5_file.create_dataset('image_features', (data_len, num_channels, num_feats, num_feats), dtype=np.float32)
#    h5_file.create_dataset('image_id', (data_len, 1), dtype=np.int)
#    with torch.no_grad():
#        for batch_ix, (images, targets) in tqdm(enumerate(data_loader),total=len(data_loader)):
#            images = list(image.to(device) for image in images)                       
#            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#            batch_feats = model(images,targets)
#            print (batch_ix,batch_feats.shape)
##            h5_file['image_features'][batch_ix] = batch_feats.cpu().numpy()
##            ids = [t['image_id'] for t in targets]
#            h5_file['image_id'][batch_ix] = int(targets[0]['image_id'].item())
#            if batch_ix == 10:
#                break 
#    h5_file.close()
    
def extract_features(model, data_loader, h5_file_full_path, data_len=None, num_channels=512, num_feats=7):
    if os.path.exists(h5_file_full_path):
        print ("file exists")
        return 
    h5_file = h5py.File(h5_file_full_path, 'w')
    with torch.no_grad():
        for batch_ix, (images, targets) in tqdm(enumerate(data_loader),total=len(data_loader)):
            images = list(image.to(device) for image in images)                       
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            batch_feats = model(images,targets)
            image_id = str(int(targets[0]['image_id'].item()))
            features =  batch_feats.cpu().numpy()
            print (features.shape)
            h5_file.create_dataset(image_id,data=features)
    h5_file.close()
    
if __name__ == '__main__':
    args = get_args()

    classifier_ckpt = os.path.join(args.ckpt_file)
    
    #just get res 50?
    #no cannot do since we need to load chekcpoint as well
    core_model = get_model_FRCNN(num_classes = 11)

    if os.path.exists(classifier_ckpt):
        print ("Reusing last checkpoint from phase:")
        print (classifier_ckpt)
        load_tbs = utils.load_checkpoint(classifier_ckpt)
        core_model.load_state_dict(load_tbs['state_dict'])
        dataset_test = VOC('07', 'edgeboxes', 'test',included=[0,1,2,3,4,5,6,7,8,9,10])
        base_val_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size = 3, shuffle=False,
                num_workers=1,collate_fn=utils.collate_fn)
        core_model.to(device)
        evaluate(core_model, base_val_loader, device=device) 
        #eval the  checkpoint to verify


    model = ModelWrapper(core_model, output_layer_names=[args.extract_features_from], return_single=True)

    model.eval()
    model.to(device)
    
    dataset = VOC('07', 'selective_search', 'trainval')
    dataset_test = VOC('07', 'selective_search', 'test')

    # define training and validation data loaders
    base_train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=1,collate_fn=utils.collate_fn)


    base_val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=1,collate_fn=utils.collate_fn)


    features_save_dir = args.features_save_dir
    if not os.path.exists(features_save_dir):
        os.makedirs(features_save_dir)
        
    extract_features(model, base_train_loader,
                     os.path.join(args.features_save_dir , args.extract_features_from + "_trainval.h5"),
                     len(base_train_loader.dataset),
                     num_channels=args.num_channels, num_feats=args.num_feats)
    
    extract_features(model, base_val_loader,
                     os.path.join(args.features_save_dir, args.extract_features_from + "_test.h5"),
                     len(base_val_loader.dataset),
                     num_channels=args.num_channels, num_feats=args.num_feats)



