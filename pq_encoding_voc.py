import faiss
import numpy as np
import h5py
import math
from tqdm import tqdm
import data_pardigm

d = 2048  # data dimension
cs = 128  # code size (bytes)

# train set
train_data_pkl = h5py.File('resnet_imagenet_features/backbone.7.0_trainval.h5', 'r')

#%%

for n,train,test in data_pardigm.train_test_half():
    pass

imgids = [ str(int(image.split("/")[-1].replace(".jpg","")))  for image in train.images]

keys = list(train_data_pkl.keys())

#%%
# Divide the training data into batches
train_data_base_init = []
# Put data in array
start_ix = 0
data_batch = []
for idx,key in enumerate(tqdm(imgids)):
    data = train_data_pkl[key][()]
    print (data.shape[-2:])
    data_tr =   np.transpose(data, (0, 2, 3, 1)).reshape(-1,d).astype("float32")   
    train_data_base_init.append(data_tr)
#
train_data_base_init = np.concatenate(train_data_base_init)
train_data_base_init = np.ascontiguousarray(train_data_base_init, dtype=np.float32)
print('Data loading done ..........')
#%%

print('Training Product Quantizer..........')
nbits = 8 #int(np.log2(d))
print ("nbits:",nbits)
pq = faiss.ProductQuantizer(d, cs, nbits)
pq.train(train_data_base_init)

print('Encoding, Decoding and saving Reconstructed Features..........')

#%%

for fname in ['backbone.7.0_test','backbone.7.0_trainval']:
    in_fname = 'resnet_imagenet_features/'+ fname + '.h5'
    data_h5 = h5py.File(in_fname, 'r')
    h5_file = fname + '_reconstructed'    
    reconstructed_h5 = h5py.File(f'resnet_imagenet_features/{h5_file}.h5', 'w')
    keys = list(data_h5.keys())
    for idx,key in enumerate(tqdm(keys)):
        data = data_h5[key][()]
        _,dim,r,c = data.shape
        assert dim == d, "dimension error"
        data_tr =   np.transpose(data, (0, 2, 3, 1)).reshape(-1,d) #r,c,d    
        codes = pq.compute_codes(np.ascontiguousarray(data_tr))
        codes = pq.decode(codes)
        codes = np.reshape(codes,(-1,r,c,dim)) #r,c,d
        data_batch_reconstructed = np.transpose(codes, (0,3,1,2))
        assert data.shape == data_batch_reconstructed.shape, "recons error"
        reconstructed_h5.create_dataset(key,data = data_batch_reconstructed)
    data_h5.close()
    reconstructed_h5.close()
