import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from skimage import io
import os
from torchvision import datasets, transforms
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
# from changedetection.utils import get_binary_change_map

def get_all_files(path:str, file_type=None)->list:
    """Returns all the files in the specified directory and its subdirectories.
    
    e.g 2021/s1_imgs/ will return all the files in `2021/s1_imgs/` subfolders which are `train` and `test`
    
    it will return the names like `train/2021_01_01.tif` and `test/2021_01_01.tif` if subfolders are present
    if not it will return the names like `2021_01_01.tif`
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file_type is None:
                file_list.append(os.path.relpath(os.path.join(root, file), path))
            elif file.endswith(file_type):
                file_list.append(os.path.relpath(os.path.join(root, file), path))
            
    return file_list

def find_difference(list1, list2):
    """Find the difference between two lists."""
    # Use list comprehension to find elements in list1 that are not in list2
    difference1 = [item for item in list1 if item not in list2]
    
    # Use list comprehension to find elements in list2 that are not in list1
    difference2 = [item for item in list2 if item not in list1]
    
    # Concatenate the two differences to get the complete list of different elements
    result = difference1 + difference2
    
    return result


class RGBNirDataset(Dataset):
    """Dataset class for RGB-NIR image pairs."""
    def __init__(self,
               rgbn_dir: str,
               transform = None,

               verbose=False):


        self.verbose = verbose
        # Set the directories for the four sets of images
        self.rgbn_dir = rgbn_dir


        self.file_names= get_all_files(self.rgbn_dir + "//")
        self.file_names.sort()

        
        self.transform = transform


    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.file_names)

    def __getitem__(self, index):
        """Return the index-th items from the dataset."""

        img_name = self.file_names[index] 

        if self.verbose: print(f"Image name: {img_name}")  
        
        rgbn_img_path = os.path.join(self.rgbn_dir,img_name)
        rgbn_img = io.imread(rgbn_img_path)
        
        if self.transform:
            rgbn_img = self.transform(rgbn_img)

        
        
        rgb = rgbn_img[:3,:,:]
        # nir = rgbn_img[:,:,3] # cant retrive last band using single index results in 2d array
        nir = rgbn_img[3:,:,:] # retrive last band using slice index results in 3d array 


        if self.verbose: print(f"RGB shape: {rgb.shape}")
        if self.verbose: print(f"NIR shape: {nir.shape}")

        check_tensor_values([rgb, nir],
                            ["rgb", "nir"])
        
        

        return rgb, nir

def check_tensor_values(tensor_list, input_names):
    for i, tensor in enumerate(tensor_list):
        if torch.any(tensor > 1) or torch.any(tensor < -1):
            input_name = input_names[i]
            raise ValueError(f"Values of {input_name} tensor must be between -1 and 1.")
        
#########################################################################################################################################
###################################################### Transfroms #######################################################################
#########################################################################################################################################      
      
        
class myToTensor:
    """Transform a pair of numpy arrays to PyTorch tensors"""
    def __init__(self,dtype=torch.float16):
        """Transform a pair of numpy arrays to PyTorch tensors
        
        Args
        ---
            `dtype` (torch.dtype): Data type for the output tensor (default: torch.float16)
        """
        self.dtype = dtype
        
    def reshape_tensor(self,tensor):
        """Reshape a 2D or 3D tensor to the expected shape of pytorch models which is (channels, height, width)
        
        Args
        ---
            `tensor`(numpy.ndarray): Input tensor to be reshaped
        
        Returns:
            torch.Tensor: Reshaped tensor
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]:
            tensor = tensor.permute((2,0,1)) # Channels first
        elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]:
            pass
        else:
            raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
        return tensor

    def __call__(self,sample):
        return self.reshape_tensor(torch.from_numpy(sample)).to(dtype=self.dtype)

class Normalize:
    

    def __init__(self, min=0, max=255, check_nan=True, fix_nan=False):
  
        self.min = min
        self.max = max
        self.check_nan = check_nan
        self.fix_nan = fix_nan

    def __call__(self, img):

        # Sentinel 2 image  is between 0 and 1 it is surface reflectance so it can't be more than 1 or less than 0
        img[img >= self.max] = self.max - 0.0001
        img[img <= self.min] = self.min + 0.0001

        # Normalize the image Between -1 and 1
        img = (img - self.min) / (self.max - self.min)
        img = img * 2 - 1
        

        if self.check_nan:
            if np.isnan(img).any():
                raise ValueError("s2_img contains NaN values")
        elif self.fix_nan:
            if np.isnan(img).any():
                img[np.isnan(img)] = 0.0

        return img




if __name__ == "__main__":
    from utils.plot_utils import *
    
    transform = transforms.Compose([Normalize(),myToTensor()])
    
    print("Testing Dataset...")
    s1s2_dataset = RGBNirDataset("D:\\python\\Palm-Segmentation-Phase2\\dataset\\rgbn_dataset_patched",transform=transform,verbose=False)
    print(f"Dataset length: {len(s1s2_dataset)}")
    rgb, nir = s1s2_dataset[0]
    print(f"RGB shape: {rgb.shape} | NIR shape: {nir.shape}")
    print(f"RGB dtype: {rgb.dtype} | NIR dtype: {nir.dtype}")
    print(f"RGB min: {rgb.min()} | RGB max: {rgb.max()}")
    print(f"NIR min: {nir.min()} | NIR max: {nir.max()}")
    print(f"RGB mean: {rgb.mean()} | RGB std: {rgb.std()}")
    print(f"NIR mean: {nir.mean()} | NIR std: {nir.std()}")


  