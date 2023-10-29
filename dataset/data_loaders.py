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


class Sen12Dataset(Dataset):
    """Dataset class for the Sen12MS dataset."""
    def __init__(self,
               s1_dir,
               s2_dir,
               crop_map_dir,
               s2_bands: list = None ,
               s1_transform = None,
               s2_transform = None,
               crop_map_transform = None,
               verbose=False):
        """
        Args
        ---
            `s1_t2_dir` (str): Path to the directory containing the S1 time-2 images.
            `s2_t2_dir` (str): Path to the directory containing the S2 time-2 images.
            `s1_t1_dir` (str): Path to the directory containing the S1 time-1 images.
            `s2_t1_dir` (str): Path to the directory containing the S2 time-1 images.
            `s2_bands` (list, optional): List of indices indicating which bands to use from the S2 images.
                                       If not specified, all bands are used.
            `transform` (callable, optional): Optional transform to be applied to the S2 images.
           ` hist_match` (bool, optional): Whether to perform histogram matching between the S2 time-2 
                                         and S2 time-1 images.
            `two_way` (bool, optional): is used to determine whether to return images from both time directions (i.e. time-2 to time-1 and time-1 to time-2). If two_way=True, __len__ returns twice the number of images in the dataset, with the first half of the indices corresponding to the time-2 to time-1 direction and the second half corresponding to the time-1 to time-2 direction.
            `binary_s1cm` (bool, optional): Whether to return the binary change map for the S1 images.
            
        
        __getitem__ Return
        ---
            `(s2_t2_img, s1_t2_img, s2_t1_img, s1_t1_img, diff_map, reversed_diff_map)` or `(s2_t1_img, s1_t1_img, s2_t2_img, s1_t2_img, diff_map, reversed_diff_map)` if reversed index
             
            tuple: A tuple containing the `S2 time-2` image, `S1 time-2` image, `S2 time-1` image, `S1 time-1` image, Difference map and Reversed difference map.
            * `Difference map`: `np.abs(s2_t2_img - s2_t1_img)`
            * `Reversed difference map`: `np.max(diff_map) - diff_map + np.min(diff_map) `
            *  when reversed is activated the index `i` in a reversed mode has the index `len(dataset) - i`
        """
        self.verbose = verbose
        # Set the directories for the four sets of images
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.crop_map_dir = crop_map_dir
 
        
        # Get the names of the S2 and S1 time-2 images and sort them
        self.s1_dates = os.listdir(s1_dir)
        self.s1_dates.sort()
        self.num_dates = len(self.s1_dates)
        self.s2_dates = os.listdir(s2_dir)
        self.s2_dates.sort()
        self.file_names= get_all_files(self.s1_dir + "//" + self.s1_dates[0])
        self.file_names.sort()

        assertion_names = get_all_files(self.s2_dir + "//" + self.s2_dates[3])
        assertion_names.sort()
        crop_map_names = get_all_files(crop_map_dir)
        crop_map_names.sort()
        
        
        # Verify that the four sets of images have the same names
        if self.s1_dates != self.s2_dates:
            diff = find_difference(self.s1_dates, self.s2_dates)
            raise ValueError(f"S1 and S2 directories do not contain the same dates | Diff Len: {len(diff)} | Diffrencce: {diff}")
        if self.file_names != assertion_names:
            diff = find_difference(self.file_names, assertion_names)
            raise ValueError(f"S2 date directories do not contain the same image pairs | Diff Len: {len(diff)} | Diffrencce: {diff}")
        if self.file_names != crop_map_names:
            diff = find_difference(self.file_names, crop_map_names)
            raise ValueError(f"S1 and S2 directories do not contain the same image pairs as Cropmap | Diff Len: {len(diff)} | Diffrencce: {diff}")
        
        
        self.s2_bands = s2_bands if s2_bands else None 

        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.crop_map_transform = crop_map_transform


    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.file_names)

    def __getitem__(self, index):
        """Get the S2 time-2 image, S1 time-2 image, S2 time-1 image, S1 time-1 image, 
           difference map and reversed difference map for the specified index.
           
        Args:
            index (int): Index of the image to get.
            
        Returns:
            tuple: A tuple containing the S2 time-2 image, S1 time-2 image, S2 time-1 image, S1 time-1 image, Difference map and Reversed difference map.
            * Difference map: `s2_t2_img - s2_t1_img` Values are in the range [-1, 1].
            * Reversed difference map: 1 - torch.abs(diff_map)  Values are in the range [0, 1].
            * S1_abs_diff_map: `abs(s1_t2_img - s1_t1_img)` Values are in the range [0, 1].
        """


        img_name = self.file_names[index] 

        if self.verbose: print(f"Image name: {img_name}")  
        
        s1_images = []
        s2_images = []
        
        for d in self.s1_dates:
            s1_img_path = os.path.join(self.s1_dir,d,img_name)
            s1_img = io.imread(s1_img_path)
            s1_images.append(s1_img)
            s2_img_path = os.path.join(self.s2_dir,d,img_name)
            s2_img = io.imread(s2_img_path)
            if self.s2_bands: s2_img = s2_img[self.s2_bands,:,:]
            s2_images.append(s2_img)
            
        crop_map_path = os.path.join(self.crop_map_dir,img_name)
        crop_map = io.imread(crop_map_path)
        if self.verbose: print(f'crop_map shape apon reading: {crop_map.shape}')
            
        if self.verbose: print(f's2 shape apon reading: {s2_img.shape}')
        if self.verbose: print(f's1 shape apon reading: {s1_img.shape}')
        
        if self.s1_transform:
            s1_images = [self.s1_transform(s1_img) for s1_img in s1_images]
        if self.s2_transform:
            s2_images = [self.s2_transform(s2_img) for s2_img in s2_images]
        if self.crop_map_transform:
            crop_map = self.crop_map_transform(crop_map)
        if self.verbose: print(f'crop_map shape apon transform: {crop_map.shape}')
                
        
        # Stack images for 3D Convolution to shape (channels, depth, height, width)
        s1_img = torch.stack(s1_images , dim=1)
        s2_img = torch.stack(s2_images , dim=1)
        
        if self.verbose: print(f's2 shape apon stacking: {s2_img.shape}')
        if self.verbose: print(f's1 shape apon stacking: {s1_img.shape}')
        
        




        if self.verbose:
            print(f"stacked s1_img shape: {s1_img.shape}")
            print(f"stacked s2_img shape: {s2_img.shape}")
            print(f"crop_map shape: {crop_map.shape}")
        
        check_tensor_values([s2_img, s1_img],
                            ["s2_img", "s1_img"])
        
        

        return s1_img, s2_img, crop_map

def check_tensor_values(tensor_list, input_names):
    for i, tensor in enumerate(tensor_list):
        if torch.any(tensor > 1) or torch.any(tensor < 0):
            input_name = input_names[i]
            raise ValueError(f"Values of {input_name} tensor must be between 0 and 1.")
        
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

class NormalizeS1:
    """
    Class for normalizing Sentinel-1 images between 0 and 1 for use with a pix2pix model.
    """

    def __init__(self, s1_min=-25, s1_max=10, check_nan=True, fix_nan=False):
        """
        Args
        ---
            `s1_min` (float): Minimum value for Sentinel-1 data. Default is -25.
            `s1_max` (float): Maximum value for Sentinel-1 data. Default is 10.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.s1_min = s1_min
        self.s1_max = s1_max
        self.check_nan = check_nan
        self.fix_nan = fix_nan

    def __call__(self, s1_img):
        """
        Normalize Sentinel-1 images for use with a pix2pix model.

        Args:
            s1_img (numpy.ndarray): Sentinel-1 image as a numpy array.

        Returns:
            numpy.ndarray: Normalized Sentinel-1 image. Between 0 and 1
        """
        # Sentinel 1 VV image  is between -25 and 10 dB (we insured that in the data preparation step)
        # print(np.min(target),np.max(target))
        s1_img[s1_img > self.s1_max] = self.s1_max
        s1_img[s1_img < self.s1_min] = self.s1_min

        s1_img = (s1_img - np.min(s1_img)) / (np.max(s1_img) - np.min(s1_img))
        s1_img[s1_img >= 1] = 1 - 0.0001
        s1_img[s1_img <= 0] = 0 + 0.0001

        if self.check_nan:
            if np.isnan(s1_img).any():
                raise ValueError("s1_img contains NaN values")
        elif self.fix_nan:
            if np.isnan(s1_img).any():
                s1_img[np.isnan(s1_img)] = 0.01

        return s1_img


class NormalizeS2:
    """
    Class for normalizing Sentinel-2 images between 0 and 1 for use with a pix2pix model.
    """

    def __init__(self, s2_min=0, s2_max=1, check_nan=True, fix_nan=False):
        """
        Args
        ---
            `s2_min` (float): Minimum value for Sentinel-2 data. Default is 0.
            `s2_max` (float): Maximum value for Sentinel-2 data. Default is 1.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.s2_min = s2_min
        self.s2_max = s2_max
        self.check_nan = check_nan
        self.fix_nan = fix_nan

    def __call__(self, s2_img):
        """
        Normalize Sentinel-2 images for use with a pix2pix model.

        Args:
            s2_img (numpy.ndarray): Sentinel-2 image as a numpy array.

        Returns:
            numpy.ndarray: Normalized Sentinel-2 image.
        """
        # Sentinel 2 image  is between 0 and 1 it is surface reflectance so it can't be more than 1 or less than 0
        s2_img[s2_img >= self.s2_max] = self.s2_max - 0.0001
        s2_img[s2_img <= self.s2_min] = self.s2_min + 0.0001



        if self.check_nan:
            if np.isnan(s2_img).any():
                raise ValueError("s2_img contains NaN values")
        elif self.fix_nan:
            if np.isnan(s2_img).any():
                s2_img[np.isnan(s2_img)] = 0.01

        return s2_img


class S2S1Normalize:
    """
    Class for normalizing Sentinel-2 and Sentinel-1 images for use with a pix2pix model.
    """

    def __init__(self, s1_min=-25, s1_max=10, s2_min=0, s2_max=1, check_nan=False, fix_nan=False):
        """
        Args
        ---
            `s1_min` (float): Minimum value for Sentinel-1 data. Default is -25.
            `s1_max` (float): Maximum value for Sentinel-1 data. Default is 10.
            `s2_min` (float): Minimum value for Sentinel-2 data. Default is 0.
            `s2_max` (float): Maximum value for Sentinel-2 data. Default is 1.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.normalize_s1 = NormalizeS1(s1_min=s1_min, s1_max=s1_max, check_nan=check_nan, fix_nan=fix_nan)
        self.normalize_s2 = NormalizeS2(s2_min=s2_min, s2_max=s2_max, check_nan=check_nan, fix_nan=fix_nan)

    def __call__(self, sample):
        """
        Normalize Sentinel-2 and Sentinel-1 images for use with a pix2pix model.

        Args:
            sample (tuple): Tuple containing Sentinel-2 and Sentinel-1 images as numpy arrays.

        Returns:
            tuple: Tuple containing normalized Sentinel-2 and Sentinel-1 images.
        """
        s2_img, s1_img = sample
        s2_img = self.normalize_s2(s2_img)
        s1_img = self.normalize_s1(s1_img)

        return s2_img, s1_img

import numpy as np


class CropMapTransform:
    def __init__(self, crop_values=[211, 212, 213, 214, 215, 216, 217,
                              218, 219, 221, 222, 223, 230, 231,
                              232, 233, 240, 250, 290, 300, 500]
                 ):
        self.crop_values = crop_values

    def __call__(self, crop_map):
        # Convert crop map to binary bands
        binary_bands = []
        crop_map = crop_map.squeeze()
        # Iterate through each crop value
        for crop in self.crop_values:
            # Create a binary band where crop_map equals crop value 
            binary_band = np.where(crop_map == crop, 1, 0)
            # Append the binary band to the list
            binary_bands.append(binary_band)

        # Stack bands along the first axis
        result = np.stack(binary_bands, axis=0)

        return result

# Example usage:
# transform = CropMapTransform()
# crop_map = np.random.randint(low=211, high=500, size=(1, 64, 64), dtype=np.int16)
# result = transform(crop_map)
   


if __name__ == "__main__":
    from utils.plot_utils import *
    
    s1_transform = transforms.Compose([NormalizeS1(),myToTensor()])
    s2_transform = transforms.Compose([NormalizeS2(),myToTensor()])
    crop_map_transform = transforms.Compose([CropMapTransform(),myToTensor(dtype=torch.int16)])
    
    print("Testing Dataset...")
    s1s2_dataset = Sen12Dataset(s1_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\s1\\",
                                s2_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\s2\\",
                                crop_map_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\crop_map\\",
                                s1_transform=s1_transform,
                                s2_transform=s2_transform,
                                crop_map_transform=crop_map_transform,
                                verbose=True)

    print(f"Dataset length: {len(s1s2_dataset)}")
    print(f"s1_img type: {type(s1s2_dataset[0][0])}")
    print(f"s2_img type: {type(s1s2_dataset[0][1])}")
    print(f"crop_map type: {type(s1s2_dataset[0][2])}")
    print(f"s1_img shape: {s1s2_dataset[0][0].shape}")
    print(f"s2_img shape: {s1s2_dataset[0][1].shape}")
    print(f"crop_map shape: {s1s2_dataset[0][2].shape}")