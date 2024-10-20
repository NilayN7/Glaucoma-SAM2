import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt 
import random 

def get_mask_paths(img_paths):
    img_file_names = [str(f).split(".npy")[0].split("Hiroshi_ONH_OCT/")[-1] for f in img_paths]
    mask_paths = [f"/local2/acc/Glaucoma/Hiroshi_ONH_OCT_masks/{f}_mask.npy" for f in img_file_names]

    return mask_paths 

def upscale_slices(img_dir, num_samples=125):
    img_paths = random.sample([f for f in glob.glob(img_dir + "*npy")], num_samples)

    train_img_paths = random.sample(img_paths, int(num_samples*0.8))
    
    test_img_paths = []
    for i in img_paths:
        if i not in train_img_paths: 
            test_img_paths.append(i)

    train_mask_paths = get_mask_paths(train_img_paths)
    test_mask_paths = get_mask_paths(test_img_paths)

    all_paths = {
        "train": train_img_paths + train_mask_paths, 
        "test": test_img_paths + test_mask_paths,
    }    

    for split, file_paths in all_paths.items():
        print(f"saving {split} flies")
        for file_path in file_paths:
            np_img = np.load(file_path)
            for idx in range(np_img.shape[0]):
                # upsample = transforms.Resize(size=(512, 1024), interpolation=transforms.InterpolationMode.LANCZOS)
                
                file_name = str(file_path).split("/")[-1].split(".npy")[0]
                
                if "mask" in file_name:
                    image_slice = Image.fromarray((np_img[:, :, idx]*255).astype(np.uint8))
                else: 
                    image_slice = Image.fromarray(np_img[:, :, idx])

                # upsampled_slice = upsample(image_slice)
                # pad = transforms.Pad((0, 256))    # 1024 x 1024
                pad = transforms.Pad((0, 32))       # 128 x 128
                padded_slice = pad(image_slice)

                if "mask" in file_name:
                    if split == "train":
                        save_path = f"/local2/nilay/Hiroshi_ONH_OCT_original_dataset/train/GT/{file_name.split('_mask')[0]}_slice_{idx+1}.png"
                    elif split == "test":
                        save_path = f"/local2/nilay/Hiroshi_ONH_OCT_original_dataset/test/GT/{file_name.split('_mask')[0]}_slice_{idx+1}.png"
                    else: 
                        raise ValueError("Invalid split")
                else:
                    if split == "train":
                        save_path = f"/local2/nilay/Hiroshi_ONH_OCT_original_dataset/train/Imgs/{file_name}_slice_{idx+1}.png"
                    elif split == "test":
                        save_path = f"/local2/nilay/Hiroshi_ONH_OCT_original_dataset/test/Imgs/{file_name}_slice_{idx+1}.png"
                    else: 
                        raise ValueError("Invalid split")
                    
                padded_slice.save(save_path)


# def check_images_GT(img_path, gt_path):
#     images = [f for f in glob.glob(img_path + "*.png")]
#     gt = [f for f in glob.glob(gt_path + "*.png")]

#     for i in images: 
#         if i not in gt:
#             print("this is the file that is not common in both folders: ", i)

#     for i in images: 
#         if i == "":
#             print("this us the file that has an empty string as its name in the images  folder: ", i)


#     for i in gt:
#         if i == "":
#             print("this is the file that has an empty string as its name in th GT folder:", i)


# img_path = "/local2/nilay/Hiroshi_ONH_OCT_dataset/test/Imgs"
# gt_path = "/local2/nilay/Hiroshi_ONH_OCT_dataset/test/GT"
# check_images_GT(img_path, gt_path)

if __name__ == "__main__":
    img_dir = "/local2/acc/Glaucoma/Hiroshi_ONH_OCT/"
    upscale_slices(img_dir)

