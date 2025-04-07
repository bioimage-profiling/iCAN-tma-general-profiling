import os
import argparse
import re
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import skimage
import skimage.registration
import skimage.transform
import skimage.util
    
def parseArgs():
    parser = argparse.ArgumentParser(description="Image registration parameters")
    parser.add_argument('-path_fixed', required=True, default="")
    parser.add_argument('-path_moving', required=True, default="")
    parser.add_argument('-path_output', required=True, default="")
    parser.add_argument('-channel', default="DAPI")
    parser.add_argument('-thumb_size', default="(512,512)")
    args = parser.parse_args()
    return args

def cal_phase_correlate(fixed, moving):
    # calculate phase correlations
    shift, error, phasediff = skimage.registration.phase_cross_correlation(fixed, moving)
    shift = [shift[1], shift[0]]
    shift = np.array(shift)
    shift = shift*-1
    return shift

def transform_phase_correlate(moving, shift):
    # create the correction transform
    transform = skimage.transform.AffineTransform(translation=shift)
    # apply the transformation to the moving image
    trans_moving = skimage.transform.warp(moving, transform, preserve_range=True).astype(moving.dtype)
    return trans_moving

def create_qc_thumbnails(fixed, moving, thumb_size):
    thumb_fixed = skimage.transform.resize(fixed, thumb_size, preserve_range=True).astype(np.float32)
    thumb_fixed = thumb_fixed / thumb_fixed.max()
    thumb_moving = skimage.transform.resize(moving, thumb_size, preserve_range=True).astype(np.float32)
    thumb_moving = thumb_moving / thumb_moving.max()
    thumb_rgb = np.zeros((thumb_fixed.shape[0], thumb_fixed.shape[1], 3), dtype=np.float32)
    thumb_rgb[:,:,0] = thumb_fixed
    thumb_rgb[:,:,1] = thumb_moving
    thumb_rgb = (thumb_rgb * 255).astype(np.uint8)
    thumb_diff = (np.abs(thumb_fixed - thumb_moving) * 255).astype(np.uint8)
    return thumb_rgb,thumb_diff

def register_rounds(path_fixed, path_moving, path_output, ch_pattern='DAPI', thumb_size=(512,512)):
    path_fixed = Path(path_fixed)
    path_moving = Path(path_moving)
    path_output = Path(path_output)
    path_output_qc = path_output / 'qc'

    fixed_dapi = [x for x in path_fixed.glob('*DAPI*.tif')]
    fixed_dapi.sort()
    path_output.mkdir(parents=True, exist_ok=True)
    path_output_qc.mkdir(parents=True, exist_ok=True)
    round_num = int(re.search('round(\d+)', path_moving.name).group(1))

    # Iterate over fixed images
    transforms = {}
    for i,fpath in tqdm(enumerate(fixed_dapi)):
        # Create path to moving image
        mpath = path_moving / fpath.name.replace('round1',f'round{round_num}')
        # open fixed & moving DAPIs
        fixed = imageio.imread(fpath)
        moving = imageio.imread(mpath)

        # Check and fix the shape of images
        padding = ()
        if fixed.shape != moving.shape:
            padshape = (max(fixed.shape[0],moving.shape[0]), max(fixed.shape[1],moving.shape[1]))
            if fixed.shape[0] < padshape[0] or fixed.shape[1] < padshape[1]:
                dpad = (padshape[0]-fixed.shape[0], padshape[1]-fixed.shape[1])
                padding  = ((dpad[0]//2, dpad[0]//2+dpad[0]%2), (dpad[1]//2, dpad[1]//2+dpad[1]%2))
                fixed = np.pad(fixed, padding)
            if moving.shape[0] < padshape[0] or moving.shape[1] < padshape[1]:
                dpad = (padshape[0]-moving.shape[0], padshape[1]-moving.shape[1])
                padding = ((dpad[0]//2, dpad[0]//2+dpad[0]%2), (dpad[1]//2, dpad[1]//2+dpad[1]%2))
                moving = np.pad(moving, padding)
        
        # calculate transformation
        shift = cal_phase_correlate(fixed, moving)

        # transform dapi
        moving = transform_phase_correlate(moving, shift)
        corrcoef = np.corrcoef(fixed.flatten(), moving.flatten())[0,1]

        transforms[i] = {'fixed': fpath.name,
                         'moving': mpath.name,
                         'shiftX': shift[0],
                         'shiftY': shift[1],
                         'corrcoef': corrcoef}

        # create qc thumbnail
        thumb_rgb,thumb_diff = create_qc_thumbnails(fixed, moving, thumb_size)
        imageio.imwrite(path_output_qc / (fpath.name + '_rgb.jpg'), thumb_rgb)
        imageio.imwrite(path_output_qc / (fpath.name + '_diff.jpg'), thumb_diff)

        # transform all channels
        for path_chimg in path_moving.glob(mpath.name.replace(ch_pattern,'*')):
            # open image
            moving = imageio.imread(path_chimg)

            # transform image
            if padding:
                moving = np.pad(moving, padding)
            moving = transform_phase_correlate(moving, shift)
            imageio.imwrite(path_output / path_chimg.name, moving)

    df_transforms = pd.DataFrame.from_dict(transforms).transpose()
    df_transforms.to_csv(path_output_qc / 'transforms.csv')

def main():
    # load all image names into lists
    args = parseArgs()
    path_fixed = args.path_fixed
    path_moving = args.path_moving
    path_output = args.path_output
    ch_pattern = args.channel
    thumb_size = eval(args.thumb_size)
    register_rounds(path_fixed, path_moving, path_output, ch_pattern, thumb_size)
    

if __name__ == "__main__":
    main()
