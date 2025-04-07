import sys
import numpy as np
import pandas as pd
import imageio
from pathlib import Path
from tqdm import tqdm

ipath = Path(sys.argv[1])
num_rounds = int(sys.argv[2])
opath = ipath / 'corrcoef.csv'

spots = [x for x in ipath.rglob('*round1_DAPI*.tif')]
spots.sort()

output = {}
idx = 0
for spot in tqdm(spots):
    r1_img = imageio.imread(spot)
    for r in range(2,num_rounds+1):
        rimg_path = Path(spot.as_posix().replace('round1','round{:d}'.format(r)))
        rimg = imageio.imread(rimg_path)
        try:
            corrcoef = np.corrcoef(r1_img.flatten(), rimg.flatten())[0,1]
        except:
            corrcoef = 0.0
            print("Size mismatch:"+rimg_path.name)

        output[idx] = {"fixed": spot.name, "moving": rimg_path.name, "corrcoef": corrcoef}
        idx += 1

results = pd.DataFrame.from_dict(output).transpose()
results.to_csv(opath, index=False)
