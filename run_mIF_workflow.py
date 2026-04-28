import sys
import os
import configparser
import shutil
import re
from pathlib import Path
from imagereg import register_rounds
# Move
import imageio
import numpy as np
import pandas as pd
import functools
import itertools
import multiprocessing
from tqdm import tqdm
import skimage
import skimage.transform
import skimage.measure
import skimage.segmentation
import skimage.morphology
import scipy
import scipy.stats
#from visualization import merged_thumbnails


def pre_analysis(df_feat, basecols, posthresh, featpat='Intensity_mean_'):
    ratiocols = [x for x in df_feat.columns if featpat in x]
    clscols = []
    for rc in ratiocols:
        marker = rc.split(featpat)[1]
        # Do not process nuclei channel
        if 'Nuclei' not in marker:
            df_feat[marker+'+'] = (df_feat[rc] >= posthresh[marker])
            clscols.append(marker+'+')
    df_cls = df_feat[basecols+clscols]
    return df_cls

def CAFpanel_analysis_v2(df_cls):
    markers = ['Epithelial+','PDGFRbeta+','FAP+','pSTAT3+','aSMA+','Collagen I+']
    # Check marker combinations

    single_markers = df_cls[markers]
    comb = []
    for i in range(1,len(markers)):
        comb = comb+list(map(list,itertools.combinations(markers[1:], i)))

    comb_pos = np.zeros((single_markers.shape[0],len(comb)), dtype=bool)
    # Name columns
    comb_cols = []
    for row in comb:
        comb_str = 'Epithelial-' + "".join(row)
        comb_cols.append(comb_str)

    df_caf = pd.DataFrame(columns=comb_cols, data=comb_pos, index=df_cls.index)
    for c in comb:
        df_caf.loc[:,"Epithelial-"+"".join(c)] = np.logical_and.reduce([~single_markers['Epithelial+'].values,np.logical_and.reduce(single_markers[c],axis=1)])

    df_class = pd.concat([df_cls, df_caf], axis=1)
    # Add specific epi+ classes
    df_class['Epithelial+aSMA+'] = (single_markers['Epithelial+'] & single_markers['aSMA+'])
    df_class['Epithelial+pSTAT3+'] = (single_markers['Epithelial+'] & single_markers['pSTAT3+'])
    df_class['Epithelial+aSMA+pSTAT3+'] = (single_markers['Epithelial+'] & single_markers['aSMA+'] & single_markers['pSTAT3+'])
    df_class['Other+'] = (~single_markers['Epithelial+'] & ~single_markers['PDGFRbeta+'] & ~single_markers['FAP+'] & ~single_markers['pSTAT3+'] & ~single_markers['aSMA+'] & ~single_markers['Collagen I+'])
    
    return df_class

def CAFpanel_analysis_v1(df_cls):
    markers = ['PDGFRbeta+','FAP+','pSTAT3+','aSMA+','Collagen I+','Epithelial+']
    # Check marker combinations
    single_markers = df_cls[markers]
    comb = list(map(list,itertools.product([0,1], repeat=len(markers))))
    comb_pos = np.zeros((single_markers.shape[0],len(comb)), dtype=bool)
    for i in single_markers.index:
        comb_pos[i,comb.index(single_markers.loc[i,:].to_list())] = True

    # Name columns
    comb_cols = []
    for row in comb:
        comb_str = ''
        for i in range(len(row)):
            if row[i] == 1:
                comb_str += markers[i]
            else:
                comb_str += markers[i].replace('+','-')
        comb_cols.append(comb_str)

    df_comb = pd.DataFrame(columns=comb_cols, data=comb_pos, index=df_cls.index)
    df_class = pd.concat([df_cls, df_comb], axis=1)
    
    return df_class

def CAFpanel_analysis(df_cls, version=2):
    if version==2:
        return CAFpanel_analysis_v2(df_cls)
    elif version==1:
        return CAFpanel_analysis_v1(df_cls)
    return df_cls

def classify_TMEpanel_marker_multi_v2(row):
    if not row['CD45+'] and not row['HLA-DR+'] and row['CD68+'] and row['CD56+'] and row['CD31+']: # autofluorescent
        return 'RBC+'
    elif row['CD20+'] and row['CD45+']: # B-cell
        return 'CD20+CD45+'
    elif row['CD3+'] and not row['CD56+'] and row['CD45+']: # T-cell
        return 'CD3+CD56-CD45+'
    elif row['CD68+'] and row['HLA-DR+'] and row['CD45+']: # Macrophage HLA+
        return 'CD68+HLA-DR+CD45+'
    elif row['CD68+'] and not row['HLA-DR+'] and row['CD45+']: # Macrophage HLA-
        return 'CD68+HLA-DR-CD45+'
    elif row['CD11b+'] and row['CD45+'] and row['HLA-DR+']: # myeloid1 (non-macrophage)
        return 'CD11b+HLA-DR+CD45+'
    elif row['CD11b+'] and row['CD45+'] and not row['HLA-DR+']: # myeloid2 (non-macrophage)
        return 'CD11b+HLA-DR-CD45+'
    elif row['CD45+'] and row['HLA-DR+'] and not row['CD3+'] and not row['CD20+'] and not row['CD11b+']: # myeloid3 (non-macrophage)
        return 'CD45+CD3-CD20-CD11b-HLA-DR+'
    elif row['CD45+'] and not row['HLA-DR+'] and not row['CD3+'] and not row['CD20+'] and not row['CD11b+']: # myeloid4 (non-macrophage)
        return 'CD45+CD3-CD20-CD11b-HLA-DR-'
    elif row['Epithelial+'] and row['HLA-DR+'] and row['CD56+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial+HLA-DR+CD56+CD45-CD3-CD20-'
    elif row['Epithelial+'] and row['CD56+'] and not row['HLA-DR+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial+HLA-DR-CD56+CD45-CD3-CD20-'
    elif row['Epithelial+'] and row['HLA-DR+'] and not row['CD56+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial+HLA-DR+CD56-CD45-CD3-CD20-'
    elif row['Epithelial+'] and not row['HLA-DR+'] and not row['CD56+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial+HLA-DR-CD56-CD45-CD3-CD20-'
    elif row['CD3+'] and row['CD56+'] and row['CD45+']: # NKT-cell
        return 'CD3+CD56+CD45+'
    elif row['CD56+'] and not row['CD3+'] and row['CD45+']: # NK cell
        return 'CD3-CD56+CD45+'
    elif not row['Epithelial+'] and row['CD31+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial-CD31+CD45-CD3-CD20-'
    elif not row['Epithelial+'] and row['PDGFRbeta+'] and row['CD56+'] and row['HLA-DR+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial-PDGFRbeta+CD56+HLA-DR+CD45-CD3-CD20-'
    elif not row['Epithelial+'] and row['PDGFRbeta+'] and row['CD56+'] and not row['HLA-DR+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial-PDGFRbeta+CD56+HLA-DR-CD45-CD3-CD20-'
    elif not row['Epithelial+'] and row['PDGFRbeta+'] and not row['CD56+'] and row['HLA-DR+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial-PDGFRbeta+CD56-HLA-DR+CD45-CD3-CD20-'
    elif not row['Epithelial+'] and row['PDGFRbeta+'] and not row['CD56+'] and not row['HLA-DR+'] and not row['CD45+'] and not row['CD3+'] and not row['CD20+']:
        return 'Epithelial-PDGFRbeta+CD56-HLA-DR-CD45-CD3-CD20-'
    elif not row['Epithelial+'] and row['CD56+'] and not row['CD3+'] and not row['CD20+'] and not row['CD45+']:
        return 'Epithelial-CD56+CD45-CD3-CD20-'
    elif row['CD45+']:
        return 'Other+CD45+'
    else:
        return 'Other+CD45-'

def classify_TMEpanel_marker_multi(row):
    if row['CD20+'] and row['CD45+']:
        return 'CD20+CD45+'
    elif row['CD3+'] and row['CD56+'] and row['CD45+']:
        return 'CD3+CD56+CD45+'
    elif row['CD3+'] and not row['CD56+'] and row['CD45+']:
        return 'CD3+CD56-CD45+'
    elif row['CD56+'] and not row['CD3+'] and row['CD45+']:
        return 'CD3-CD56+CD45+'
    elif row['CD11b+'] and row['CD45+'] and row['HLA-DR+']:
        return 'CD11b+HLA-DR+CD45+' # myeloid1
    elif row['CD11b+'] and row['CD45+'] and not row['HLA-DR+']:
        return 'CD11b+HLA-DR-CD45+' # myeloid2
    elif row['CD45+'] and not row['CD3+'] and not row['CD20+'] and not row['CD11b+']:
        return 'CD45+CD3-CD20-CD11b-' # myeloid3
    elif row['HLA-DR+'] and row['CD45+']:
        return 'HLA-DR+CD45+'
    elif row['Epithelial+'] and row['HLA-DR+'] and row['CD56+'] and not row['CD45+']:
        return 'Epithelial+HLA-DR+CD56+CD45-'
    elif row['Epithelial+'] and row['CD56+'] and not row['HLA-DR+'] and not row['CD45+']:
        return 'Epithelial+HLA-DR-CD56+CD45-'
    elif row['Epithelial+'] and row['HLA-DR+'] and not row['CD56+'] and not row['CD45+']:
        return 'Epithelial+HLA-DR+CD56-CD45-'
    elif row['PDGFRbeta+'] and row['CD56+'] and row['HLA-DR+'] and not row['CD45+']:
        return 'PDGFRbeta+CD56+HLA-DR+CD45-'
    elif row['PDGFRbeta+'] and row['CD56+'] and not row['HLA-DR+'] and not row['CD45+']:
        return 'PDGFRbeta+CD56+HLA-DR-CD45-'
    elif row['PDGFRbeta+'] and not row['CD56+'] and row['HLA-DR+'] and not row['CD45+']:
        return 'PDGFRbeta+CD56-HLA-DR+CD45-'
    elif row['CD56+'] and not row['CD45+'] and not row['CD3+']:
        return 'CD56+CD3-CD45-'
    else:
        return 'Negative/Other'

def TMEpanel_analysis_v2(df_cls):
    celltypes = ['RBC+','CD20+CD45+','CD68+HLA-DR+CD45+','CD68+HLA-DR-CD45+','CD3+CD56+CD45+','CD3+CD56-CD45+','CD3-CD56+CD45+','CD11b+HLA-DR+CD45+','CD11b+HLA-DR-CD45+','CD45+CD3-CD20-CD11b-HLA-DR+','CD45+CD3-CD20-CD11b-HLA-DR-','Other+CD45+','Epithelial+HLA-DR+CD56+CD45-CD3-CD20-','Epithelial+HLA-DR-CD56+CD45-CD3-CD20-','Epithelial+HLA-DR+CD56-CD45-CD3-CD20-','Epithelial+HLA-DR-CD56-CD45-CD3-CD20-','Epithelial-CD31+CD45-CD3-CD20-','Epithelial-PDGFRbeta+CD56+HLA-DR+CD45-CD3-CD20-','Epithelial-PDGFRbeta+CD56+HLA-DR-CD45-CD3-CD20-','Epithelial-PDGFRbeta+CD56-HLA-DR+CD45-CD3-CD20-','Epithelial-PDGFRbeta+CD56-HLA-DR-CD45-CD3-CD20-','Epithelial-CD56+CD45-CD3-CD20-','Other+CD45-']

    df_class = pd.DataFrame(columns=celltypes, data=np.zeros((df_cls.shape[0],len(celltypes)),dtype=bool), index=df_cls.index)
    for i,row in df_cls.iterrows():
        df_class.loc[i,classify_TMEpanel_marker_multi_v2(row)] = True

    # Add derived single marker positivity
    markers = ['CD68+','CD3+','CD56+','CD11b+','PDGFRbeta+','CD20+','CD31+','CD45+','HLA-DR+','Epithelial+']
    for marker in markers:
        cols = [x for x in df_class.columns if marker in x]
        df_class['der'+marker] = df_class[cols].astype(np.uint8).sum(axis=1) >= 1

    df_class = pd.concat([df_cls, df_class], axis=1)
    return df_class

def TMEpanel_analysis_v1(df_cls):
    celltypes = ['CD20+CD45+','CD3+CD56+CD45+','CD3+CD56-CD45+','CD3-CD56+CD45+','CD11b+HLA-DR+CD45+','CD11b+HLA-DR-CD45+','CD45+CD3-CD20-CD11b-','HLA-DR+CD45+','Epithelial+HLA-DR+CD56+CD45-','Epithelial+HLA-DR-CD56+CD45-','Epithelial+HLA-DR+CD56-CD45-','PDGFRbeta+CD56+HLA-DR+CD45-','PDGFRbeta+CD56+HLA-DR-CD45-','PDGFRbeta+CD56-HLA-DR+CD45-','CD56+CD3-CD45-','Negative/Other']

    df_class = pd.DataFrame(columns=celltypes, data=np.zeros((df_cls.shape[0],len(celltypes)),dtype=np.uint8), index=df_cls.index)
    for i,row in df_cls.iterrows():
        df_class.loc[i,classify_TMEpanel_marker_multi(row)] = 1

    df_class = pd.concat([df_cls, df_class], axis=1)
    return df_class

def TMEpanel_analysis(df_cls, version=2):
    if version==2:
        return TMEpanel_analysis_v2(df_cls)
    elif version==1:
        return TMEpanel_analysis_v1(df_cls)
    return df_cls

def TcellPanel_analysis_v2(df_cls):
    immune_markers = ['PD-1+', 'FOXP3+', 'PD-L1+', 'TIM-3+', 'Ki67+', 'GranzymeB+']
    panepi_markers = ['Ki67+', 'PD-L1+', 'TIM-3+']
    markers = ['PD-1+','CD3+','PD-L1+','CD8+','TIM-3+','FOXP3+','GranzymeB+','Ki67+','CD4+','Epithelial+']
    single_markers = df_cls[markers]

    # ADD RBC class
    df_cls['RBC+'] = (single_markers['PD-1+'] & ~single_markers['CD3+'] & single_markers['PD-L1+'] & ~single_markers['CD4+'] & ~single_markers['Epithelial+'])

    # Check marker combinations
    comb_immune = []
    for i in range(1,len(immune_markers)+1):
        comb_immune += list(map(list,itertools.combinations(immune_markers, i)))
    comb_immune_pos = np.zeros((df_cls.shape[0],len(comb_immune)), dtype=bool)
    for i,ci in enumerate(comb_immune):
        comb_immune_pos[:,i] = (df_cls[ci].astype(np.uint8).sum(axis=1) == len(ci))
    # Add fixed markers
    comb_fixed1 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed1[:,i] = single_markers['CD3+'] & single_markers['CD8+'] & comb_immune_pos[:,i] & ~df_cls['RBC+']
    comb_fixed1_cols = [['CD3+','CD8+'] + x for x in comb_immune]
    
    comb_fixed2 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed2[:,i] = single_markers['CD3+'] & ~single_markers['CD8+'] & single_markers['CD4+'] & comb_immune_pos[:,i] & ~df_cls['RBC+']
    comb_fixed2_cols = [['CD3+','CD8-','CD4+'] + x for x in comb_immune]
    
    comb_fixed3 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed3[:,i] = single_markers['CD3+'] & ~single_markers['CD8+'] & ~single_markers['CD4+'] & comb_immune_pos[:,i] & ~df_cls['RBC+']
    comb_fixed3_cols = [['CD3+','CD8-','CD4-'] + x for x in comb_immune]

    comb_fixed4 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed4[:,i] = ~single_markers['CD3+'] & single_markers['CD4+'] & comb_immune_pos[:,i] & ~df_cls['RBC+']
    comb_fixed4_cols = [['CD3-','CD4+'] + x for x in comb_immune]
    
    # Merge immune classification
    comb_immune_pos = np.concatenate([comb_fixed1, comb_fixed2, comb_fixed3, comb_fixed4], axis=1)
    comb_immune = comb_fixed1_cols + comb_fixed2_cols + comb_fixed3_cols + comb_fixed4_cols
    comb_immune = [''.join(x) for x in comb_immune]
    df_immune = pd.DataFrame(columns=comb_immune, data=comb_immune_pos, index=df_cls.index)

    # Prepare panepi cols
    comb_panepi = []
    for i in range(1,len(panepi_markers)+1):
        comb_panepi += list(map(list,itertools.combinations(panepi_markers, i)))
    comb_panepi_pos = np.zeros((df_cls.shape[0],len(comb_panepi)), dtype=bool)
    for i,cp in enumerate(comb_panepi):
        comb_panepi_pos[:,i] = (df_cls[cp].astype(np.uint8).sum(axis=1) == len(cp))
    for i in range(comb_panepi_pos.shape[1]):
        comb_panepi_pos[:,i] = ~single_markers['CD3+'] & ~single_markers['CD4+'] & single_markers['Epithelial+'] & comb_panepi_pos[:,i] & ~df_cls['RBC+']
    comb_panepi = [['Epithelial+'] + x for x in comb_panepi]
    comb_panepi = [''.join(x) for x in comb_panepi]
    df_panepi = pd.DataFrame(columns=comb_panepi, data=comb_panepi_pos, index=df_cls.index)

    df_merge = pd.concat([df_immune, df_panepi], axis=1)
    df_merge['Other+'] = (~df_cls['RBC+'] & ~single_markers['CD3+'] & ~single_markers['CD4+'] & ~single_markers['Epithelial+'])

    df_class = pd.concat([df_cls, df_merge], axis=1)
    return df_class

def TcellPanel_analysis_v1(df_cls):
    immune_markers = ['PD-1+', 'FOXP3+', 'GranzymeB+', 'PD-L1+', 'TIM-3+', 'Ki67+']
    panepi_markers = ['PD-L1+', 'Ki67+', 'TIM-3+']

    # Check marker combinations
    comb_immune = []
    for i in range(1,len(immune_markers)+1):
        comb_immune += list(map(list,itertools.combinations(immune_markers, i)))
    comb_immune_pos = np.zeros((df_cls.shape[0],len(comb_immune)), dtype=bool)
    for i,ci in enumerate(comb_immune):
        comb_immune_pos[:,i] = (df_cls[ci].astype(np.uint8).sum(axis=1) == len(ci))
    # Add fixed markers
    comb_fixed1 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed1[:,i] = df_cls['CD3+'] & df_cls['CD8+'] & comb_immune_pos[:,i]
    comb_fixed1_cols = [['CD3+','CD8+'] + x for x in comb_immune]
    
    comb_fixed2 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed2[:,i] = df_cls['CD3+'] & ~df_cls['CD8+'] & df_cls['CD4+'] & comb_immune_pos[:,i]
    comb_fixed2_cols = [['CD3+','CD8-','CD4+'] + x for x in comb_immune]
    
    comb_fixed3 = np.zeros(comb_immune_pos.shape, dtype=bool)
    for i in range(comb_immune_pos.shape[1]):
        comb_fixed3[:,i] = df_cls['CD3+'] & ~df_cls['CD8+'] & ~df_cls['CD4+'] & comb_immune_pos[:,i]
    comb_fixed3_cols = [['CD3+','CD8-','CD4-'] + x for x in comb_immune]
    
    # Merge immune classification
    comb_immune_pos = np.concatenate([comb_fixed1, comb_fixed2, comb_fixed3], axis=1)
    comb_immune = comb_fixed1_cols + comb_fixed2_cols + comb_fixed3_cols
    comb_immune = [''.join(x) for x in comb_immune]
    df_immune = pd.DataFrame(columns=comb_immune, data=comb_immune_pos, index=df_cls.index)

    # Prepare panepi cols
    comb_panepi = []
    for i in range(1,len(panepi_markers)+1):
        comb_panepi += list(map(list,itertools.combinations(panepi_markers, i)))
    comb_panepi_pos = np.zeros((df_cls.shape[0],len(comb_panepi)), dtype=bool)
    for i,cp in enumerate(comb_panepi):
        comb_panepi_pos[:,i] = (df_cls[cp].astype(np.uint8).sum(axis=1) == len(cp))
    for i in range(comb_panepi_pos.shape[1]):
        comb_panepi_pos[:,i] = ~df_cls['CD3+'] & df_cls['Epithelial+'] & comb_panepi_pos[:,i]
    comb_panepi = [['CD3-','Epithelial+'] + x for x in comb_panepi]
    comb_panepi = [''.join(x) for x in comb_panepi]
    df_panepi = pd.DataFrame(columns=comb_panepi, data=comb_panepi_pos, index=df_cls.index)

    df_class = pd.concat([df_cls, df_immune, df_panepi], axis=1)
    return df_class

def TcellPanel_analysis(df_cls, version=2):
    if version==2:
        return TcellPanel_analysis_v2(df_cls)
    elif version==1:
        return TcellPanel_analysis_v1(df_cls)
    return df_cls

def classify_cells(config):
    rootpath = Path(config['GENERAL']['rootpath'])
    panels = config['GENERAL']['panels'].split(',')
    for panel in panels:
        df_markers = pd.read_csv(config['GENERAL']['markerpath'])
        df_markers = df_markers[df_markers['Panel']==panel]
        df_thresh = pd.read_csv(f'{rootpath}/{panel}/thresholds.csv')
        df_thresh = df_thresh[df_thresh['panel']==panel]
        featpath = rootpath / f'{panel}/features'
        classpath = rootpath / f'{panel}/classification'
        if not classpath.exists():
            classpath.mkdir()

        # Iterate over all slides
        sfpaths = [x for x in featpath.glob('*.csv')]
        sfpaths.sort()
        df_aggs = []
        for sfpath in tqdm(sfpaths):
            df_feat = pd.read_csv(sfpath)
            posthresh = {}
            for i,row in df_thresh[df_thresh['slide'].apply(lambda x: x.split('_round')[0]) == sfpath.name[:-4]].iterrows():
                if row['channel'] != 'DAPI' or 'round1' in row['slide']:
                    marker = df_markers.loc[(df_markers['Round']==int(row['slide'][-1]))&(df_markers['Staining']==row['channel']),'Marker'].values[0]
                    posthresh[marker] = row['threshold']

            basecols = ['Panel','Slide','ROI','ID','Location_X','Location_Y','Size']
            df_cls = pre_analysis(df_feat, basecols, posthresh, featpat = config['CLASSIFICATION']['pos_threshold'])
            
            if panel == 'CAFpanel':
                df_class = CAFpanel_analysis(df_cls, version=config['CLASSIFICATION'].getint('version'))
            elif panel == 'TMEpanel':
                df_class = TMEpanel_analysis(df_cls, version=config['CLASSIFICATION'].getint('version'))
            else:
                df_class = TcellPanel_analysis(df_cls, version=config['CLASSIFICATION'].getint('version'))

            # Change type for all other columns except basecols
            columns = [x for x in df_class.columns if x not in basecols]
            df_class = pd.concat([df_class[basecols],df_class[columns].astype(np.uint8)], axis=1)

            clspath = sfpath.name
            if config['CLASSIFICATION'].getint('version') == 2:
                clspath = clspath.replace('.csv', '_v2.csv')
            df_class.to_csv(classpath / clspath, index=False)
            # Create aggregates
            df_aggclass = df_class.groupby(['Panel','Slide','ROI']).mean()
            df_aggclass = df_aggclass.reset_index()
            df_aggclass.insert(3,'Cells',df_class.groupby(['Panel','Slide','ROI']).count()['ID'].values)
            df_aggclass = df_aggclass.drop(columns=['ID','Location_X','Location_Y','Size'], axis=1)
            aggclspath = sfpath.name.replace('.csv','_agg.csv')
            if config['CLASSIFICATION'].getint('version') == 2:
                aggclspath = aggclspath.replace('.csv', '_v2.csv')
            df_aggclass.to_csv(classpath / aggclspath, index=False)


def measure_objects(labelpath, imgpaths, thresholds='', markers='', panel='', dpath=''):
    """
    Measure mean, stdev, median, mad, lower quartile and upper quartile
    """
    cellseg = imageio.imread(labelpath)
    # Measure obj location and size
    centroidX = np.zeros(cellseg.max(), dtype=np.float32)
    centroidY = np.zeros(cellseg.max(), dtype=np.float32)
    size = np.zeros(cellseg.max(), dtype=np.int32)
    for i,label in enumerate(range(1,cellseg.max()+1)):
        coords = np.where(cellseg==label)
        coordsX = coords[1]
        coordsX.sort()
        centroidX[i] = coordsX[0] + (coordsX[-1] - coordsX[0]) / 2
        coordsY = coords[0]
        coordsY.sort()
        centroidY[i] = coordsY[0] + (coordsY[-1] - coordsY[0]) / 2
        size[i] = coordsX.shape[0]

    df = pd.DataFrame(data={'ID': np.arange(cellseg.max())+1,
                            'Location_X': centroidX,
                            'Location_Y': centroidY,
                            'Size': size})
    dfs = [df]
    rchre = re.compile('(round\d_[A-Z0-9]+)')
    for chnum,impath in enumerate(imgpaths):
        rch = rchre.search(impath.name).group(1)
        marker = markers[rch]
        chimg = imageio.imread(impath).squeeze()
        if chimg.shape != cellseg.shape: # Make sure that registered cellseg and chimg are of the same shape
            dpad = (chimg.shape[0]-cellseg.shape[0], chimg.shape[1]-cellseg.shape[1])
            padding  = ((dpad[0]//2, dpad[0]//2+dpad[0]%2), (dpad[1]//2, dpad[1]//2+dpad[1]%2))
            if min(dpad) < 0:
                print("Error: {} image has smaller shape than {}".format(impath.name, labelpath))
            chimg = chimg[padding[0][0]:cellseg.shape[0]+padding[0][0], padding[1][0]:cellseg.shape[1]+padding[1][0]]
        thrchimg = np.zeros(chimg.shape, dtype=np.float32)
        thrchimg[chimg >= thresholds[chnum]] = 1.0
        # Debug
        imageio.imwrite(dpath / impath.name.replace('.tif','.png'), (thrchimg*255).astype(np.uint8))

        posratio = np.zeros(cellseg.max(), dtype=np.float32)
        intmean = np.zeros(cellseg.max(), dtype=np.float32)
        intstd = np.zeros(cellseg.max(), dtype=np.float32)
        intmedian = np.zeros(cellseg.max(), dtype=np.float32)
        intmad = np.zeros(cellseg.max(), dtype=np.float32)
        intlower = np.zeros(cellseg.max(), dtype=np.float32)
        intupper = np.zeros(cellseg.max(), dtype=np.float32)

        for i,label in enumerate(range(1,cellseg.max()+1)):
            obj = chimg[cellseg == label]
            lowerq,median,upperq = np.quantile(obj, [0.25, 0.5, 0.75])
            posratio[i] = thrchimg[cellseg == label].mean()
            intmean[i] = obj.mean()
            intstd[i] = obj.std()
            intmedian[i] = median
            intmad[i] = scipy.stats.median_abs_deviation(obj)
            intlower[i] = lowerq
            intupper[i] = upperq

        df = pd.DataFrame(data={'Intensity_posratio_'+marker: posratio,
                                'Intensity_mean_'+marker: intmean,
                                'Intensity_stdev_'+marker: intstd,
                                'Intensity_median_'+marker: intmedian,
                                'Intensity_mad_'+marker: intmad,
                                'Intensity_lowerquartile_'+marker: intlower,
                                'Intensity_upperquartile_'+marker: intupper})
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    roinum = int(re.search('_roi(\d+).',labelpath.name).group(1))
    df.insert(0,'ROI',roinum)
    df.insert(0,'Slide',labelpath.name.split(panel)[0][:-1])
    df.insert(0,'Panel',panel)
    
    return df

def extract_features(config):
    panels = config['GENERAL']['panels'].split(',')
    rreg = re.compile('roi(\d+)')
    for panel in panels:
        regpath = Path(config['GENERAL']['rootpath']) / f'{panel}/reg'
        cellpath = Path(config['GENERAL']['rootpath']) / f'{panel}/cellseg'
        featpath = Path(config['GENERAL']['rootpath']) / f'{panel}/features'
        if not featpath.exists():
            featpath.mkdir()
        maskpath = featpath / 'masks'
        if not maskpath.exists():
            maskpath.mkdir()
        df_thresh = pd.read_csv(Path(config['GENERAL']['rootpath']) / f'{panel}/thresholds.csv')
        df_markers = pd.read_csv(config['GENERAL']['markerpath'])
        df_markers = df_markers[df_markers['Panel']==panel]
        markers = {}
        for i,row in df_markers.iterrows():
            markers['round{}_{}'.format(row['Round'],row['Staining'])] = row['Marker']

        # Find all segmented cells
        slide_paths = [x for x in cellpath.glob('*round1*')]
        slide_paths.sort()
        for sp in slide_paths:
            df_slide = df_thresh[df_thresh['slide'].isin([sp.name.replace('round1','round{:d}'.format(x+1)) for x in range(config[panel].getint('rounds'))])]
            df_slide = df_slide[~((df_slide['channel']=='DAPI')&(~df_slide['slide'].str.contains('round1')))].sort_values(by=['slide','channel'])
            slmaskpath = maskpath / sp.name.split('round')[0][:-1]
            if not slmaskpath.exists():
                slmaskpath.mkdir()

            # Extract features from all ROIs in the slide
            rpaths = [x for x in sp.glob('*.png')]
            rpaths.sort()
            rois = []
            for rpath in rpaths:
                roinum = rreg.search(rpath.name).group(1)
                chpaths = []
                for i,row in df_slide.iterrows():
                    # Do not include DAPI from rounds 2+
                    if not (row['channel'] == 'DAPI' and int(re.search('round(\d)',rpath.name).group(1)) > 1):
                        chpaths.append(regpath / row['slide'] / (row['slide'] + '_{}_roi{}.tif'.format(row['channel'],roinum)))
                rois.append((rpath, chpaths))
                #df_feat = measure_objects(rpath, chpaths, df_slide['threshold'].values, markers, featpath)

            # Launch parallel processing pool
            measure_func = functools.partial(measure_objects, thresholds=df_slide['threshold'].values, markers=markers, panel=panel, dpath=slmaskpath)
            pool = multiprocessing.Pool(processes=config['GENERAL'].getint('max_processes'))
            dfs = tqdm(pool.starmap(measure_func, rois))
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df = df.sort_values(by=['Panel','Slide','ROI','ID']).reset_index(drop=True)
            df.to_csv(featpath / (sp.name.split(panel)[0]+panel+'.csv'), index=False)


def segmentation(config):
    # Segmentation settings
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = config['SEGMENTATION']['model_path']
    import cellpose
    import cellpose.models
    model = cellpose.models.Cellpose(gpu=True, model_type='nuclei')
    flow_threshold = config['SEGMENTATION'].getfloat('flow_threshold')
    cellprob_threshold = config['SEGMENTATION'].getfloat('cellprob_threshold')
    min_size = config['SEGMENTATION'].getint('min_size')
    dil_rad = config['SEGMENTATION'].getint('dilation_radius')
    #
    elem = skimage.morphology.disk(dil_rad)
    
    panels = config['GENERAL']['panels'].split(',')
    for panel in panels:
        regpath = Path(config['GENERAL']['rootpath']) / f'{panel}/reg'
        nucpath = Path(config['GENERAL']['rootpath']) / f'{panel}/nucseg'
        cellpath = Path(config['GENERAL']['rootpath']) / f'{panel}/cellseg'

        # Find all round1 slides
        slide_paths = [x for x in regpath.glob('*round1')]
        slide_paths.sort()
        for slidepath in slide_paths:
            nspath = nucpath / slidepath.name
            cspath = cellpath / slidepath.name
            nspath.mkdir(parents=True, exist_ok=True)
            cspath.mkdir(parents=True, exist_ok=True)

            dapifiles = [x for x in slidepath.glob('*round1_DAPI*.tif')]
            dapifiles.sort()
            dfs = []
            for ipath in tqdm(dapifiles):
                img = imageio.imread(ipath)
                masks, flows, styles, diams = model.eval(img, channels=[0,0], diameter=None, resample=True, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, niter=2000, do_3D=False)
                # Filter nuclei and create cells
                df_rprops = pd.DataFrame(skimage.measure.regionprops_table(masks, img, properties=['num_pixels','intensity_mean']))
                for ind in df_rprops[df_rprops['num_pixels'] < min_size].index + 1:
                    masks[masks==ind] = 0

                df_rprops = df_rprops[df_rprops['num_pixels'] >= min_size].reset_index(drop=True)
                nuclei,_,_ = skimage.segmentation.relabel_sequential(masks)
                # dilate nuclei segmentation to create cell segmentation
                cells = skimage.morphology.dilation(nuclei, footprint=elem)

                # Write and save output
                imageio.imwrite(nspath / ipath.name.replace('.tif','.png'), nuclei)
                imageio.imwrite(cspath / ipath.name.replace('.tif','.png'), cells)
                df_rprops['image'] = ipath.name
                df_rprops['id'] = df_rprops.index + 1
                df_rprops = df_rprops[['image','id','num_pixels','intensity_mean']]
                dfs.append(df_rprops)

            df_nuc = pd.concat(dfs, axis=0, ignore_index=True)
            df_nuc.to_csv(nspath / 'nuclei_properties.csv', index=False)

def registration(config):
    panels = config['GENERAL']['panels'].split(',')
    for panel in panels:
        num_rounds = config[panel].getint('rounds')
        orgpath = Path(config['GENERAL']['rootpath']) / f'{panel}/Original'
        regpath = Path(config['GENERAL']['rootpath']) / f'{panel}/reg'

        # Find all round1 slides
        fixed_rounds = [x for x in orgpath.glob('*round1')]
        for fixed_round in fixed_rounds:
            for num_moving in range(2,num_rounds+1):
                moving_round = fixed_round.parent / fixed_round.name.replace('round1',f'round{num_moving}')
                outpath = regpath / moving_round.name
                # Run registration
                register_rounds(fixed_round, moving_round, outpath, ch_pattern='DAPI', thumb_size=(512,512))
            
            if config['REGISTRATION'].getboolean('copy_fixed'):
                shutil.copytree(fixed_round,regpath / fixed_round.name)

def colorize(img, iname, thrlow, thrhigh, alpha=False):
    if alpha:
        rgbimg = np.zeros((*img.shape,4), dtype=img.dtype)
    else:
        rgbimg = np.zeros((*img.shape,3), dtype=img.dtype)

    img = (img - thrlow) / (thrhigh - thrlow)
    img[img<0] = 0.0
    img[img>1] = 1.0

    rgbimg[:,:,0] = img
    rgbimg[:,:,1] = img
    rgbimg[:,:,2] = img

    if alpha:
        rgbimg[:,:,3] = img
    
    return rgbimg

def slide_auto_threshold(slide_path, panel=""):
    # Find all channels
    chreg = re.compile('_round\d_(\w+)_')
    slide_channels = {}
    slide_dict = {}
    dict_key = 0
    count_thresh = 1000
    quantile = 0.998
    logbins = 256
    logbin_edges = np.linspace(0.0, np.log1p(2**16), logbins)
    for imgname in [x.name for x in slide_path.glob('*roi*.tif')]:
        slide_channels[chreg.search(imgname).group(1)] = 1
    channels = [x for x in slide_channels.keys()]

    hist_log = np.zeros((len(channels),logbins), dtype=np.int64)
    for i,ch in enumerate(channels):
        for imgpath in slide_path.glob('*_{}_*.tif'.format(ch)):
            img = imageio.imread(imgpath).flatten()
            img = img[img > 0]
            hist_log[i,:] += np.histogram(np.log1p(img), bins=logbins, range=(0.0, np.log1p(2**16)))[0]

    for i,ch in enumerate(channels):
        ch_hist = hist_log[i,:].copy()
        mode = np.argmax(ch_hist)
        fder = ch_hist[1:] - ch_hist[:-1]
        thresh_cand = np.where((fder[:-1] < -count_thresh) & (fder[1:] > count_thresh))[0]

        auto_thr = 1
        try:
            thresh = logbin_edges[thresh_cand[np.where(thresh_cand > mode)[0]]][0]
        except: # TEST
            mean = np.average(logbin_edges, weights=ch_hist)
            var = np.average((logbin_edges - mean)**2, weights=ch_hist)
            thresh = logbin_edges[mode] + np.log1p(np.sqrt(np.expm1(var)))
            auto_thr = 0
                
        thresh = np.expm1(thresh)

        # check quantile
        quantbin = ch_hist.shape[0]-1
        while quantbin > 0:
            if ch_hist[quantbin:].sum() > ch_hist.sum()*(1-quantile):
                break
            quantbin -= 1
        quant = np.expm1(logbin_edges[quantbin])

        slide_dict[dict_key] = {'panel': panel,
                                'slide': slide_path.name,
                                'channel': ch,
                                'threshold': thresh,
                                'quantile': quant,
                                'auto_thresh': auto_thr}
        dict_key += 1

    df_slide = pd.DataFrame.from_dict(slide_dict).transpose()
    return df_slide

def auto_threshold(config):
    rootpath = Path(config['GENERAL']['rootpath'])
    panels = config['GENERAL']['panels'].split(',')
    for panel in panels:
        panelpath = rootpath / panel
        num_rounds = config[panel].getint('rounds')
        regpath = Path(f'{rootpath}/{panel}/reg')

        # Check all slides in the panel
        slides = [x for x in regpath.glob('*round*')]
        slides.sort()

        # Iterate over slides
        pool = multiprocessing.Pool(processes=config['GENERAL'].getint('max_processes'))
        slide_func = functools.partial(slide_auto_threshold, panel=panel)
        df_slides = []
        with tqdm(total=len(slides)) as pbar:
            for df_slide in pool.imap(slide_func, slides):
                df_slides.append(df_slide)
                pbar.update()

        df_panel = pd.concat(df_slides, axis=0, ignore_index=True)
        df_panel = df_panel.sort_values(by=['slide','channel'])
        df_panel.to_csv(panelpath / 'thresholds.csv', index=False)

def create_roi_thumbnails(roi, regpath=None, thumbpath=None, df_thresh=None, downscale=4):
    df_rt = df_thresh[df_thresh['slide']=='_'.join(roi.name.split('_')[:4])]
    #df_rt = df_thresh[df_thresh['slide']=='_'.join(roi.name.split('_')[:3])]
    # Find rest roi images
    roiname = roi.name
    roundx_chs = regpath.rglob(roiname.replace('round1','round*').replace('DAPI','*'))
    roundx_chs = [x for x in roundx_chs if 'DAPI' not in x.name]
    ch_paths = [roi] + roundx_chs
    ch_paths.sort()
    chreg = re.compile('round\d_(\w+)_roi')
    roishape = imageio.imread(roi).shape
    for i,ch_path in enumerate(ch_paths):
        chimg = imageio.imread(ch_path)
        if roishape != chimg.shape: # Crop chimg if shape is bigger
            diff = (chimg.shape[0]-roishape[0], chimg.shape[1]-roishape[1])
            if min(diff) < 0:
                print("Error: {} image has smaller shape than {}".format(ch_path.name, roiname))
            chimg = chimg[diff[0]//2:roishape[0]+diff[0]//2, diff[1]//2:roishape[1]+diff[1]//2]
        chimg = skimage.transform.resize(chimg, (chimg.shape[0]//downscale, chimg.shape[1]//downscale), preserve_range=True)
        roich = df_rt[df_rt['channel']==chreg.search(ch_path.name).group(1)]
        rgba = colorize(chimg, ch_paths[i].name, roich['threshold'].to_numpy(), roich['quantile'].to_numpy(), alpha=True)
        rgba = (rgba * 255).astype(np.uint8)

        parts = ch_path.name.split('_')
        mpath = thumbpath / '_'.join(parts[:4])
        #mpath = thumbpath / '_'.join(parts[:3])
        mname = ch_path.name
        mname = mname.replace('.tif','.png')
        imageio.imwrite(mpath / mname, rgba)


def create_thumbnails(config):
    rootpath = config['GENERAL']['rootpath']
    panels = config['GENERAL']['panels'].split(',')
    for panel in panels:
        num_rounds = config[panel].getint('rounds')
        regpath = Path(f'{rootpath}/{panel}/reg')
        thumbpath = Path(f'{rootpath}/{panel}/thumb')
        thumbpath.mkdir(parents=True, exist_ok=True)
        df_thresh = pd.read_csv(f'{rootpath}/{panel}/thresholds.csv')
        # Create output folders
        for s in [x.name for x in regpath.glob('*round*')]:
            (thumbpath / s).mkdir(parents=True, exist_ok=True)

        rois = regpath.rglob('*round1_DAPI_roi*.tif')
        rois = [x for x in rois]
        thumb_func = functools.partial(create_roi_thumbnails, regpath=regpath, thumbpath=thumbpath, df_thresh=df_thresh, downscale=config['THUMBNAIL'].getint('downscale'))
        pool = multiprocessing.Pool(processes=config['GENERAL'].getint('max_processes'))
        with tqdm(total=len(rois)) as pbar:
            for _ in pool.imap_unordered(thumb_func, rois):
                pbar.update()


def main():
    # Read settings file
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # 1. Perform registration
    if config['RUN'].getboolean('registration'):
        registration(config)

    # 2. Create thumbnails for visualization
    if config['RUN'].getboolean('thumbnail'):
        auto_threshold(config)
        create_thumbnails(config)
    
    # 3. Do nuclei segmentation
    if config['RUN'].getboolean('nuclei_segmentation'):
        segmentation(config)

    # 4. Feature extraction
    if config['RUN'].getboolean('feature_extraction'):
        extract_features(config)

    # 5. Cell classification
    if config['RUN'].getboolean('classification'):
        classify_cells(config)

if __name__ == "__main__":
    main()
