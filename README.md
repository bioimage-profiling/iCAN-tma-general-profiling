# iCAN-tma-general-profiling
Code used for analyses of multiplexed immunofluorescence microscopy images of tissue microarrays in the [iCAN Digital Precision Cancer Medicine Flagship Project](https://ican.fi/flagship-project/).

The main script is run_mIF_workflow.py that is called with one argument which is the settings.ini file. The script performs following tasks (if enabled in the settings file):
1. Registration of different cycles using the DAPI channel.
2. Creation of thumbnail images for visualization purposes (TDB in the iCAN Discovery Platform).
3. Cellular segmentation using Cellpose and dilation.
4. Extraction of features from segmented cells.
5. Classification of cells into predefined cell types (more information in the iCAN Discovery Platform handbook).
