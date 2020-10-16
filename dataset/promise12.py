"""
Format
Each downloaded file contains MR scans, stored in Meta (or MHD/RAW) format.
This format stores an image as an ASCII readable header file with extension .mhd and a separate binary file for the image data with extension .raw.
This format is ITK compatible.
If you want to write your own code to read the data, note that in the header file you can find the dimensions of the scan and the voxel spacing.
In the raw file the values for each voxel are stored consecutively with index running first over x, then y, then z. The voxel-to-world matrix is also available in this header file.

    The voxel type for T2-weighted images is SHORT (16 bit signed).
    The voxel type for the reference standard image is CHAR (8 bit signed).
    The reference standard image only contains the values 1 for prostate and 0 for background.
"""

dataset_root_path = '/home/hlli/project/yufei/MASF/dataset/PROMISE12/'


