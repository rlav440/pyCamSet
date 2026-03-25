from pyCamSet.calibration_targets import target_Ccube as cc
from pathlib import Path

'''
Purpose: Generate a raster pdf, vector pdf, or svg of a Ccube target.
Warning: The vector pdf and svg files need to be tested as they are effectively 
conversions of the original OpenCV raster images. These conversions were created as, 
with higher resolution cameras, rasters create artifacts that might reduce the accuracy 
of corner and marker detection.
'''

n_points = 8 # number of squares and markers
length = 8 # length in mm

f_path = Path(r'D:\Work\calibration_targets\2D') # main output directory
f_name = f'ccube_{n_points}points_{length}mm.pdf' # name of file
f_out = f_path / f_name # absolute path to file

# visualize and create ccube
cube = cc.Ccube(length, n_points)
# cube.plot()
cube.save_to_pdf(f_out) # Default is raster
# cube.save_to_pdf(f_out, data_format='vector')
# cube.save_to_svg(f_out)
