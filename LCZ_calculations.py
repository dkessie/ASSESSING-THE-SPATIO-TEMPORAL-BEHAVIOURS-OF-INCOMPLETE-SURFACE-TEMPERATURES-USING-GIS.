#final one

import os
import geopandas as gpd
from shapely.geometry import Polygon

#TWO THINGS TO CHANGE

cell_size = 200  # in meters
# Set the file path and grid cell size
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\LCZ8\study_area'


if cell_size == 200:
    down_step_size = 40
elif cell_size == 100:
    down_step_size = 20
elif cell_size == 50:
    down_step_size = 10
else:
    # Handle the case where cell_size is not one of the expected values
    # This could be an error, a default value, or some other behavior
    print(f"Unexpected cell size: {cell_size}")

# Set the step sizes and number of steps to move the grid

down_steps = 4
right_step_size = down_step_size
right_steps = down_steps



# Read the shapefile into a GeoDataFrame
gdf = gpd.read_file(file_path)

# If the CRS is not already in Web Mercator, project it
if gdf.crs != 'epsg:3857':
    gdf = gdf.to_crs('epsg:3857')

# Get the bounding box of the study area
xmin, ymin, xmax, ymax = gdf.total_bounds

# Create a grid of square polygons
cols = list(range(int(xmin), int(xmax), cell_size))
rows = list(range(int(ymin), int(ymax), cell_size))
rows.reverse()  # Reverse the order to match the expected grid order
polygons = []
for x in cols:
    for y in rows:
        polygons.append(Polygon([(x, y), (x+cell_size, y), (x+cell_size, y-cell_size), (x, y-cell_size)]))

# Create a GeoDataFrame of the grid polygons
grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs='epsg:3857')

# Intersect the grid with the study area
grid_gdf = gpd.overlay(grid_gdf, gdf, how='intersection')

# If necessary, project the grid back to the original CRS
if gdf.crs != 'epsg:3857':
    grid_gdf = grid_gdf.to_crs(gdf.crs)

# Write the grid to a shapefile
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
output_file = os.path.join(output_dir, 'grids.shp')
grid_gdf.to_file(output_file)

print('DONE CREATING GRIDS')

import geopandas as gpd
import os

import glob

# Get the .shp file within the file_path directory
study_area_path = glob.glob(os.path.join(file_path, '*.shp'))[0]


# Set the path to the shapefiles

grid_path = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\grids.shp"
output_folder = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"


# Read in the shapefiles
study_area = gpd.read_file(study_area_path)
grid = gpd.read_file(grid_path)



if down_step_size == 40:
    minimum_area = 39998
elif down_step_size == 20:
    minimum_area = 9998
elif down_step_size == 10:
    minimum_area = 2498
else:
    # Handle the case where down_step_size is not one of the specified values
    print("Invalid down_step_size value")


for i in range(down_steps+1):
    for j in range(right_steps+1):
        if i == 0 and j == 0:
            # First grid at (0, 0)
            clipped_grid = gpd.clip(grid, study_area)
            output_path = os.path.join(output_folder, f"{right_step_size}m_right_0m_down")
            clipped_grid.to_file(output_path)
        else:
            # Move grid down
            grid_down = grid.translate(xoff=0, yoff=-down_step_size*i)
            if j == 0:
                # Clip grid to study area
                clipped_grid = gpd.clip(grid_down, study_area)
                # Save grid to output folder
                output_path = os.path.join(output_folder, f"{right_step_size*j}m_right_{i*down_step_size}m_down")
            else:
                # Move grid right
                grid_moved = grid_down.translate(xoff=right_step_size*j, yoff=0)
                # Clip grid to study area
                clipped_grid = gpd.clip(grid_moved, study_area)
                # Save grid to output folder
                if j == right_steps:
                    output_path = os.path.join(output_folder, f"{right_steps*right_step_size}m_right_{i*down_step_size}m_down")
                else:
                    output_path = os.path.join(output_folder, f"{right_step_size*j}m_right_{i*down_step_size}m_down")
            clipped_grid.to_file(output_path)

print('DONE MOVING GRIDS')


import os

# Set the directory path
directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Iterate through all subfolders
for subdir, dirs, files in os.walk(directory):
    for file in files:
        # Check if file is a shapefile
        if file.endswith('.shp'):
            # Check if file name is not "grids"
            if file != "grids.shp":
                # Rename file name to "grids"
                os.rename(os.path.join(subdir, file), os.path.join(subdir, "grids.shp"))
                # Rename all extensions to ".shp", ".shx", etc.
                for extension in ['.cpg', '.dbf', '.prj', '.sbn', '.sbx', '.shp.xml', '.shx']:
                    try:
                        os.rename(os.path.join(subdir, file.split('.')[0] + extension), 
                                  os.path.join(subdir, "grids" + extension))
                    except FileNotFoundError:
                        # If extension is missing, move on to next extension
                        continue
print('DONE CHANGING ALL SHAPEFILE NAMES TO GRIDS')






import os
import geopandas as gpd

# set the path to the parent directory
parent_dir = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# set the minimum area threshold
min_area = minimum_area

# iterate through all subdirectories and shapefiles
for root, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.endswith(".shp"):
            # load the shapefile as a GeoDataFrame
            filepath = os.path.join(root, file)
            grid = gpd.read_file(filepath)
            
            # calculate the plan area of each polygon
            grid['P_area'] = grid.geometry.area
            
            # delete polygons with area less than the threshold
            grid = grid[grid['P_area'] >= min_area]
            
            # save the updated GeoDataFrame back to the shapefile
            grid.to_file(filepath, driver="ESRI Shapefile")
print('DONE REMOVING UNWANTED POLYGONS')

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# Specify the directory to search for shapefiles
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION'

# Set the line directory path
line_dir = os.path.join(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\line\line.shp')


# Iterate through all folders and subfolders in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a grids shapefile
        if file == 'grids.shp':
            # Read in the grid and polyline shapefiles
            grid = gpd.read_file(os.path.join(root, file))
            line = gpd.read_file(line_dir)

            # Create new columns in the grid dataframe to store the summed Day_T, Night_T, and wall_area values
            grid['D_T_Sum'] = 0
            grid['N_T_Sum'] = 0
            grid['W_area'] = 0
            grid['S_W_D_T'] = 0
            grid['S_W_N_T'] = 0
            grid['S_W_width'] = 0
            grid['S_HGT_AGL'] = 0
            
            # Iterate through each polygon in the grid dataframe
            for i, row in grid.iterrows():
                # Intersect the polygon with the polyline and select any intersecting lines
                intersecting_lines = line[line.intersects(row['geometry'])]
                # If there are intersecting lines, sum the Day_T, Night_T, and wall_area values and store the results in the respective columns for the polygon
                if not intersecting_lines.empty:
                    day_t_sum = intersecting_lines['Day_T'].sum()
                    night_t_sum = intersecting_lines['Night_T'].sum()
                    wall_area = intersecting_lines['wall_area'].sum()
                    Wall_D_T_sum = intersecting_lines['wall_D_T'].sum()
                    Wall_N_T_sum = intersecting_lines['wall_N_T'].sum()                    
                    wall_width_sum = intersecting_lines['wall_width'].sum()
                    hgt_agl_sum = intersecting_lines['HGT_AGL'].sum()
                    
                    
                    grid.loc[i, 'D_T_Sum'] = day_t_sum
                    grid.loc[i, 'N_T_Sum'] = night_t_sum
                    grid.loc[i, 'W_area'] = wall_area
                    grid.loc[i, 'S_W_D_T'] = Wall_D_T_sum
                    grid.loc[i, 'S_W_N_T'] = Wall_N_T_sum                    
                    grid.loc[i, 'S_W_width'] = wall_width_sum
                    grid.loc[i, 'S_HGT_AGL'] = hgt_agl_sum

                    
                else:
                    # If there are no intersecting lines, set the values to 0
                    grid.loc[i, 'D_T_Sum'] = 0
                    grid.loc[i, 'N_T_Sum'] = 0
                    grid.loc[i, 'W_area'] = 0
                    grid.loc[i, 'S_W_D_T'] = 0
                    grid.loc[i, 'S_W_N_T'] = 0
                    grid.loc[i, 'S_W_width'] = 0
                    grid.loc[i, 'S_HGT_AGL'] = 0

            # Reset the index of the grid dataframe
            grid = grid.reset_index(drop=True)

            # Save the updated grid shapefile with the new Day_T_Sum, Night_T_Sum, wall_area, plan_area, and ground_area columns and without any polygons that don't intersect with any polylines or buildings
            grid.to_file(os.path.join(root, file))

print('DONE EXTRACTING INFORMATION FROM POLYLINE')




#EXTRACTING plan_day_T
import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# Set the directory paths for the shapefiles and the raster data
shapefile_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\Day Temperature\Day_T.tif'

# Define a function to calculate the mean pixel value for each polygon in a shapefile
def calculate_mean_pixel_values(shapefile_path, raster_file):
    # Read the shapefile into a geopandas dataframe
    gdf = gpd.read_file(shapefile_path)

    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Loop through each subfolder in the directory
for subdir, dirs, files in os.walk(shapefile_dir):
    # Loop through each file in the subfolder
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp'):
            # Set the file path for the shapefile
            shapefile_path = os.path.join(subdir, file)

            # Calculate the mean pixel values for each polygon in the shapefile
            mean_pixel_values = calculate_mean_pixel_values(shapefile_path, raster_file)

            # Read the shapefile into a geopandas dataframe
            gdf = gpd.read_file(shapefile_path)

            # Add a new column to the dataframe with the mean pixel values
            gdf['AVG_Pl_DT'] = mean_pixel_values  # AVG_Pl_DT is the average Daytime plan temperature for one cell

            # Write the updated dataframe back to the shapefile
            gdf.to_file(shapefile_path, driver='ESRI Shapefile')


print("DONE EXTRACTING AVG_Plan_DayT")



#EXTRACTING plan_night_T
import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# Set the directory paths for the shapefiles and the raster data
shapefile_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\Night Temperature\Night_T.tif'

# Define a function to calculate the mean pixel value for each polygon in a shapefile
def calculate_mean_pixel_values(shapefile_path, raster_file):
    # Read the shapefile into a geopandas dataframe
    gdf = gpd.read_file(shapefile_path)

    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Loop through each subfolder in the directory
for subdir, dirs, files in os.walk(shapefile_dir):
    # Loop through each file in the subfolder
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp'):
            # Set the file path for the shapefile
            shapefile_path = os.path.join(subdir, file)

            # Calculate the mean pixel values for each polygon in the shapefile
            mean_pixel_values = calculate_mean_pixel_values(shapefile_path, raster_file)

            # Read the shapefile into a geopandas dataframe
            gdf = gpd.read_file(shapefile_path)

            # Add a new column to the dataframe with the mean pixel values
            gdf['AVG_Pl_NT'] = mean_pixel_values # AVG_Pl_NT is the average Nighttime plan temperature for one cell


            # Write the updated dataframe back to the shapefile
            gdf.to_file(shapefile_path, driver='ESRI Shapefile')


print("DONE EXTRACTING AVG_Plan_NightT")


#EXTRACTING SVF

import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# Set the directory paths for the shapefiles and the raster data
shapefile_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\DEM\SVF.tif'

# Define a function to calculate the mean pixel value for each polygon in a shapefile
def calculate_mean_pixel_values(shapefile_path, raster_file):
    # Read the shapefile into a geopandas dataframe
    gdf = gpd.read_file(shapefile_path)

    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Loop through each subfolder in the directory
for subdir, dirs, files in os.walk(shapefile_dir):
    # Loop through each file in the subfolder
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp'):
            # Set the file path for the shapefile
            shapefile_path = os.path.join(subdir, file)

            # Calculate the mean pixel values for each polygon in the shapefile
            mean_pixel_values = calculate_mean_pixel_values(shapefile_path, raster_file)

            # Read the shapefile into a geopandas dataframe
            gdf = gpd.read_file(shapefile_path)

            # Add a new column to the dataframe with the mean pixel values
            gdf['AVG_SVF'] = mean_pixel_values  # AVG_SVF is the average Sky View Factor for one cell

            # Write the updated dataframe back to the shapefile
            gdf.to_file(shapefile_path, driver='ESRI Shapefile')

print("DONE EXTRACTING AVG_SVF")





#Ground area and Avg ground day T
import os
import shutil
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats

# Specify the directory to search for shapefiles
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Specify the path to the buildings shapefile
buildings_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\buildings\buildings.shp'

# Specify the path to the raster file
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\Day Temperature\Day_T.tif'

# Define a function to calculate the mean pixel value for each polygon in a shapefile
def calculate_mean_pixel_values(gdf, raster_file):
    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Iterate through all folders and subfolders in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a grids shapefile
        if file == 'grids.shp':
            # Copy the grid files to a temporary location
            temp_dir = os.path.join(root, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            for ext in ['.dbf', '.prj', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx', '.cpg']:
                orig_path = os.path.join(root, f'grids{ext}')
                temp_path = os.path.join(temp_dir, f'grids{ext}')
                if os.path.exists(orig_path):
                    shutil.copy2(orig_path, temp_path)

            # Read in the grid and polyline shapefiles
            grid = gpd.read_file(os.path.join(temp_dir, 'grids.shp'))

            # Load the buildings shapefile and erase it from the grid shapefile
            buildings_gdf = gpd.read_file(buildings_path)
            grid = gpd.overlay(grid, buildings_gdf, how='difference')

            # Calculate the area of each polygon in the updated grid dataframe and add it to a new column called 'ground_area'
            grid['G_area'] = grid.geometry.area 

            # Calculate the mean pixel values for each polygon in the updated grid dataframe and add it to a new column called 'AVG_Pl_DT'
            grid['AvgGr_DT'] = calculate_mean_pixel_values(grid, raster_file)

            # Save the updated grid shapefile with the new Day_T_Sum, Night_T_Sum, wall_area, plan_area, ground_area, and AVG_Pl_DT columns and without any polygons that don't intersect with any polylines or buildings
            grid.to_file(os.path.join(temp_dir, 'grids.shp'))

            # Copy the ground area and AVG_Pl_DT columns from the temporary grid file to the original grid file
            temp_grid = gpd        .read_file(os.path.join(temp_dir, 'grids.shp'))
        orig_grid = gpd.read_file(os.path.join(root, 'grids.shp'))
    orig_grid['G_area'] = temp_grid['G_area']
    orig_grid['AvgGr_DT'] = temp_grid['AvgGr_DT']

        # Save the updated original grid shapefile with the new 'ground_area' and 'AVG_Pl_DT' columns
    orig_grid.to_file(os.path.join(root, 'grids.shp'))

        # Remove the temporary grid files
    shutil.rmtree(temp_dir)

print('DONE EXTRACTING G_area, Ground day T')





#Avg ground dNight T
import os
import shutil
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats

# Specify the directory to search for shapefiles
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Specify the path to the buildings shapefile
buildings_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\buildings\buildings.shp'

# Specify the path to the raster file
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\Night Temperature\Night_T.tif'
# Define a function to calculate the mean pixel value for each polygon in a shapefile
def calculate_mean_pixel_values(gdf, raster_file):
    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Iterate through all folders and subfolders in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a grids shapefile
        if file == 'grids.shp':
            # Copy the grid files to a temporary location
            temp_dir = os.path.join(root, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            for ext in ['.dbf', '.prj', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx', '.cpg']:
                orig_path = os.path.join(root, f'grids{ext}')
                temp_path = os.path.join(temp_dir, f'grids{ext}')
                if os.path.exists(orig_path):
                    shutil.copy2(orig_path, temp_path)

            # Read in the grid and polyline shapefiles
            grid = gpd.read_file(os.path.join(temp_dir, 'grids.shp'))

            # Load the buildings shapefile and erase it from the grid shapefile
            buildings_gdf = gpd.read_file(buildings_path)
            grid = gpd.overlay(grid, buildings_gdf, how='difference')

         

            # Calculate the mean pixel values for each polygon in the updated grid dataframe and add it to a new column called 'AVG_Pl_DT'
            grid['AVG_Gr_NT'] = calculate_mean_pixel_values(grid, raster_file)

            # Save the updated grid shapefile with the new Day_T_Sum, Night_T_Sum, wall_area, plan_area, ground_area, and AVG_Pl_DT columns and without any polygons that don't intersect with any polylines or buildings
            grid.to_file(os.path.join(temp_dir, 'grids.shp'))

            # Copy the ground area and AVG_Pl_DT columns from the temporary grid file to the original grid file
            temp_grid = gpd        .read_file(os.path.join(temp_dir, 'grids.shp'))
        orig_grid = gpd.read_file(os.path.join(root, 'grids.shp'))

    orig_grid['AVG_Gr_NT'] = temp_grid['AVG_Gr_NT']

        # Save the updated original grid shapefile with the new 'ground_area' and 'AVG_Pl_DT' columns
    orig_grid.to_file(os.path.join(root, 'grids.shp'))

        # Remove the temporary grid files
    shutil.rmtree(temp_dir)

print('DONE EXTRACTING Ground Night T')



import os
import glob
import geopandas as gpd

# Iterate through all subfolders in the specified directory
folder_path = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"
# Get the path to the shapefile in the specified directory
shapefile_path = glob.glob(os.path.join(folder_path, "*.shp"))[0]

# Read the shapefile into a GeoDataFrame
gdf = gpd.read_file(shapefile_path)

# Calculate TincO and TincI
gdf["TincO"] = ((gdf["G_area"] * gdf["AvgGr_DT"]) + gdf["S_W_D_T"]) / (gdf["W_area"] + gdf["G_area"])
gdf["TincI"] = ((gdf["G_area"] * gdf["AVG_Gr_NT"]) + gdf["S_W_N_T"]) / (gdf["W_area"] + gdf["G_area"])
# Calculate R_area and LambdaP
gdf["R_area"] = gdf["P_area"] - gdf["G_area"]
gdf["LambdaP"] = gdf["R_area"] / gdf["P_area"]


# Write the updated GeoDataFrame back to the shapefile
gdf.to_file(shapefile_path)




for folder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, folder)
    if os.path.isdir(subfolder_path):
        # Get the path to the shapefile in the subfolder
        shapefile_path = glob.glob(os.path.join(subfolder_path, "*.shp"))[0]
        
        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(shapefile_path)
        
      
        
        # Calculate TincO and TincI
        gdf["TincO"] = ((gdf["G_area"] * gdf["AvgGr_DT"]) + gdf["S_W_D_T"]) / (gdf["W_area"] + gdf["G_area"])
        gdf["TincI"] = ((gdf["G_area"] * gdf["AVG_Gr_NT"]) + gdf["S_W_N_T"]) / (gdf["W_area"] + gdf["G_area"])
        # Calculate R_area and LambdaP
        gdf["R_area"] = gdf["P_area"] - gdf["G_area"]
        gdf["LambdaP"] = gdf["R_area"] / gdf["P_area"]
        
        # Write the updated GeoDataFrame back to the shapefile
        gdf.to_file(shapefile_path)



print('DONE CALCLUATING TINCO and TINCI')


import os
from osgeo import gdal, ogr, gdalconst

# Set the directory path where shapefiles are located
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Set the output raster file format
driver = gdal.GetDriverByName('GTiff')

# Set the spatial resolution
spatial_resolution = down_step_size * 5

# List to store all the rasters to be overlaid
rasters_to_overlay = []

# Iterate through each subdirectory in the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp'):
            # Get the name of the shapefile
            shapefile_name = os.path.splitext(file)[0]
            # Open the shapefile
            shapefile = ogr.Open(os.path.join(root, file))
            # Get the layer of the shapefile
            layer = shapefile.GetLayer()
            # Get the extent of the layer
            xmin, xmax, ymin, ymax = layer.GetExtent()
            # Calculate the size of the raster
            xres = (xmax - xmin) / spatial_resolution
            yres = (ymax - ymin) / spatial_resolution
            # Set the output raster file name and path
            output_raster = os.path.join(root, shapefile_name + '.tif')
            # Create the output raster file
            output = driver.Create(output_raster, spatial_resolution, spatial_resolution, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
            # Set the projection of the output raster
            output.SetProjection(layer.GetSpatialRef().ExportToWkt())
            # Set the geotransform of the output raster
            output.SetGeoTransform((xmin, xres, 0, ymax, 0, -yres))
            # Set the pixel values of the output raster
            gdal.RasterizeLayer(output, [1], layer, options=["ATTRIBUTE=TincO"])
            # Read the pixel values of the output raster
            band = output.GetRasterBand(1)
            data = band.ReadAsArray()
            # Set all zero pixels to a NoData value
            nodata = 99999  # arbitrary NoData value
            data[data == 0] = nodata
            band.SetNoDataValue(nodata)
            # Write the modified pixel values to the output raster
            band.WriteArray(data)
            # Close the output raster file
            output = None
            
            # Add the raster to the list of rasters to be overlaid
            rasters_to_overlay.append(output_raster)

# Overlay all the rasters in the list into one raster
final_overlay = os.path.join(directory, 'FINAL_OVERLAY.tif')
gdal.Warp(final_overlay, rasters_to_overlay, format='GTiff')

print('DONE OVERLAYING ALL TincO RASTERS')







import os
import geopandas as gpd

# Set up the directory path
dir_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Iterate through all subdirectories and files in the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith('.shp'):
            # Load the shapefile into a GeoDataFrame
            filepath = os.path.join(root, file)
            gdf = gpd.read_file(filepath)
            
            # Add two columns for x and y positions
            gdf['x_position'] = gdf.geometry.centroid.x
            gdf['y_position'] = gdf.geometry.centroid.y
            
            # Save the updated shapefile
            gdf.to_file(filepath)
print('DONE ADDING CORDINATES')


#SECOND X AND Y POSITION
import os
import geopandas as gpd
import numpy as np

# Set up the directory path
dir_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Define step sizes for x and y coordinates
x_step = 1.0
y_step = 1.0

# Keep track of the position of the first polygon in the entire directory
first_pos = None

# Iterate through all subdirectories and files in the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith('.shp'):
            # Load the shapefile into a GeoDataFrame
            filepath = os.path.join(root, file)
            gdf = gpd.read_file(filepath)
            
            # Calculate the position of each polygon relative to the position of the first polygon in the entire directory
            if first_pos is None:
                # Calculate the position of the first polygon in the entire directory
                first_pos = gdf.geometry.centroid.iloc[0]
                
            # Calculate the relative position of each polygon
            gdf['x_N_position'] = (gdf.geometry.centroid.x - first_pos.x) / x_step
            gdf['y_N_position'] = (gdf.geometry.centroid.y - first_pos.y) / y_step
            
            # Save the updated shapefile
            gdf.to_file(filepath)
            
print('DONE ADDING CREATING CORDINATES IN RELATION TO THE LOCATION OF THE FIRST CELL IN THE FIRST GRID')




import os
from osgeo import gdal, ogr, gdalconst

# Set the directory path where shapefiles are located
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Set the output raster file format
driver = gdal.GetDriverByName('GTiff')

# Set the spatial resolution
spatial_resolution = down_step_size * 5

# List to store all the rasters
rasters = []

# Iterate through each subdirectory in the given directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp'):
            # Get the name of the shapefile
            shapefile_name = os.path.splitext(file)[0]
            # Open the shapefile
            shapefile = ogr.Open(os.path.join(root, file))
            # Get the layer of the shapefile
            layer = shapefile.GetLayer()
            # Get the extent of the layer
            xmin, xmax, ymin, ymax = layer.GetExtent()
            # Calculate the size of the raster
            xres = (xmax - xmin) / spatial_resolution
            yres = (ymax - ymin) / spatial_resolution
            # Set the output raster file name and path
            output_raster = os.path.join(root, shapefile_name + '.tif')
            # Create the output raster file
            output = driver.Create(output_raster, spatial_resolution, spatial_resolution, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
            # Set the projection of the output raster
            output.SetProjection(layer.GetSpatialRef().ExportToWkt())
            # Set the geotransform of the output raster
            output.SetGeoTransform((xmin, xres, 0, ymax, 0, -yres))
            # Set the pixel values of the output raster
            gdal.RasterizeLayer(output, [1], layer, options=["ATTRIBUTE=TincO"])
            # Read the pixel values of the output raster
            band = output.GetRasterBand(1)
            data = band.ReadAsArray()
            # Set all zero pixels to a NoData value
            nodata = 99999  # arbitrary NoData value
            data[data == 0] = nodata
            band.SetNoDataValue(nodata)
            # Write the modified pixel values to the output raster
            band.WriteArray(data)
            # Close the output raster file
            output = None
            
            # Add the raster to the list of rasters
            rasters.append(output_raster)

print('DONE CREATING THE RASTER FILES OF THE GROUND AREA')

import os
import glob
from os.path import join, splitext, basename
import geopandas as gpd
from shapely.geometry.point import Point


def polygon_to_point(row):
    # calculate the bounding box of the polygon
    bounds = row.geometry.bounds

    # calculate the x and y coordinates of the midpoint of the bounding box
    x_mid = (bounds[0] + bounds[2]) / 2.0
    y_mid = (bounds[1] + bounds[3]) / 2.0

    # create a new Point object at the midpoint
    return Point(x_mid, y_mid)


def process_shapefile(filename):
    # load shapefile
    df = gpd.read_file(filename)

    # convert polygons to points
    df['geometry'] = df.apply(polygon_to_point, axis=1)

    # save new point shapefile in same directory as original shapefile
    output_filename = splitext(filename)[0] + '_points.shp'
    df.to_file(output_filename)


def process_directory(directory):
    # find all shapefiles in the directory and its subfolders
    search_pattern = join(directory, '**', 'grids.shp')
    shapefile_paths = glob.glob(search_pattern, recursive=True)

    for shapefile_path in shapefile_paths:
        # process shapefile
        process_shapefile(shapefile_path)


process_directory(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids')


print("DONE CONVERTING POLYGONS TO POINTS (grids_points_shapefiles)")



import os
import glob
import numpy as np
from osgeo import gdal, ogr

# Define the resolution
resolution = down_step_size

# Define the function to convert point shapefile to raster
def point2raster(point_shapefile):
    # Open the point shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(point_shapefile, 0)
    layer = datasource.GetLayer()
    extent = layer.GetExtent()
    x_min, x_max, y_min, y_max = extent

    # Calculate the number of rows and columns
    cols = int((x_max - x_min) / resolution)
    rows = int((y_max - y_min) / resolution)

    # Check if cols and rows are greater than 0
    if cols <= 0 or rows <= 0:
        print(f"Invalid dimensions for raster for file {point_shapefile}. cols: {cols}, rows: {rows}. Skipping this file.")
        return

    # Create the output raster file
    raster_filename = os.path.splitext(point_shapefile)[0] + ".tif"
    output_raster = gdal.GetDriverByName('GTiff').Create(raster_filename, cols, rows, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform((x_min, resolution, 0, y_max, 0, -resolution))
    output_raster.SetProjection(layer.GetSpatialRef().ExportToWkt())

    # Convert each point to a pixel value
    layer.SetAttributeFilter(None)
    gdal.RasterizeLayer(output_raster, [1], layer, options=["ATTRIBUTE=TincO"])

    # Close the files
    output_raster = None
    datasource = None

# Define the main function to iterate through the directory and subfolders
def main():
    # Iterate through the directory and subfolders
    for root, dirs, files in os.walk(r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"):
        for file in files:
            if file.endswith(".shp") and file.startswith("grids_points"):
                # Convert the point shapefile to raster
                point_shapefile = os.path.join(root, file)
                point2raster(point_shapefile)

if __name__ == '__main__':
    main()

print('DONE CONVERTING grids_to_points_shapefiles TO RASTERS')


import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge

# Set the directory path to search for the geotiff files
directory_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Find all the geotiff files in the directory and its subfolders
tif_files = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.tif') and file == 'grids_points.tif':
            tif_files.append(os.path.join(root, file))

# Merge the geotiff files
src_files_to_mosaic = []
for file in tif_files:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)

mosaic, out_trans = merge(src_files_to_mosaic)

# Remove all pixels with 0 values
mosaic[mosaic == 0] = np.nan

# Save the merged geotiff file
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans})

out_file_path = os.path.join(directory_path, 'Final_grids_to_points_Raster.tif')

with rasterio.open(out_file_path, "w", **out_meta) as dest:
    dest.write(mosaic)


print('Done combining grids_to_points rasters to Final_grids_to_points_Raster')


import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

# Set file paths
raster_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Final_grids_to_points_Raster.tif'
shapefile_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\grids.shp'
output_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Final_Masked_grids_to_points_Raster.tif'

# Read in shapefile and raster file
gdf = gpd.read_file(shapefile_path)
with rasterio.open(raster_path) as src:
    out_image, out_transform = mask(src, gdf.geometry, crop=True)
    out_meta = src.meta

# Remove all pixels with 0 values
out_image = np.where(out_image == 0, np.nan, out_image)

# Update metadata
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 "nodata": np.nan})

# Write out masked raster
with rasterio.open(output_path, 'w', **out_meta) as dest:
    dest.write(out_image[0], 1)

print('Done creating Final_Masked_grids_to_points_Raster')






#20m
import os
from osgeo import ogr

# Define the directory where the shapefiles are stored
directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Define the buffer distance (in meters)
buffer_distance = down_step_size

# Iterate through all subdirectories and files in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a point shapefile
        if file.endswith(".shp") and "grids_points" in file:
            # Open the point shapefile
            point_ds = ogr.Open(os.path.join(root, file))
            point_lyr = point_ds.GetLayer()

            # Create a new polygon shapefile
            polygon_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(os.path.join(root, file[:-4] + "_polygons.shp"))
            polygon_lyr = polygon_ds.CreateLayer("polygon", srs=point_lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)

            # Add attribute fields to the polygon layer
            point_defn = point_lyr.GetLayerDefn()
            for i in range(point_defn.GetFieldCount()):
                field_defn = point_defn.GetFieldDefn(i)
                polygon_lyr.CreateField(field_defn)

            # Iterate through all points in the point layer
            for point_feat in point_lyr:
                point_geom = point_feat.geometry()

                # Create a buffer around the point geometry
                buffer_geom = point_geom.Buffer(buffer_distance)

                # Convert the buffer geometry to a polygon geometry
                (minX, maxX, minY, maxY) = buffer_geom.GetEnvelope()
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(minX, minY)
                ring.AddPoint(minX + buffer_distance, minY)
                ring.AddPoint(minX + buffer_distance, minY + buffer_distance)
                ring.AddPoint(minX, minY + buffer_distance)
                ring.AddPoint(minX, minY)
                square_geom = ogr.Geometry(ogr.wkbPolygon)
                square_geom.AddGeometry(ring)

                # Create a new polygon feature in the polygon layer
                polygon_feat = ogr.Feature(polygon_lyr.GetLayerDefn())
                polygon_feat.SetGeometry(square_geom)

                # Copy attribute values from the corresponding point feature
                for i in range(point_defn.GetFieldCount()):
                    value = point_feat.GetField(i)
                    polygon_feat.SetField(i, value)

                # Add the new polygon feature to the polygon layer
                polygon_lyr.CreateFeature(polygon_feat)

            # Clean up
            del point_ds
            del polygon_ds
print('DONE CREATING BUFFERS AROUND GRIDS POINTS')


import os
import glob
import numpy as np
from osgeo import gdal, ogr

# set the output spatial resolution in meters
pixel_size = down_step_size

# get a list of all the shapefiles in the directory and subfolders
shapefiles = glob.glob(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\**\grids_points_polygons.shp', recursive=True)

# iterate through each shapefile and convert it to a raster
for shapefile in shapefiles:
    # open the shapefile using OGR
    datasource = ogr.Open(shapefile)
    layer = datasource.GetLayer()

    # get the extent of the layer
    extent = layer.GetExtent()

    # calculate the size of the output raster based on the pixel size
    x_size = int((extent[1] - extent[0]) / pixel_size)
    y_size = int((extent[3] - extent[2]) / pixel_size)

    # create a new raster dataset using GDAL
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(os.path.join(os.path.dirname(shapefile), os.path.splitext(os.path.basename(shapefile))[0] + '.tif'), x_size, y_size, 1, gdal.GDT_Float32)

    # set the projection and geotransform of the output raster to match the input shapefile
    out_raster.SetProjection(layer.GetSpatialRef().ExportToWkt())
    out_raster.SetGeoTransform((extent[0], pixel_size, 0, extent[3], 0, -pixel_size))

    # rasterize the shapefile using the TincO field as the pixel value
    gdal.RasterizeLayer(out_raster, [1], layer, options=["ATTRIBUTE=TincO"])

    # open the output raster as a numpy array and replace any zero values with NaN
    out_array = out_raster.GetRasterBand(1).ReadAsArray()
    out_array[out_array == 0] = np.nan

    # write the modified array back to the raster
    out_raster.GetRasterBand(1).WriteArray(out_array)

    # close the input and output datasets
    datasource = None
    out_raster = None

print('DONE CONVERTING BUFFER TO RASTER FILES')



import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

# Define the directory containing the input raster files
input_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Find all the GeoTIFF raster files in the input directory and its subdirectories
input_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('grids_points_polygons.tif'):
            input_files.append(os.path.join(root, file))

# Open all the input raster files
src_files_to_mosaic = []
for file in input_files:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)

# Merge the input raster files into a single raster
mosaic, out_trans = merge(src_files_to_mosaic)

# Write the merged raster to a new GeoTIFF file
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans})
out_file = os.path.join(input_dir, 'final_raster.tif')
with rasterio.open(out_file, "w", **out_meta) as dest:
    dest.write(mosaic)

print('DONE CREATING FINAL RASTER')




import os
import pandas as pd
import shapefile

# Define the directory where the shapefiles are stored
dir_path = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Create an empty list to store the DataFrames for each shapefile
dfs = []

# Loop through each subfolder in the directory
for subdir, _, _ in os.walk(dir_path):
    # Only process subfolders that contain shapefiles
    if any(f.endswith('.shp') for f in os.listdir(subdir)):
        # Load the shapefile into a PyShp shapefile object
        shape_file = shapefile.Reader(os.path.join(subdir, 'grids.shp'))

        # Extract the attribute table from the shapefile
        fields = shape_file.fields[1:]
        records = shape_file.records()

        # Convert the attribute table to a Pandas DataFrame and append it to the list
        df = pd.DataFrame(records, columns=[field[0] for field in fields])
        dfs.append(df)

        # Write the DataFrame to an Excel file and save it in the subfolder
        df.to_excel(os.path.join(subdir, 'grids.xlsx'), index=False)

# Combine all the DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to an Excel file and save it in the main directory
combined_df.to_excel(os.path.join(dir_path, 'combined_grids.xlsx'), index=False)

print ('DONE CREATING EXCEL FILES')



import os
import glob
import geopandas as gpd
import pandas as pd

# Define the directory to search for shapefiles
rootdir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Define the name of the output shapefile
FINAL_SHAPEFILE = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\FINAL_SHAPEFILE.shp'

# Create an empty GeoDataFrame to store the merged shapefiles
merged_gdf = gpd.GeoDataFrame()

# Iterate through the directory and subfolders
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Check if the file is a grids_points_polygons shapefile
        if file.endswith('grids_points_polygons.shp'):
            # Read the shapefile into a GeoDataFrame
            gdf = gpd.read_file(os.path.join(subdir, file))
            # Append the GeoDataFrame to the merged GeoDataFrame
            merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)

# Write the merged GeoDataFrame to a shapefile
merged_gdf.to_file(FINAL_SHAPEFILE)


print('DONE CREATING FINAL SHAPEFILE')


import os
import geopandas as gpd

# Set the root directory path
root_directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Recursive function to find shapefiles with the name "FINAL_SHAPEFILE"
def find_shapefiles(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.shp') and file == 'FINAL_SHAPEFILE.shp':
                shapefile_path = os.path.join(root, file)
                process_shapefile(shapefile_path)

# Process the shapefile: calculate latitude and longitude fields
def process_shapefile(shapefile_path):
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Set the CRS to EPSG 3857
    gdf.crs = 'epsg:3857'
    
    # Create new 'latitude' and 'longitude' fields of type double
    gdf['latitude'] = gdf['geometry'].centroid.y.astype(float)
    gdf['longitude'] = gdf['geometry'].centroid.x.astype(float)
    
    # Save the updated shapefile
    gdf.to_file(shapefile_path)

# Call the function to find and process shapefiles
find_shapefiles(root_directory)

print('DONE CREATING LAT AND LONG')

import os
import pandas as pd
import geopandas as gpd

# Set the file path of the shapefile
shapefile_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\FINAL_SHAPEFILE.shp'

# Read the shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Convert attribute table to pandas DataFrame
attribute_table = pd.DataFrame(gdf.drop('geometry', axis=1))

# Set the output file path
output_file_path = os.path.splitext(shapefile_path)[0] + '_combined_grids_with_coordinates.xlsx'

# Save attribute table to XLSX file
attribute_table.to_excel(output_file_path, index=False)

print("Attribute table converted and saved as XLSX file.")




import os
import glob
import geopandas as gpd

# Define the directory to search for shapefiles
rootdir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Define the name of the output shapefile
FINAL_SHAPEFILE = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\FINAL_POINT_SHAPEFILE.shp'

# Create an empty GeoDataFrame to store the merged shapefiles
merged_gdf = gpd.GeoDataFrame()

# Iterate through the directory and subfolders
counter = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Check if the file is a grids_points shapefile
        if file.endswith('grids_points.shp'):
            # Read the shapefile into a GeoDataFrame
            gdf = gpd.read_file(os.path.join(subdir, file))
            # Append the GeoDataFrame to the merged GeoDataFrame
            merged_gdf = merged_gdf.append(gdf)
            counter += 1
            print(f'Merged shapefile {counter}: {file}')

# Check if the output shapefile already exists
if os.path.isfile(FINAL_SHAPEFILE):
    raise ValueError('Output shapefile already exists. Please delete it or choose a different output filename.')

# Set the geometry column in the merged GeoDataFrame
merged_gdf = merged_gdf.set_geometry('geometry')

# Write the merged GeoDataFrame to a shapefile
merged_gdf.to_file(FINAL_SHAPEFILE)

print('Done creating final_point_shapefile.')




import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# Set the directory paths for the shapefiles and the raster data
shapefile_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
raster_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\Day Temperature\Day_T.tif'
buildings_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\buildings\buildings.shp'

# Read the buildings shapefile into a geopandas dataframe
buildings_gdf = gpd.read_file(buildings_file)

# Define a function to calculate the mean pixel value for each polygon in a shapefile after erasing the buildings
def calculate_mean_pixel_values(shapefile_path, raster_file, buildings_gdf):
    # Read the shapefile into a geopandas dataframe
    gdf = gpd.read_file(shapefile_path)

    # Erase the buildings from the shapefile
    gdf = gpd.overlay(gdf, buildings_gdf, how='difference')

    # Get the geometry of the shapefile and convert it to the same CRS as the raster data
    geometry = gdf.geometry.to_crs(rasterio.open(raster_file).crs)

    # Use the zonal_stats() function to calculate the mean pixel value for each polygon in the shapefile
    stats = zonal_stats(geometry, raster_file, stats=['mean'])

    # Create a list of mean pixel values for each polygon
    mean_pixel_values = [stat['mean'] for stat in stats]

    return mean_pixel_values

# Loop through each subfolder in the directory
for subdir, dirs, files in os.walk(shapefile_dir):
    # Loop through each file in the subfolder
    for file in files:
        # Check if the file is a shapefile
        if file.endswith('.shp') and os.path.basename(file).startswith("grid"):
            # Set the file path for the shapefile
            shapefile_path = os.path.join(subdir, file)

            # Calculate the mean pixel values for each polygon in the shapefile
            mean_pixel_values = calculate_mean_pixel_values(shapefile_path, raster_file, buildings_gdf)

            # Read the shapefile into a geopandas dataframe
            gdf = gpd.read_file(shapefile_path)

            # Erase the buildings from the shapefile
            gdf = gpd.overlay(gdf, buildings_gdf, how='difference')

            # If the length of the GeoPandas dataframe is greater than the length of the mean pixel values list, skip the shapefile
            if len(gdf) > len(mean_pixel_values):
                print(f"Skipping {file}: Length of GeoPandas dataframe ({len(gdf)}) is greater than the length of mean pixel values list ({len(mean_pixel_values)})")
                continue

            # Add a new column to the dataframe with the mean pixel values
            gdf['Ave_GR_DT'] = mean_pixel_values # AvgGr_DT is the average Daytime ground temperature for one cell

            # Write the updated dataframe back to the shapefile
            gdf.to_file(shapefile_path, driver='ESRI Shapefile')
print('DONE ERASING ROOFS FROM POLYGONS')





import os
from osgeo import gdal, ogr

# Set the input shapefile path and name
shapefile_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
shapefile_name = 'FINAL_SHAPEFILE.shp'

# Set the output directory for the rasters
output_directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'

# Set the pixel size
pixel_size = down_step_size


def create_raster(input_shapefile, field_name, output_raster):
    # Open the shapefile
    shapefile = ogr.Open(input_shapefile)
    layer = shapefile.GetLayer()

    # Get the extent and spatial reference
    extent = layer.GetExtent()
    spatial_ref = layer.GetSpatialRef()

    # Calculate the raster dimensions based on the extent and pixel size
    x_min, x_max, y_min, y_max = extent
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    # Create the raster dataset
    driver = gdal.GetDriverByName('GTiff')
    raster_dataset = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Float32)

    # Set the projection and geotransform
    raster_dataset.SetProjection(spatial_ref.ExportToWkt())
    raster_dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

    # Get the layer's field index
    field_index = layer.GetLayerDefn().GetFieldIndex(field_name)

    # Create a new raster band
    band = raster_dataset.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Rasterize the shapefile field
    gdal.RasterizeLayer(raster_dataset, [1], layer, options=['ATTRIBUTE=' + field_name])

    # Close the shapefile and raster dataset
    shapefile = None
    raster_dataset = None

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Field mappings for output filenames
field_mappings = {
    'AVG_Pl_DT': 'Daytime Tplan',
    'AVG_Pl_NT': 'Nighttime Tplan',
    'TincO': 'Daytime Tped',
    'TincI': 'Nighttime Tped'
}

# Iterate over the fields and create rasters for each
fields = ['AVG_Pl_DT', 'AVG_Pl_NT', 'TincO', 'TincI']
for field in fields:
    field_name = field_mappings.get(field, field)
    output_raster = os.path.join(output_directory, field_name + '.tif')
    input_shapefile = os.path.join(shapefile_path, shapefile_name)
    create_raster(input_shapefile, field, output_raster)

print('Rasters created successfully!')

# Input shapefile and raster paths
shapefile_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\buildings\buildings.shp'
raster_path = os.path.join(output_directory, 'Daytime Tplan.tif')

# Output directory and shapefile name
output_directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
output_shapefile_name = 'Buildings.shp'

# Convert raster to shapefile
gdal.UseExceptions()
src_ds = gdal.Open(raster_path)
srcband = src_ds.GetRasterBand(1)
dst_layername = "POLYGONIZED_STUFF"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(os.path.join(output_directory, "temp_shape.shp"))
dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)

gdal.Polygonize(srcband, None, dst_layer, -1, [], callback=None)
dst_ds.Destroy()
src_ds = None

# Now perform clipping operation
# Create a new shapefile from the intersection of the input shapefile and the raster shapefile
input_ds = ogr.Open(shapefile_path)
input_layer = input_ds.GetLayer()

raster_ds = ogr.Open(os.path.join(output_directory, "temp_shape.shp"))
raster_layer = raster_ds.GetLayer()

output_ds = drv.CreateDataSource(os.path.join(output_directory, output_shapefile_name))
output_layer = output_ds.CreateLayer(output_shapefile_name, input_layer.GetSpatialRef())

# Add fields from input to output
inLayerDefn = input_layer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
    fieldDefn = inLayerDefn.GetFieldDefn(i)
    output_layer.CreateField(fieldDefn)

# Perform the intersection
input_layer.Clip(raster_layer, output_layer)

# Close datasets
input_ds.Destroy()
output_ds.Destroy()
raster_ds.Destroy()

input_ds = None
output_ds = None
raster_ds = None

# Remove temporary shapefiles
for filename in os.listdir(output_directory):
    if filename.startswith("temp_shape."):
        os.remove(os.path.join(output_directory, filename))

print("Clipping done!")


#Move files to needed folder

import os
import shutil

# Specify the source directory
source_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Specify the target directory
target_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files"

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.mkdir(target_directory)

# List of files to be moved
files_to_move = [
    "Daytime Tplan.tif",
    "Nighttime Tplan.tif",
    "Daytime Tped.tif",
    "Nighttime Tped.tif",
    "Buildings.dbf",
    "Buildings.prj",
    "Buildings.shp",
    "Buildings.shx",
    "combined_grids.xlsx"
]

# Move the files to the target directory
for file_name in files_to_move:
    source_file_path = os.path.join(source_directory, file_name)
    target_file_path = os.path.join(target_directory, file_name)
    if os.path.exists(source_file_path):
        shutil.move(source_file_path, target_file_path)
        print(f"Moved '{file_name}' to 'Needed files' folder.")
    else:
        print(f"File '{file_name}' not found in the source directory.")

print('DONE MOVING BUILDINGS TO NEEDED FILES')


import os
import glob
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar

# Set the paths to the shapefile and directory containing raster files
shapefile_path = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Buildings.shp"
raster_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files"

# Read the shapefile
shapes = gpd.read_file(shapefile_path)

# Get the list of raster files in the directory
raster_files = glob.glob(os.path.join(raster_directory, "*.tif"))

# Process each raster file
for rasterfile_path in raster_files:
    # Open the raster file
    raster = rasterio.open(rasterfile_path)

    # Create the figure with one axis
    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot the raster file and shapefile on the axis
    show(raster, ax=ax, cmap="inferno")
    shapes.plot(ax=ax, facecolor="none", edgecolor="black")

    # Get the valid pixel values of the raster
    raster_data = raster.read(1, masked=True)
    valid_pixel_values = raster_data.compressed()

    # Compute the minimum and maximum pixel values from the valid pixels and round to the nearest whole number
    raster_min = round(valid_pixel_values.min())
    raster_max = round(valid_pixel_values.max())

    # Normalize the pixel values between 0 and 1
    norm = Normalize(vmin=raster_min, vmax=raster_max)

    # Create a ScalarMappable object with the colormap and normalization
    cmap = get_cmap("inferno")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create a colorbar using the ScalarMappable on the axis
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)

    raster_label = os.path.splitext(os.path.basename(rasterfile_path))[0]

    # Check for 'plan' in the label and make it subscript
    raster_label = raster_label.replace("Tplan", r"$T_{plan}$")
    raster_label = raster_label.replace("Tped", r"$T_{ped}$")

    # Append degrees Celsius to the label
    raster_label += r" (°C)"

    cbar.set_label(raster_label)

    # Remove x-axis and y-axis from the axis
    ax.axis('off')

 

    # Add scale bar and adjust its position
    scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower center', pad=-5,
                        color='black', frameon=True)
    ax.add_artist(scalebar) #decrease the pad value to move the scale bar downwards. you can also change to lowerr right or left

    # Add north arrow to the axis and adjust its position
    arrow_props = dict(arrowstyle="->", mutation_scale=100, linewidth=1.5, color='black')
    ax.annotate('N', xy=(0.05, 1.25), xycoords='axes fraction', xytext=(0.05, 1.1),
                 textcoords='axes fraction', arrowprops=arrow_props, ha='center', va='center', fontsize=8) #change the y values of the (ie x, y) to move the north arrow up higher

    # Create a folder called "jpeg_files" in the raster_directory if it doesn't exist
    jpeg_directory = os.path.join(raster_directory, "jpeg_files")
    os.makedirs(jpeg_directory, exist_ok=True)

    # Extract the filename from the raster file path
    raster_filename = os.path.splitext(os.path.basename(rasterfile_path))[0]

    # Save the figure as a JPEG file in the "jpeg_files" folder
    output_filename = raster_filename + ".jpeg"
    output_path = os.path.join(jpeg_directory, output_filename)
    plt.savefig(output_path, format='jpeg')

    # Close the figure
    plt.close(fig)

print('DONE CREATING JPEG FILES')


import os
import shutil

# Define the source file path
source_file = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\FINAL_SHAPEFILE_combined_grids_with_coordinates.xlsx"

# Define the destination folder path
destination_folder = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files"

# Define the new filename
new_filename = "Final_File_with_coordinates.xlsx"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Construct the new file path with the destination folder and new filename
new_file_path = os.path.join(destination_folder, new_filename)

# Move the file to the destination folder and rename it
shutil.move(source_file, new_file_path)

print("Final excel file with cordinates moved to needed files and renamed successfully.")


import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory path in a single variable
dir_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# Path to the Excel file
file_path = os.path.join(dir_path, 'Final_File_with_coordinates.xlsx')

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Extract the required fields for Daytime plot
x_values_dt = df['AVG_Pl_DT']
y_values_dt = df['TincO']

# Create a scatter plot for Daytime
plt.figure(figsize=(10, 8))
plt.scatter(x_values_dt, y_values_dt, facecolors='none', edgecolors='blue')
plt.xlabel('$T_{plan}$ (°C)', fontsize=18)
plt.ylabel('$T_{ped}$ (°C)', fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.title('Daytime', fontsize=20)

# Get the minimum and maximum values for x and y for Daytime
x_min_dt = x_values_dt.min()
x_max_dt = x_values_dt.max()
y_min_dt = y_values_dt.min()
y_max_dt = y_values_dt.max()

# Plot the identity line for Daytime
min_val_dt = min(x_min_dt, y_min_dt)
max_val_dt = max(x_max_dt, y_max_dt)
plt.plot([min_val_dt, max_val_dt], [min_val_dt, max_val_dt], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for Daytime
plt.xlim(min_val_dt - 1, max_val_dt + 1)
plt.ylim(min_val_dt - 1, max_val_dt + 1)

# Save the Daytime plot as a JPEG file
plt.savefig(os.path.join(dir_path, 'Daytime_scatter.jpeg'), format='jpeg')
plt.close()

print('DONE CREATING DAYTIME SCATTER PLOTS')

# Extract the required fields for Nighttime plot
x_values_nt = df['AVG_Pl_NT']
y_values_nt = df['TincI']


# Create a scatter plot for Nighttime
plt.figure(figsize=(10, 8))
plt.scatter(x_values_nt, y_values_nt, facecolors='none', edgecolors='blue')
plt.xlabel('$T_{plan}$ (°C)', fontsize=18)
plt.ylabel('$T_{ped}$ (°C)', fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.title('Nighttime', fontsize=20)


# Get the minimum and maximum values for x and y for Nighttime
x_min_nt = x_values_nt.min()
x_max_nt = x_values_nt.max()
y_min_nt = y_values_nt.min()
y_max_nt = y_values_nt.max()

# Plot the identity line for Nighttime
min_val_nt = min(x_min_nt, y_min_nt)
max_val_nt = max(x_max_nt, y_max_nt)
plt.plot([min_val_nt, max_val_nt], [min_val_nt, max_val_nt], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for Nighttime
plt.xlim(min_val_nt - 1, max_val_nt + 1)
plt.ylim(min_val_nt - 1, max_val_nt + 1)

# Save the Nighttime plot as a JPEG file
plt.savefig(os.path.join(dir_path, 'Nighttime_scatter.jpeg'), format='jpeg')
plt.close()

print('DONE CREATING NIGHTTIME SCATTER PLOTS')







#daytime regression




import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_DT']]
y = df['TincO']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()

# Create a scatter plot of actual TincO (x-axis) vs. predicted TincO (y-axis)
plt.figure(figsize=(10, 8))
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Calculated $T_{ped}$ (°C)', fontsize=18)
plt.ylabel('Predicted $T_{ped}$ (°C)', fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.title('Regression Model: Calculated vs. Predicted $T_{ped}$ (°C)')

# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
plt.savefig(os.path.join(output_dir, 'Daytime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincO'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)






#Nighttime regression

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_NT']]
y = df['TincI']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()


plt.figure(figsize=(10, 8))
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Calculated $T_{ped}$ (°C)', fontsize=18)
plt.ylabel('Predicted $T_{ped}$ (°C)', fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.title('Regression Model: Calculated vs. Predicted $T_{ped}$ (°C)')



# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
plt.savefig(os.path.join(output_dir, 'Nighttime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincI'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)


print('DONE CREATING MULTIPLE REGRESSIONS')





import pandas as pd
import random
import matplotlib.pyplot as plt

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Filter the DataFrame to select points with AVG_Pl_DT between 50 and 55
above_50 = df[(df['AVG_Pl_DT'] > 50) & (df['AVG_Pl_DT'] < 55)]

# Filter the DataFrame to select points with TincO between 40 and 45
between_40_45 = df[(df['TincO'] > 50) & (df['TincO'] < 55)]

# Randomly select 10 points with AVG_Pl_DT between 50 and 55, or all rows if fewer than 10
if len(above_50) >= 10:
    random_points_avg_pl_dt = above_50.sample(n=10, random_state=42)
else:
    random_points_avg_pl_dt = above_50.sample(n=len(above_50), random_state=42)

# Randomly select 10 points with TincO between 40 and 45, or all rows if fewer than 10
if len(between_40_45) >= 10:
    random_points_tinco = between_40_45.sample(n=10, random_state=42)
else:
    random_points_tinco = between_40_45.sample(n=len(between_40_45), random_state=42)

# Combine the randomly selected points
random_points = pd.concat([random_points_avg_pl_dt, random_points_tinco])

# Extract the required fields for Daytime plot
x_values_dt = random_points['AVG_Pl_DT']
y_values_dt = random_points['TincO']
lambda_p_values = random_points['LambdaP']
s_hgt_agl_values = random_points['S_HGT_AGL']
avg_svf_values = random_points['AVG_SVF']
w_area_values = random_points['W_area']

# Normalize the Lambda P values to a range between 10 and 50
min_lambda_p = lambda_p_values.min()
max_lambda_p = lambda_p_values.max()
normalized_lambda_p = 10 + ((lambda_p_values - min_lambda_p) / (max_lambda_p - min_lambda_p)) * 40

# Normalize S_HGT_AGL, AVG_SVF, and W_area to a range between 10 and 50
min_s_hgt_agl = s_hgt_agl_values.min()
max_s_hgt_agl = s_hgt_agl_values.max()
normalized_s_hgt_agl = 10 + ((s_hgt_agl_values - min_s_hgt_agl) / (max_s_hgt_agl - min_s_hgt_agl)) * 40

min_avg_svf = avg_svf_values.min()
max_avg_svf = avg_svf_values.max()
normalized_avg_svf = 10 + ((avg_svf_values - min_avg_svf) / (max_avg_svf - min_avg_svf)) * 40

min_w_area = w_area_values.min()
max_w_area = w_area_values.max()
normalized_w_area = 10 + ((w_area_values - min_w_area) / (max_w_area - min_w_area)) * 40

# Create a scatter plot for LambdaP
fig1, ax1 = plt.subplots()
ax1.scatter(x_values_dt, y_values_dt, s=normalized_lambda_p, facecolors='none', edgecolors='blue')
ax1.set_xlabel('$T_{plan}$ (°C)')
ax1.set_ylabel('$T_{ped}$ (°C)')
ax1.set_title('LambdaP')

# Get the minimum and maximum values for x and y for LambdaP
x_min_lambda_p = x_values_dt.min()
x_max_lambda_p = x_values_dt.max()
y_min_lambda_p = y_values_dt.min()
y_max_lambda_p = y_values_dt.max()

# Plot the identity line for LambdaP
min_val_lambda_p = min(x_min_lambda_p, y_min_lambda_p)
max_val_lambda_p = max(x_max_lambda_p, y_max_lambda_p)
ax1.plot([min_val_lambda_p, max_val_lambda_p], [min_val_lambda_p, max_val_lambda_p], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for LambdaP
ax1.set_xlim(min_val_lambda_p - 1, max_val_lambda_p + 1)
ax1.set_ylim(min_val_lambda_p - 1, max_val_lambda_p + 1)

# Save the figure
fig1.savefig(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\LambdaP.png')

# Create a scatter plot for S_HGT_AGL
fig2, ax2 = plt.subplots()
ax2.scatter(x_values_dt, y_values_dt, s=normalized_s_hgt_agl, facecolors='none', edgecolors='red')
ax2.set_xlabel('$T_{plan}$ (°C)')
ax2.set_ylabel('$T_{ped}$ (°C)')
ax2.set_title('S_HGT_AGL')

# Get the minimum and maximum values for x and y for S_HGT_AGL
x_min_s_hgt_agl = x_values_dt.min()
x_max_s_hgt_agl = x_values_dt.max()
y_min_s_hgt_agl = y_values_dt.min()
y_max_s_hgt_agl = y_values_dt.max()

# Plot the identity line for S_HGT_AGL
min_val_s_hgt_agl = min(x_min_s_hgt_agl, y_min_s_hgt_agl)
max_val_s_hgt_agl = max(x_max_s_hgt_agl, y_max_s_hgt_agl)
ax2.plot([min_val_s_hgt_agl, max_val_s_hgt_agl], [min_val_s_hgt_agl, max_val_s_hgt_agl], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for S_HGT_AGL
ax2.set_xlim(min_val_s_hgt_agl - 1, max_val_s_hgt_agl + 1)
ax2.set_ylim(min_val_s_hgt_agl - 1, max_val_s_hgt_agl + 1)

# Save the figure
fig2.savefig(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\S_HGT_AGL.png')

# Create a scatter plot for AVG_SVF
fig3, ax3 = plt.subplots()
ax3.scatter(x_values_dt, y_values_dt, s=normalized_avg_svf, facecolors='none', edgecolors='green')
ax3.set_xlabel('$T_{plan}$ (°C)')
ax3.set_ylabel('$T_{ped}$ (°C)')
ax3.set_title('AVG_SVF')

# Get the minimum and maximum values for x and y for AVG_SVF
x_min_avg_svf = x_values_dt.min()
x_max_avg_svf = x_values_dt.max()
y_min_avg_svf = y_values_dt.min()
y_max_avg_svf = y_values_dt.max()

# Plot the identity line for AVG_SVF
min_val_avg_svf = min(x_min_avg_svf, y_min_avg_svf)
max_val_avg_svf = max(x_max_avg_svf, y_max_avg_svf)
ax3.plot([min_val_avg_svf, max_val_avg_svf], [min_val_avg_svf, max_val_avg_svf], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for AVG_SVF
ax3.set_xlim(min_val_avg_svf - 1, max_val_avg_svf + 1)
ax3.set_ylim(min_val_avg_svf - 1, max_val_avg_svf + 1)

# Save the figure
fig3.savefig(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\AVG_SVF.png')

# Create a scatter plot for W_area
fig4, ax4 = plt.subplots()
ax4.scatter(x_values_dt, y_values_dt, s=normalized_w_area, facecolors='none', edgecolors='orange')
ax4.set_xlabel('$T_{plan}$ (°C)')
ax4.set_ylabel('$T_{ped}$ (°C)')
ax4.set_title('W_area')

# Get the minimum and maximum values for x and y for W_area
x_min_w_area = x_values_dt.min()
x_max_w_area = x_values_dt.max()
y_min_w_area = y_values_dt.min()
y_max_w_area = y_values_dt.max()

# Plot the identity line for W_area
min_val_w_area = min(x_min_w_area, y_min_w_area)
max_val_w_area = max(x_max_w_area, y_max_w_area)
ax4.plot([min_val_w_area, max_val_w_area], [min_val_w_area, max_val_w_area], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values for W_area
ax4.set_xlim(min_val_w_area - 1, max_val_w_area + 1)
ax4.set_ylim(min_val_w_area - 1, max_val_w_area + 1)

# Save the figure
fig4.savefig(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\W_area.png')

print('DONE CREATING AVG SVF LAMBDA P S_HGT_AGL PLOTS')




import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Set the style of the plots
sns.set_style("whitegrid")

def create_combined_histogram(data, column1, column2, xlabel, title, file_name, legend_labels):
    plt.figure(figsize=(10, 8))
    n1, bins1, patches1 = plt.hist(data[column1], bins=20, color='blue', edgecolor='black', alpha=0.7, density=True, label=legend_labels[0])
    n2, bins2, patches2 = plt.hist(data[column2], bins=20, color='green', edgecolor='black', alpha=0.7, density=True, label=legend_labels[1])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()

    # Save the histogram as a JPEG file
    output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
    plt.savefig(os.path.join(output_dir, file_name), format='jpeg')

# Create daytime histogram
create_combined_histogram(df, 'TincO', 'AVG_Pl_DT', '$T_{ped}, T_{plan}$ (°C)', 'Combined Histogram of Daytime $T_{ped}$ and $T_{plan}$ (°C)', 'Daytime_combined_histogram.jpeg', ['$T_{ped}$', '$T_{plan}$'])

# Create nighttime histogram
create_combined_histogram(df, 'TincI', 'AVG_Pl_NT', '$T_{ped}, T_{plan}$', 'Combined Histogram of Nighttime $T_{ped}$ and $T_{plan}$ (°C)', 'Nighttime_combined_histogram.jpeg', ['$T_{ped}$', '$T_{plan}$'])

print('DONE CREATING HISTOGRAMS')


import os
import pandas as pd
import scipy.stats as stats

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

def calculate_and_save_statistics(data, column1, column2, output_file):
    # Set the precision of pandas to 2 decimal places
    pd.set_option('precision', 2)

    # Calculate the statistics
    statistics1 = data[column1].describe()
    mode1 = pd.Series({'mode': data[column1].mode()[0]})
    skewness1 = pd.Series({'skewness': data[column1].skew()})
    kurtosis1 = pd.Series({'kurtosis': data[column1].kurt()})
    iqr1 = pd.Series({'iqr': stats.iqr(data[column1])})
    range1 = pd.Series({'range': data[column1].max() - data[column1].min()})
    percentiles1 = data[column1].quantile([0.05, 0.95])

    statistics2 = data[column2].describe()
    mode2 = pd.Series({'mode': data[column2].mode()[0]})
    skewness2 = pd.Series({'skewness': data[column2].skew()})
    kurtosis2 = pd.Series({'kurtosis': data[column2].kurt()})
    iqr2 = pd.Series({'iqr': stats.iqr(data[column2])})
    range2 = pd.Series({'range': data[column2].max() - data[column2].min()})
    percentiles2 = data[column2].quantile([0.05, 0.95])

    # Combine the statistics into a DataFrame
    stats_df1 = pd.concat([statistics1, mode1, skewness1, kurtosis1, iqr1, range1, percentiles1])
    stats_df2 = pd.concat([statistics2, mode2, skewness2, kurtosis2, iqr2, range2, percentiles2])

    with open(output_file, 'w') as file:
        file.write(f'Statistics for {column1}:\n')
        file.write(stats_df1.to_string())
        file.write('\n\n')
        file.write(f'Statistics for {column2}:\n')
        file.write(stats_df2.to_string())

# Set the output directory
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# Calculate and save statistics for daytime
calculate_and_save_statistics(df, 'TincO', 'AVG_Pl_DT', os.path.join(output_dir, 'Daytime_statistics.txt'))

# Calculate and save statistics for nighttime
calculate_and_save_statistics(df, 'TincI', 'AVG_Pl_NT', os.path.join(output_dir, 'Nighttime_statistics.txt'))

print('DONE CREATING HISTOGRAM STATISTICS')


print('DONE WITH EVERYTHING')
