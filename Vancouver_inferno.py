#ALBEDO


import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import rasterio
from matplotlib_scalebar.scalebar import ScaleBar

# Set the directory
root_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Find the .tif file in the directory
tif_files = glob.glob(os.path.join(root_directory, "*.tif"))
if len(tif_files) != 1:
    raise ValueError("Expected exactly one .tif file in the directory, but found {}".format(len(tif_files)))
rasterfile_path = tif_files[0]

# Open the raster file
raster = rasterio.open(rasterfile_path)

# Get the valid pixel values of the raster
raster_data = raster.read(1, masked=True)
valid_pixel_values = raster_data.compressed()

# Compute the minimum and maximum pixel values from the valid pixels and round to the nearest whole number
raster_min = round(valid_pixel_values.min())
raster_max = round(valid_pixel_values.max())

# Normalize the pixel values between 0 and 1
norm = Normalize(vmin=raster_min, vmax=raster_max)
raster_data_norm = norm(raster_data)

# Create the figure with one axis
fig, ax = plt.subplots(figsize=(7, 8))

# Plot the normalized raster file on the axis
show(raster_data_norm, transform=raster.transform, ax=ax, cmap="inferno")

# Create a ScalarMappable object with the colormap and normalization
cmap = get_cmap("inferno")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create a colorbar using the ScalarMappable on the axis
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)

raster_label = "Albedo"
cbar.set_label(raster_label)

# Remove x-axis and y-axis from the axis
ax.axis('off')

# Add scale bar and adjust its position
scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower center', pad=-5,
                    color='black', frameon=True)
ax.add_artist(scalebar)

# Add north arrow to the axis and adjust its position
arrow_props = dict(arrowstyle="->", mutation_scale=100, linewidth=1.5, color='black')
ax.annotate('N', xy=(0.05, 1.15), xycoords='axes fraction', xytext=(0.05, 1.1),
            textcoords='axes fraction', arrowprops=arrow_props, ha='center', va='center', fontsize=8)

# Create a directory for jpeg files in the current directory if it doesn't exist
jpeg_directory = os.path.join(root_directory, "jpeg_files")
os.makedirs(jpeg_directory, exist_ok=True)

# Save the figure as a JPEG file in the "jpeg_files" folder
output_filename = os.path.splitext(os.path.basename(rasterfile_path))[0] + ".jpeg"
output_path = os.path.join(jpeg_directory, output_filename)
plt.savefig(output_path, format='jpeg')

# Close the figure
plt.close(fig)

print('JPEG FILE CREATED')




#THIS SCRIPT IS USED TO MAKE POLYGON DISCRETE COLOMAPS


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np

# Set the root directory
root_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Path to the shapefile
shapefile_path = os.path.join(root_directory, "LCZ7_Building.shp")

# Read the shapefile
shapes = gpd.read_file(shapefile_path)

# Check if 'HGT_AGL' exists in the shapefile
if 'HGT_AGL' in shapes.columns:
    # Number of colors to use
    n_colors = 10

    # Custom colors to use
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black']


    # Create a color map with custom discrete colors
    cmap = mcolors.ListedColormap(colors)

    # Create normalization
    norm = mcolors.Normalize(vmin=shapes['HGT_AGL'].min(), vmax=shapes['HGT_AGL'].max())

    # Divide 'HGT_AGL' into intervals
    shapes['HGT_AGL_discrete'] = pd.cut(shapes['HGT_AGL'], bins=n_colors, labels=False)

    # Create the figure with one axis
    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot the shapefile on the axis using 'HGT_AGL_discrete' as the color mapping field
    shapes.plot(column='HGT_AGL_discrete', ax=ax, cmap=cmap, legend=False)

    # Create a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Building Height (m)')

    # Remove x-axis and y-axis from the axis
    ax.axis('off')

    # Add scale bar and adjust its position
    scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower center', pad=-5,
                        color='black', frameon=True)
    ax.add_artist(scalebar)

    # Add north arrow to the axis and adjust its position
    arrow_props = dict(arrowstyle="->", mutation_scale=100, linewidth=1.5, color='black')
    ax.annotate('N', xy=(0.05, 1.15), xycoords='axes fraction', xytext=(0.05, 1.1),
                textcoords='axes fraction', arrowprops=arrow_props, ha='center', va='center', fontsize=8)

    # Save the figure as a JPEG file
    output_filename = "LCZ5_buildings.jpeg"
    output_path = os.path.join(root_directory, output_filename)
    plt.savefig(output_path, format='jpeg')

    # Close the figure
    plt.close(fig)

    print('DONE CREATING JPEG FILE')

else:
    print("'HGT_AGL' field does not exist in the shapefile.")





#THIS SCRIPT IS USED TO MAKE POLYGON CONTINOUS COLOMAPS


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as mcolors

# Set the root directory
root_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids"

# Path to the shapefile
shapefile_path = os.path.join(root_directory, "LCZ6_buildings.shp")

# Read the shapefile
shapes = gpd.read_file(shapefile_path)

# Check if 'HGT_AGL' exists in the shapefile
if 'HGT_AGL' in shapes.columns:

    # Create the figure with one axis
    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot the shapefile on the axis using 'HGT_AGL' as the color mapping field
    shapes.plot(column='HGT_AGL', ax=ax, cmap='cubehelix', legend=True)

    # Remove x-axis and y-axis from the axis
    ax.axis('off')

    # Add scale bar and adjust its position
    scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower center', pad=-5,
                        color='black', frameon=True)
    ax.add_artist(scalebar)

    # Add north arrow to the axis and adjust its position
    arrow_props = dict(arrowstyle="->", mutation_scale=100, linewidth=1.5, color='black')
    ax.annotate('N', xy=(0.05, 1.15), xycoords='axes fraction', xytext=(0.05, 1.1),
                textcoords='axes fraction', arrowprops=arrow_props, ha='center', va='center', fontsize=8)

    # Save the figure as a JPEG file
    output_filename = "LCZ5_buildings.jpeg"
    output_path = os.path.join(root_directory, output_filename)
    plt.savefig(output_path, format='jpeg')

    # Close the figure
    plt.close(fig)

    print('DONE CREATING JPEG FILE')

else:
    print("'HGT_AGL' field does not exist in the shapefile.")







import os
import numpy as np
from osgeo import gdal

# Path to the directory containing the tif file
directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# Name of the tif file
filename = 'F2_Tped_51_10m_final.tif'

# Full path to the tif file
tif_path = os.path.join(directory, filename)

# Open the tif file
dataset = gdal.Open(tif_path)

# Read the pixel values into a numpy array
band = dataset.GetRasterBand(1)
array = band.ReadAsArray()

# Remove pixels with 0 values
array[array == 0] = np.nan

# Deduct 273.15 from each pixel value
array -= 273.15

# Create a new tif file to save the modified array
output_path = os.path.join(directory, 'modified.tif')
driver = gdal.GetDriverByName('GTiff')
new_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
new_dataset.SetProjection(dataset.GetProjection())
new_dataset.SetGeoTransform(dataset.GetGeoTransform())

# Write the modified array to the new tif file
new_band = new_dataset.GetRasterBand(1)
new_band.WriteArray(array)
new_band.SetNoDataValue(np.nan)

# Clean up
dataset = None
new_dataset = None

# Delete the original file
os.remove(tif_path)

# Rename the new file to have the same name as the original file
os.rename(output_path, tif_path)

print("DONE DELETING 0 PIXELS AND CHANGING TO DEGREE CELSIUS")


import os
import glob
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar

# Set the root directory
root_directory = r"C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files"

# Walk through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    # Check each directory
    for filename in [f for f in filenames if f.endswith(".tif")]:
        # Full path to the current tif file
        rasterfile_path = os.path.join(dirpath, filename)
        # Full path to the corresponding shp file
        shapefile_path = os.path.join(dirpath, "Buildings.shp")

        # Check if corresponding shp file exists
        if os.path.isfile(shapefile_path):
            # Read the shapefile
            shapes = gpd.read_file(shapefile_path)

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
            ax.annotate('N', xy=(0.05, 1.15), xycoords='axes fraction', xytext=(0.05, 1.1),
                        textcoords='axes fraction', arrowprops=arrow_props, ha='center', va='center', fontsize=8) #change the y values of the (ie x, y) to move the north arrow up higher

            # Create a directory for jpeg files in the current directory if it doesn't exist
            jpeg_directory = os.path.join(dirpath, "jpeg_files")
            os.makedirs(jpeg_directory, exist_ok=True)

            # Save the figure as a JPEG file in the "jpeg_files" folder
            output_filename = os.path.splitext(filename)[0] + ".jpeg"
            output_path = os.path.join(jpeg_directory, output_filename)
            plt.savefig(output_path, format='jpeg')

            # Close the figure
            plt.close(fig)

print('DONE CREATING JPEG FILES')


#this script deducts 273.15 from TincO and Tplan columns
import pandas as pd
import os

directory = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
filename = 'Vancouver Flights.xlsx'

# Get the file path
file_path = os.path.join(directory, filename)

# Load the Excel file
excel_file = pd.ExcelFile(file_path)

# Iterate over each sheet in the Excel file
for sheet_name in excel_file.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Check if the 'AVG_Pl_DT' column exists in the DataFrame
    if 'AVG_Pl_DT' in df.columns:
        # Deduct 273.15 from each cell in the 'AVG_Pl_DT' column
        df['AVG_Pl_DT'] -= 273.15
    
    # Check if the 'TincO' column exists in the DataFrame
    if 'TincO' in df.columns:
        # Deduct 273.15 from each cell in the 'TincO' column
        df['TincO'] -= 273.15
    
    # Save the modified DataFrame back to the sheet
    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


#THIS SCRIPT CREATES ALL THE SCATTER, REGRESSION, HISTOGRAMS






import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'

# Get all sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# Loop over all sheets
for sheet in sheet_names:
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Loop over all sheets
for sheet in sheet_names:
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Check if 'Zenith angles' column exists and define x accordingly
    if 'Zenith angles' in df.columns and 'AVG_Pl_DT' in df.columns:
        x = df[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
    elif 'AVG_Pl_DT' in df.columns:
        x = df[['LambdaP', 'W_area', 'AVG_Pl_DT']]
    else:
        continue  # Skip this sheet if 'AVG_Pl_DT' doesn't exist

    


    # Create Daytime scatter plots
    x_values_dt = df['AVG_Pl_DT']
    y_values_dt = df['TincO']
    plt.scatter(x_values_dt, y_values_dt, facecolors='none', edgecolors='blue')
    plt.xlabel('$T_{plan}$ (°C)', fontsize=18)
    plt.ylabel('$T_{ped}$ (°C)', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.title('Daytime', fontsize=20)
    x_min_dt = x_values_dt.min()
    x_max_dt = x_values_dt.max()
    y_min_dt = y_values_dt.min()
    y_max_dt = y_values_dt.max()
    min_val_dt = min(x_min_dt, y_min_dt)
    max_val_dt = max(x_max_dt, y_max_dt)
    plt.plot([min_val_dt, max_val_dt], [min_val_dt, max_val_dt], color='red', linestyle='--')
    plt.xlim(min_val_dt - 1, max_val_dt + 1)
    plt.ylim(min_val_dt - 1, max_val_dt + 1)
    plt.savefig(os.path.join(output_dir, f'{sheet}_Daytime_scatter.jpeg'), format='jpeg')
    plt.clf()  # Clear the figure for the next plot

    # Create daytime regression
   
    y = df['TincO']
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    predicted_values = results.predict()
    plt.figure(figsize=(12, 8))
    plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
    
    plt.xlabel('Calculated $T_{ped}$ (°C)', fontsize=18)
    plt.ylabel('Predicted $T_{ped}$ (°C)', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)

    x_min = y.min()
    x_max = y.max()
    y_min = predicted_values.min()
    y_max = predicted_values.max()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.xlim(min_val - 1, max_val + 1)
    plt.ylim(min_val - 1, max_val + 1)
    plt.savefig(os.path.join(output_dir, f'{sheet}_Daytime_Regression_scatter.jpeg'), format='jpeg')
    plt.clf()  # Clear the figure for the next plot

    with open(os.path.join(output_dir, f'{sheet}_Daytime_Regression_summary.txt'), 'w') as f:
        f.write(str(results.summary()))
    
    df_results = df.copy()
    df_results['Predicted_TincO'] = predicted_values
    df_results_str = df_results.to_string()

    with open(os.path.join(output_dir, f'{sheet}_Daytime_Regression_results.txt'), 'w') as f:
        f.write(df_results_str)

    # Create histograms
    sns.set_style("whitegrid")

    def create_combined_histogram(data, column1, column2, xlabel, title, file_name, legend_labels):
        plt.figure(figsize=(10, 8))
        n1, bins1, patches1 = plt.hist(data[column1], bins=20, color='blue', edgecolor='black', alpha=0.7, density=True, label=legend_labels[0])
        n2, bins2, patches2 = plt.hist(data[column2], bins=20, color='green', edgecolor='black', alpha=0.7, density=True, label=legend_labels[1])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Percentage', fontsize=18)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.savefig(os.path.join(output_dir, file_name), format='jpeg')
        plt.clf()  # Clear the figure for the next plot

    # Call the function for each sheet
    create_combined_histogram(df, 'TincO', 'AVG_Pl_DT', '$T_{ped}, T_{plan}$ (°C)', 'Combined Histogram of Daytime $T_{ped}$ and $T_{plan}$ (°C)', f'{sheet}_Daytime_combined_histogram.jpeg', ['$T_{ped}$', '$T_{plan}$'])

    print(f'DONE PROCESSING {sheet}')

print('DONE PROCESSING ALL SHEETS')





import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'

# Get all sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# Loop over all sheets
for sheet in sheet_names:
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Check if 'Zenith angles' column exists and define x accordingly
    if 'Zenith angles' in df.columns and 'AVG_Pl_NT' in df.columns:
        x = df[['LambdaP', 'W_area', 'AVG_Pl_NT', 'Zenith angles']]
    elif 'AVG_Pl_NT' in df.columns:
        x = df[['LambdaP', 'W_area', 'AVG_Pl_NT']]
    else:
        continue  # Skip this sheet if 'AVG_Pl_NT' doesn't exist

    # Create nighttime regression
    y_nt = df['TincI']
    x = sm.add_constant(x)
    model_nt = sm.OLS(y_nt, x)
    results_nt = model_nt.fit()
    predicted_values_nt = results_nt.predict()

    # Create scatter plot
    plt.scatter(y_nt, predicted_values_nt, facecolors='none', edgecolors='blue')
    plt.xlabel('Calculated $T_{ped}$ (°C)')
    plt.ylabel('Predicted $T_{ped}$ (°C)')
    plt.title('Nighttime Regression Model: Calculated vs. Predicted $T_{ped}$ (°C)')
    plt.plot([y_nt.min(), y_nt.max()], [y_nt.min(), y_nt.max()], color='red', linestyle='--')
    plt.savefig(os.path.join(output_dir, f'{sheet}_Nighttime_Regression_scatter.jpeg'), format='jpeg')
    plt.clf()

    # Write the regression summary to a text file
    with open(os.path.join(output_dir, f'{sheet}_Nighttime_Regression_summary.txt'), 'w') as f:
        f.write(str(results_nt.summary()))

    # Add the predicted values to the DataFrame and write it to a text file
    df_results_nt = df.copy()
    df_results_nt['Predicted_TincI'] = predicted_values_nt
    df_results_str_nt = df_results_nt.to_string()
    with open(os.path.join(output_dir, f'{sheet}_Nighttime_Regression_results.txt'), 'w') as f:
        f.write(df_results_str_nt)

    print(f'DONE PROCESSING {sheet}')

print('DONE PROCESSING ALL NIGHTTIME REGRESSION')



#THIS SCRIPT PERFORMS THE MODEL PREDICTION.
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import numpy as np

def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df_day = pd.read_excel(xls, 'Earth_future_Day')
        flight_sheets = ['Flight1_101', 'Flight2_101', 'Flight3_101']
        flight_dfs = [pd.read_excel(xls, sheet) for sheet in flight_sheets]
        return df_day, flight_dfs
    except Exception as e:
        print("Error in loading data: ", e)
        return None, None

def train_model(df_day):
    X_train = df_day[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
    y_train = df_day['TincO']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(flight_dfs, model):
    predicted_dfs = []
    for i, df in enumerate(flight_dfs, start=1):
        X_predict = df[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
        predicted_TincO = model.predict(X_predict)
        df['Predicted_TincO'] = predicted_TincO
        df['Flight number'] = 'Flight {}'.format(i)
        predicted_dfs.append(df)
    return predicted_dfs

def calculate_metrics(predicted_dfs):
    metrics = []
    for df in predicted_dfs:
        rmse = sqrt(mean_squared_error(df['TincO'], df['Predicted_TincO']))
        r2 = r2_score(df['TincO'], df['Predicted_TincO'])
        n = df.shape[0]
        p = df.shape[1] - 1
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        mape = np.mean(np.abs((df['TincO'] - df['Predicted_TincO']) / df['TincO'])) * 100
        mae = mean_absolute_error(df['TincO'], df['Predicted_TincO'])
        metrics.append((rmse, r2, adjusted_r2, mape, mae))
    return metrics

def save_metrics(folder_path, metric, flight_number):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'model_metrics.txt'), 'w') as f:
        f.write('Metrics for Flight {}:\n'.format(flight_number))
        f.write('RMSE: {}\n'.format(metric[0]))
        f.write('R^2: {}\n'.format(metric[1]))
        f.write('Adjusted R^2: {}\n'.format(metric[2]))
        f.write('MAPE: {}\n'.format(metric[3]))
        f.write('MAE: {}\n\n'.format(metric[4]))

def save_df(dfs, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder_path, 'Vancouver_Flights_modified.xlsx')) as writer:  
        for i, df in enumerate(dfs, start=1):
            df.to_excel(writer, sheet_name='Flight{}_101_modified'.format(i))

def plot_metrics(output_folder, metrics, flights=['Flight 1', 'Flight 2', 'Flight 3']):
    fig, ax = plt.subplots()
    ax.axis('off')

    metric_names = ['Flight', 'RMSE', 'R^2', 'Adjusted R^2', 'MAPE', 'MAE']
    table_data = [metric_names]

    for flight, flight_metrics in zip(flights, metrics):
        flight_data = [flight] + ['{:.2f}'.format(value) for value in flight_metrics]
        table_data.append(flight_data)

    ax.table(cellText=table_data, loc='center', cellLoc='center', rowLoc='center')

    plt.savefig(os.path.join(output_folder, 'metrics_summary.png'))

def create_scatter_plot(predicted_dfs, folder_path):
    for i, df in enumerate(predicted_dfs, start=1):
        plt.figure(figsize=(10, 6))
        plt.scatter(df['TincO'], df['Predicted_TincO'])
        min_value = min(df['TincO'].min(), df['Predicted_TincO'].min())
        max_value = max(df['TincO'].max(), df['Predicted_TincO'].max())
        plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
        plt.title(f'Scatter plot for Calculated and predicted Tped (°C) for Flight {i}')
        plt.xlabel('Calculated $T_{ped}$ (°C)')
        plt.ylabel('Predicted $T_{ped}$ (°C)')
        plt.savefig(os.path.join(folder_path, f'Scatter_Plot_Flight{i}.png'))
        plt.close()

def main():
    input_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'
    output_folder = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\output'
    df_day, flight_dfs = load_data(input_file)
    model = train_model(df_day)
    predicted_dfs = predict(flight_dfs, model)
    metrics = calculate_metrics(predicted_dfs)
    for i, _ in enumerate(flight_dfs, start=1):
        save_metrics(os.path.join(output_folder, 'Flight{}_101'.format(i)), metrics[i-1], i)
        save_df([predicted_dfs[i-1]], os.path.join(output_folder, 'Flight{}_101'.format(i)))
    create_scatter_plot(predicted_dfs, output_folder)
    plot_metrics(output_folder, metrics)

if __name__ == "__main__":
    main()

print('DONE WITH MACHINE LEARNING')





#AN OLD ONE FOR MODEL PREDICTION
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import numpy as np

def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df_day = pd.read_excel(xls, 'Earth_future_Day')
        flight_sheets = ['Flight1_101', 'Flight2_101', 'Flight3_101']
        flight_dfs = [pd.read_excel(xls, sheet) for sheet in flight_sheets]
        return df_day, flight_dfs
    except Exception as e:
        print("Error in loading data: ", e)
        return None, None

def train_model(df_day):
    X_train = df_day[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
    y_train = df_day['TincO']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(flight_dfs, model):
    predicted_dfs = []
    for i, df in enumerate(flight_dfs, start=1):
        X_predict = df[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
        predicted_TincO = model.predict(X_predict)
        df['Predicted_TincO'] = predicted_TincO
        df['Flight number'] = 'Flight {}'.format(i)
        predicted_dfs.append(df)
    return predicted_dfs

def calculate_metrics(predicted_dfs):
    metrics = []
    for df in predicted_dfs:
        rmse = sqrt(mean_squared_error(df['TincO'], df['Predicted_TincO']))
        r2 = r2_score(df['TincO'], df['Predicted_TincO'])
        n = df.shape[0]
        p = df.shape[1] - 1
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        mape = np.mean(np.abs((df['TincO'] - df['Predicted_TincO']) / df['TincO'])) * 100
        mae = mean_absolute_error(df['TincO'], df['Predicted_TincO'])
        metrics.append((rmse, r2, adjusted_r2, mape, mae))
    return metrics

def save_metrics(folder_path, metric, flight_number):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'model_metrics.txt'), 'w') as f:
        f.write('Metrics for Flight {}:\n'.format(flight_number))
        f.write('RMSE: {}\n'.format(metric[0]))
        f.write('R^2: {}\n'.format(metric[1]))
        f.write('Adjusted R^2: {}\n'.format(metric[2]))
        f.write('MAPE: {}\n'.format(metric[3]))
        f.write('MAE: {}\n\n'.format(metric[4]))

def save_df(dfs, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder_path, 'Vancouver_Flights_modified.xlsx')) as writer:  
        for i, df in enumerate(dfs, start=1):
            df.to_excel(writer, sheet_name='Flight{}_101_modified'.format(i))

def plot_metrics(output_folder, metrics, flights=['Flight 1', 'Flight 2', 'Flight 3']):
    fig, ax = plt.subplots()
    
    ax.axis('off')

    metric_names = ['Flight', 'RMSE', 'R^2', 'Adjusted R^2', 'MAPE', 'MAE']
    table_data = [metric_names]

    for flight, flight_metrics in zip(flights, metrics):
        flight_data = [flight] + ['{:.2f}'.format(value) for value in flight_metrics]
        table_data.append(flight_data)

    ax.table(cellText=table_data, loc='center', cellLoc = 'center', rowLoc = 'center')

    plt.savefig(os.path.join(output_folder, 'metrics_summary.png'))

def main():
    input_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'
    output_folder = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\output'
    df_day, flight_dfs = load_data(input_file)
    model = train_model(df_day)
    predicted_dfs = predict(flight_dfs, model)
    metrics = calculate_metrics(predicted_dfs)
    for i, _ in enumerate(flight_dfs, start=1):
        save_metrics(os.path.join(output_folder, 'Flight{}_101'.format(i)), metrics[i-1], i)
        save_df([predicted_dfs[i-1]], os.path.join(output_folder, 'Flight{}_101'.format(i)))
    
    plot_metrics(output_folder, metrics)

if __name__ == "__main__":
    main()
print('DONE WITH MACHINE LEARNING')






#THIS IS USED TO DO THE LINE GRAPHS



#THIS IS USED TO DO THE KINE GRAPHS

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Increase the default size of the figures and font size
plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams["font.size"] = 14

# define the file path
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'

# define the input excel file
input_file = file_path + r'\Vancouver Flights.xlsx'

# Function to compute intervals for 30 minutes data step size
def compute_intervals_30min(df):
    time_values = df['time'].unique()
    hour_intervals = np.arange(start=0, stop=len(time_values), step=4)  # every 2 hours for 30 minutes interval
    return time_values, hour_intervals

# Function to compute intervals for 1 hour data step size
def compute_intervals_1hour(df):
    time_values = df['time'].unique()
    hour_intervals = np.arange(start=0, stop=len(time_values), step=2)  # every 2 hours for 1 hour interval
    return time_values, hour_intervals

# read the 'sensitivity test' sheet into a pandas DataFrame
df = pd.read_excel(input_file, sheet_name='LCZ8_roughness')

# Convert time into format HH:MM
df['time'] = df['time'].apply(lambda x: str(int(x)) + ":" + str(int((x % 1)*60)).zfill(2) if x >= 1 else "00:" + str(int((x % 1)*60)).zfill(2))

# Set the style of the plot
plt.style.use('seaborn-darkgrid')

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a line plot for each column against time
ax.plot(df['time'], df['smooth (0.07)'], label='smooth (0.07)')
ax.plot(df['time'], df['rough (0.15)'], label='rough (0.15)')
ax.plot(df['time'], df['very rough (0.5)'], label='very rough (0.5)')

# Add a legend to the plot
ax.legend()

# Set xticks rotation
time_values = df['time'].unique()
hour_intervals = np.arange(start=0, stop=len(time_values), step=4)  # modify the step size as needed
plt.xticks(hour_intervals, time_values[hour_intervals], rotation=90)

# Set labels
ax.set_xlabel('Time (Local Solar Time)')
ax.set_ylabel('Roof surface temperatures (°C)')

# Save the plot
plt.savefig(file_path + r'\roughness.jpg', dpi=300)

# Repeat for wind speed data
df = pd.read_excel(input_file, sheet_name='LCZ8_windSpeed')

# Convert time into format HH:MM
df['time'] = df['time'].apply(lambda x: str(int(x)) + ":" + str(int((x % 1)*60)).zfill(2) if x >= 1 else "00:" + str(int((x % 1)*60)).zfill(2))

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a line plot for each column against time
ax.plot(df['time'], df['Ws'], label='Ws')
ax.plot(df['time'], df['Ws-2'], label='Ws-2')
ax.plot(df['time'], df['Ws+2'], label='Ws+2')

# Add a legend to the plot
ax.legend()

# Set xticks rotation
time_values = df['time'].unique()
hour_intervals = np.arange(start=0, stop=len(time_values), step=4)  # modify the step size as needed
plt.xticks(hour_intervals, time_values[hour_intervals], rotation=90)

# Set labels
ax.set_xlabel('Time (Local Solar Time)')
ax.set_ylabel('Roof surface temperatures (°C)')

# Save the plot
plt.savefig(file_path + r'\windSpeed.jpg', dpi=300)

# For 'air temperature'
df = pd.read_excel(input_file, sheet_name='air temperature')
df['time'] = df['time'].apply(lambda x: str(int(x)) + ":" + str(int((x % 1)*60)).zfill(2) if x >= 1 else "00:" + str(int((x % 1)*60)).zfill(2))
time_values, hour_intervals = compute_intervals_1hour(df)

fig, ax = plt.subplots()
for column in df.columns:
    if column != 'time':
        ax.plot(df['time'], df[column], label=column)

ax.legend()
plt.xticks(hour_intervals, time_values[hour_intervals], rotation=90)
ax.set_xlabel('Time (Local Solar Time)')
ax.set_ylabel('air temperature (°C)')
plt.savefig(file_path + r'\air temperature.jpg', dpi=300)

# For 'humidity'
df = pd.read_excel(input_file, sheet_name='humidity')
df['time'] = df['time'].apply(lambda x: str(int(x)) + ":" + str(int((x % 1)*60)).zfill(2) if x >= 1 else "00:" + str(int((x % 1)*60)).zfill(2))
time_values, hour_intervals = compute_intervals_1hour(df)

fig, ax = plt.subplots()
for column in df.columns:
    if column != 'time':
        ax.plot(df['time'], df[column], label=column)

ax.legend()
plt.xticks(hour_intervals, time_values[hour_intervals], rotation=90)
ax.set_xlabel('Time (Local Solar Time)')
ax.set_ylabel('Humidity (%)')
plt.savefig(file_path + r'\humidity.jpg', dpi=300)

# For 'solar'
df = pd.read_excel(input_file, sheet_name='solar')
df['time'] = df['time'].apply(lambda x: str(int(x)) + ":" + str(int((x % 1)*60)).zfill(2) if x >= 1 else "00:" + str(int((x % 1)*60)).zfill(2))
time_values, hour_intervals = compute_intervals_1hour(df)

fig, ax = plt.subplots()
for column in df.columns:
    if column != 'time':
        ax.plot(df['time'], df[column], label=column)

ax.legend()
plt.xticks(hour_intervals, time_values[hour_intervals], rotation=90)
ax.set_xlabel('Time (Local Solar Time)')
ax.set_ylabel('Solar Radiation (Wm-2)')
plt.savefig(file_path + r'\solar.jpg', dpi=300)
print('DONE CREATING SENSITIVITY TEST LINE GRAPHS')
