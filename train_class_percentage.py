import xarray as xr
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def generate_bar_graph(LABELS, pixels_per_class, n_pixels, x_label, bar_width):
  labels = [LABELS[i] for i in range(len(pixels_per_class))]
  percent_class = [(class_n/n_pixels)*100 for class_n in pixels_per_class]

  plt.figure(figsize=(6, 5))
  bars = plt.bar(labels, percent_class,width=bar_width)

  for bar, class_n in zip(bars, percent_class):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{class_n:.2f}%", ha='center', va='bottom', fontsize=10)
    
  plt.xlabel(x_label)
  plt.ylabel("Pixel Count Percentage [%]")
  plt.xticks(list(LABELS.values()), rotation=45, ha='right')
  plt.tight_layout()
  plt.show()  

def main():

    with open('/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/datalists/train_dataset_cross_validation.json') as file: 
        ai4arctic_files = json.loads(file.read())

    # Convert the original scene names to the preprocessed names.
    ai4arctic_files = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in ai4arctic_files]

    n_classes = [12,7,8]

    LABELS_SIC

    for ai4arctic_file in ai4arctic_files:
        scene = xr.open_dataset(os.path.join('/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/dataset/', ai4arctic_file), engine='h5netcdf')

        SIC = scene.variables['SIC'].values
        SOD = scene.variables['SOD'].values
        FLOE = scene.variables['FLOE'].values


if __name__ == '__main__':
    main()