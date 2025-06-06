# -- Built-in modules -- #
import os
import datetime
from dateutil import relativedelta
import re
from tqdm import tqdm

# -- Third-party modules -- #
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# -- Proprietary modules -- #
from functions import rand_bbox

class TrainValDataset(Dataset):
    def __init__(self, options, ai4arctic_files, viirs_files, do_transform = False):
        # instance attributes
        self.options = options # from the config file
        self.ai4arctic_files = ai4arctic_files
        self.viirs_files = viirs_files
        self.do_transform = do_transform

        if (self.options['down_sample_scale'] == 1):
            self.downsample = False
        else:
            self.downsample = True

        ### DOWNSCALE DATA IF TRUE ###
        if self.downsample:
            self.scenes = []
            self.scenes_viirs = []
            self.amsr_scenes = []
            self.aux = []

            # CHANGE: now iterates through both ai4arctic and viirs
            for ai4arctic_file, viirs_file in tqdm(zip(self.ai4arctic_files, viirs_files)):
                scene = xr.open_dataset(os.path.join(
                    self.options['path_to_train_data'], ai4arctic_file), engine='h5netcdf')

                temp_scene = scene[self.options['full_variables']].to_array()
                temp_scene = torch.from_numpy(np.expand_dims(temp_scene, 0))
                temp_scene = torch.nn.functional.interpolate(
                    temp_scene,
                    size=(temp_scene.size(2) // self.options['down_sample_scale'],
                        temp_scene.size(3) // self.options['down_sample_scale']),
                    mode=self.options['loader_downsampling'])

                scene_size_before_padding = temp_scene.shape

                scene_viirs = xr.open_dataset(os.path.join(
                    self.options['path_to_train_data'], viirs_file), engine='h5netcdf')

                viirs_ist = self.mask_viirs(scene_viirs)

                temp_viirs = torch.from_numpy(viirs_ist).unsqueeze(0).unsqueeze(0)
                temp_viirs = torch.nn.functional.interpolate(
                    temp_viirs,
                    size=(temp_viirs.size(2) // self.options['down_sample_scale'],
                        temp_viirs.size(3) // self.options['down_sample_scale']),
                    mode=self.options['loader_downsampling'])

                ## NEW: condenses original conditional statement ##
                height_pad = max(0, self.options['patch_size'] - temp_scene.size(2) + 1)
                width_pad = max(0, self.options['patch_size'] - temp_scene.size(3) + 1)
                ## NEW: condenses original conditional statement ###

                if height_pad > 0 or width_pad > 0:
                    # full variables is ordered charts, sar_variables
                    temp_scene_y = torch.nn.functional.pad(
                        temp_scene[:, :len(self.options['charts'])],
                        (0, width_pad, 0, height_pad),
                        mode='constant',
                        value=255)
                    temp_scene_x = torch.nn.functional.pad(
                        temp_scene[:, len(self.options['charts']):],
                        (0, width_pad, 0, height_pad),
                        mode='constant',
                        value=0)
                    temp_scene = torch.cat((temp_scene_y, temp_scene_x), dim=1)

                    temp_viirs = torch.nn.functional.pad(
                        temp_viirs, (0, width_pad, 0, height_pad), mode='constant', value=0)

                if len(self.options['amsrenv_variables']) > 0:
                    temp_amsr = np.array(scene[self.options['amsrenv_variables']].to_array())
                    self.amsr_scenes.append(temp_amsr)

                if len(self.options['auxiliary_variables']) > 0:
                    temp_aux = []

                    if 'aux_time' in self.options['auxiliary_variables']:
                        scene_id = scene.attrs['scene_id']
                        norm_time = get_norm_month(scene_id)
                        time_array = torch.from_numpy(
                            np.full((scene_size_before_padding[2], scene_size_before_padding[3]), norm_time)).unsqueeze(0).unsqueeze(0)

                        if height_pad > 0 or width_pad > 0:
                            time_array = torch.nn.functional.pad(
                                time_array, (0, width_pad, 0, height_pad), mode='constant', value=0)

                        temp_aux.append(time_array)

                    if 'aux_lat' in self.options['auxiliary_variables']:
                        lat_array = scene['sar_grid2d_latitude'].values
                        lat_array = (lat_array - self.options['latitude']['mean']) / self.options['latitude']['std']

                        inter_lat_array = torch.nn.functional.interpolate(
                            input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                            size=(scene_size_before_padding[2], scene_size_before_padding[3]),
                            mode=self.options['loader_upsampling'])

                        if height_pad > 0 or width_pad > 0:
                            inter_lat_array = torch.nn.functional.pad(
                                inter_lat_array, (0, width_pad, 0, height_pad), mode='constant', value=0)

                        temp_aux.append(inter_lat_array)

                    if 'aux_long' in self.options['auxiliary_variables']:
                        long_array = scene['sar_grid2d_longitude'].values
                        long_array = (long_array - self.options['longitude']['mean']) / self.options['longitude']['std']

                        inter_long_array = torch.nn.functional.interpolate(
                            input=torch.from_numpy(long_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                            size=(scene_size_before_padding[2], scene_size_before_padding[3]),
                            mode=self.options['loader_upsampling'])

                        if height_pad > 0 or width_pad > 0:
                            inter_long_array = torch.nn.functional.pad(
                                inter_long_array, (0, width_pad, 0, height_pad), mode='constant', value=0)

                        temp_aux.append(inter_long_array)

                    self.aux.append(torch.cat(temp_aux, 1))

                temp_scene = torch.squeeze(temp_scene)
                self.scenes.append(temp_scene)

                temp_viirs = torch.squeeze(temp_viirs)
                reshaped_viirs = np.array(temp_viirs)
                self.scenes_viirs.append(reshaped_viirs)

            self.patch_c = len(
                self.options['train_variables']) + len(self.options['charts'])

    def __len__(self):
        """
        Provide number of iterations per epoch. Function required by Pytorch
        dataset.
        Returns
        -------
        Number of iterations per epoch.
        """
        return self.options['epoch_len']

    def mask_viirs(self, scene_viirs):

        data = scene_viirs.variables['IST'].values
        flags = [0,1,2,3,4,5,6,33]
        mask_indices = np.full_like(data, False)
        mask_indices = np.isin(data,flags) | np.isnan(data)
        data[mask_indices] = 0

        data_no_zeros = data[mask_indices == False]

        # Calculate mean and standard deviation
        mean = np.mean(data_no_zeros)
        std_dev = np.std(data_no_zeros)

        norm_data = (data - mean)/std_dev
        norm_data[mask_indices] = 0

        return norm_data

    def random_crop(self, scene, viirs_ist): #viirs_ist because it's been masked
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a scene from ASID3 ready-to-train challenge
            dataset.

        Returns
        -------
        x_patch :
            torch array with shape (len(train_variables),
            patch_height, patch_width). None if empty patch.
        y_patch :
            torch array with shape (len(charts),
            patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros((len(self.options['full_variables']) +
                          len(self.options['amsrenv_variables']) +
                          len(self.options['auxiliary_variables']) +
                          len(self.options['viirs_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        # Get random index to crop from.
        row_rand = np.random.randint(
            low=0, high=scene['SIC'].values.shape[0]
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=scene['SIC'].values.shape[1]
            - self.options['patch_size'])
        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rand / self.options['amsrenv_delta']
        # Used in determining the location of the crop in between pixels.
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))
        amsrenv_row_index_crop = amsrenv_row_dec * self.options['amsrenv_delta'] * amsrenv_row_dec
        amsrenv_col = col_rand / self.options['amsrenv_delta']
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options['amsrenv_delta'] * amsrenv_col_dec

        # - Discard patches with too many meaningless pixels (optional).
        
        ### Todo: consider increasing the VIIRS patch threshold ###       
        if (np.sum(viirs_ist[row_rand: row_rand + self.options['patch_size'], col_rand: col_rand + self.options['patch_size']] != self.options['train_fill_value']) != 0 ) and (np.sum(scene['SIC'].values[row_rand: row_rand + self.options['patch_size'], col_rand: col_rand + self.options['patch_size']] != self.options['class_fill_values']['SIC']) > 1):
        ### VIIRS ###

            # Crop full resolution variables.
            patch[0:len(self.options['full_variables']), :, :] = \
                scene[self.options['full_variables']].isel(
                sar_lines=range(row_rand, row_rand +
                                self.options['patch_size']),
                sar_samples=range(col_rand, col_rand
                                  + self.options['patch_size'])).to_array().values
            if len(self.options['amsrenv_variables']) > 0:
                # Crop and upsample low resolution variables.
                patch[len(self.options['full_variables']):len(self.options['full_variables'])+len(self.options['amsrenv_variables']), :, :] = torch.nn.functional.interpolate(
                    input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values[
                        :,
                        int(amsrenv_row): int(amsrenv_row + np.ceil(self.options['amsrenv_patch'])),
                        int(amsrenv_col): int(amsrenv_col + np.ceil(self.options['amsrenv_patch']))]
                    ).unsqueeze(0),
                    size=self.options['amsrenv_upsample_shape'],
                    mode=self.options['loader_upsampling']).squeeze(0)[
                    :,
                    int(np.around(amsrenv_row_index_crop)): int(np.around
                                                                (amsrenv_row_index_crop
                                                                 + self.options['patch_size'])),
                    int(np.around(amsrenv_col_index_crop)): int(np.around
                                                                (amsrenv_col_index_crop
                                                                 + self.options['patch_size']))].numpy()
            # Only add auxiliary_variables if they are called
            if len(self.options['auxiliary_variables']) > 0:

                aux_feat_list = []

                if 'aux_time' in self.options['auxiliary_variables']:
                    # Get Scene time
                    scene_id = scene.attrs['scene_id']
                    # Convert Scene time to number data
                    norm_time = get_norm_month(scene_id)

                    #
                    time_array = np.full((self.options['patch_size'],
                                         self.options['patch_size']), norm_time)

                    aux_feat_list.append(time_array)

                if 'aux_lat' in self.options['auxiliary_variables']:
                    # Get Latitude
                    lat_array = scene['sar_grid2d_latitude'].values

                    lat_array = (lat_array - self.options['latitude']['mean'])/self.options['latitude']['std']

                    # Interpolate to size of original scene
                    inter_lat_array = torch.nn.functional.interpolate(input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])), size=scene['nersc_sar_primary'].values.shape,
                                                                      mode=self.options['loader_upsampling']).numpy()
                    # Crop to correct patch size
                    crop_inter_lat_array = inter_lat_array[0, 0, row_rand: row_rand + self.options['patch_size'],
                                                           col_rand: col_rand + self.options['patch_size']]
                    # Append to array
                    aux_feat_list.append(crop_inter_lat_array)

                if 'aux_long' in self.options['auxiliary_variables']:
                    # Get Longuitude
                    long_array = scene['sar_grid2d_longitude'].values

                    long_array = (long_array - self.options['longitude']['mean'])/self.options['longitude']['std']

                    # Interpolate to size of original scene
                    inter_long_array = torch.nn.functional.interpolate(input=torch.from_numpy(long_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])), size=scene['nersc_sar_primary'].values.shape,
                                                                       mode=self.options['loader_upsampling']).numpy()
                    # Crop to correct patch size
                    crop_inter_long_array = inter_long_array[0, 0, row_rand: row_rand + self.options['patch_size'],
                                                             col_rand: col_rand + self.options['patch_size']]
                    # Append to array
                    aux_feat_list.append(crop_inter_long_array)

                aux_np_array = np.stack(aux_feat_list, axis=0)

                patch[len(self.options['full_variables']) + len(self.options['amsrenv_variables']):len(self.options['full_variables']) + len(self.options['amsrenv_variables'])+len(self.options['auxiliary_variables']), :, :] = aux_np_array

            ### VIIRS ###
            if len(self.options['viirs_variables']) > 0:
                # if it throws an error here, confirm that this is the right data type and or patch
                patch[len(self.options['full_variables']) + len(self.options['amsrenv_variables']) + len(self.options['auxiliary_variables']):, :, :] = viirs_ist[row_rand: row_rand + self.options['patch_size'], col_rand: col_rand + self.options['patch_size']]
            ### VIIRS ###
            
            # Separate in to x (train variables) and y (targets) and downscale if needed
            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)

            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def random_crop_downsample(self, idx):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        idx :
            Index from self.files to parse 

        Returns
        -------
        patch :
            Numpy array with shape (len(train_variables),
            patch_height, patch_width). None if empty patch.
        """

        patch = np.zeros((len(self.options['full_variables']) +
                          len(self.options['amsrenv_variables']) +
                          len(self.options['auxiliary_variables']) + 
                          len(self.options['viirs_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        # Get random index to crop from.
        row_rand = np.random.randint(
            low=0, high=self.scenes[idx].size(1)
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=self.scenes[idx].size(2)
            - self.options['patch_size'])

        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rand / self.options['amsrenv_delta']
        # Used in determining the location of the crop in between pixels.
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))
        amsrenv_row_index_crop = amsrenv_row_dec * self.options['amsrenv_delta'] * amsrenv_row_dec
        amsrenv_col = col_rand / self.options['amsrenv_delta']
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options['amsrenv_delta'] * amsrenv_col_dec

        # - Discard patches with too many meaningless pixels (optional).

        ### VIIRS ###
        sic_check = np.sum(self.scenes[idx][0, row_rand: row_rand + self.options['patch_size'], col_rand: col_rand + self.options['patch_size']].numpy() != self.options['class_fill_values']['SIC']) 

        viirs = self.scenes_viirs[idx]

        viirs_check = np.sum(viirs[row_rand:row_rand+self.options['patch_size'], col_rand:col_rand+self.options['patch_size']] != self.options['train_fill_value'])

        if viirs_check != 0 and sic_check > 1:
            patch[0:len(self.options['full_variables']), :, :] = \
                self.scenes[idx][:, row_rand:row_rand + int(self.options['patch_size']),
                                 col_rand:col_rand + int(self.options['patch_size'])].numpy()

            if len(self.options['amsrenv_variables']) > 0:
                # Crop and upsample low resolution variables.
                amsrenv = torch.from_numpy(self.amsr_scenes[idx][:,
                                                           int(amsrenv_row): int(amsrenv_row + np.ceil(self.options['amsrenv_patch'])),
                                                           int(amsrenv_col): int(amsrenv_col + np.ceil(self.options['amsrenv_patch']))]
                                           ).unsqueeze(0)
                # Add padding in case the patch size return is smaller than the expected one. 
                if amsrenv.size(2) < self.options['amsrenv_patch']:
                    height_pad = int(np.ceil(self.options['amsrenv_patch'])) - amsrenv.size(2)
                else:
                    height_pad = 0

                if amsrenv.size(3) < self.options['amsrenv_patch']:
                    width_pad = int(np.ceil(self.options['amsrenv_patch'])) - amsrenv.size(3)
                else:
                    width_pad = 0

                if height_pad > 0 or width_pad > 0:
                    amsrenv = torch.nn.functional.pad(amsrenv, (0, width_pad, 0, height_pad), mode='constant', value=0)
                # TODO The square bracket part is redundant []. for Example if size=2560 then doing [0:2560] after interpolate is redundant
                amsrenv = torch.nn.functional.interpolate(
                    input=amsrenv,
                    size=self.options['amsrenv_upsample_shape'],
                    mode=self.options['loader_upsampling']).squeeze(0)[
                    :,
                    int(np.around(amsrenv_row_index_crop)): int(np.around
                                                                (amsrenv_row_index_crop
                                                                 + self.options['patch_size'])),
                    int(np.around(amsrenv_col_index_crop)): int(np.around
                                                                (amsrenv_col_index_crop
                                                                 + self.options['patch_size']))]
                
                patch[len(self.options['full_variables']):len(self.options['full_variables']) +
                      len(self.options['amsrenv_variables']), :, :] = amsrenv.numpy()

            # Only add auxiliary_variables if they are called
            # No need to do the patch
            if len(self.options['auxiliary_variables']) > 0:
                patch[len(self.options['full_variables']) + len(self.options['amsrenv_variables']):len(self.options['full_variables']) + len(self.options['amsrenv_variables'])+len(self.options['auxiliary_variables']), :, :] = self.aux[idx][0, :, row_rand: row_rand + self.options['patch_size'], col_rand: col_rand + self.options['patch_size']]
            ### VIIRS ###
            if len(self.options['viirs_variables']) > 0:
                # if it throws an error here, confirm that this is the right data type and or patch
                patch[len(self.options['full_variables']) + len(self.options['amsrenv_variables']) + len(self.options['auxiliary_variables']):, :, :] = viirs[row_rand:row_rand+self.options['patch_size'], col_rand:col_rand+self.options['patch_size']]
            ### VIIRS ###

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)

            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)
        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def prep_dataset(self, x_patches, y_patches):

        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        x_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] containing only the trainable variables.
        y_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] contrainng only the targets.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """

        # Convert training data to tensor float.
        x = x_patches.type(torch.float)

        # Store charts in y dictionary.

        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = y_patches[:, idx].type(torch.long)

        return x, y    

    def transform(self, x_patch, y_patch):
        
        data_aug_options = self.options['data_augmentations']
        if torch.rand(1) < data_aug_options['Random_h_flip']:
            x_patch = TF.hflip(x_patch)
            y_patch = TF.hflip(y_patch)

        if torch.rand(1) < data_aug_options['Random_v_flip']:
            x_patch = TF.vflip(x_patch)
            y_patch = TF.vflip(y_patch)

        assert (data_aug_options['Random_rotation'] <= 180)
        if data_aug_options['Random_rotation'] != 0 and \
                torch.rand(1) < data_aug_options['Random_rotation_prob']:
            random_degree = np.random.randint(-data_aug_options['Random_rotation'],
                                                data_aug_options['Random_rotation']
                                                )
        else:
            random_degree = 0

        scale_diff = data_aug_options['Random_scale'][1] - \
            data_aug_options['Random_scale'][0]
        assert (scale_diff >= 0)
        if scale_diff != 0 and torch.rand(1) < data_aug_options['Random_scale_prob']:
            random_scale = np.random.rand()*(data_aug_options['Random_scale'][1] -
                                                data_aug_options['Random_scale'][0]) +\
                data_aug_options['Random_scale'][0]
        else:
            random_scale = data_aug_options['Random_scale'][1]

        x_patch = TF.affine(x_patch, angle=random_degree, translate=(0, 0),
                            shear=0, scale=random_scale, fill=0)
        y_patch = TF.affine(y_patch, angle=random_degree, translate=(0, 0),
                            shear=0, scale=random_scale, fill=255)
        
        return x_patch, y_patch

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Placeholder to fill with data.

        x_patches = torch.zeros((self.options['batch_size'], len(self.options['train_variables']),
                                 self.options['patch_size'], self.options['patch_size']))
        y_patches = torch.zeros((self.options['batch_size'], len(self.options['charts']),
                                 self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(
                low=0, high=len(self.ai4arctic_files), size=1).item()

            # - Extract patches
            try:
                if self.downsample:
                    x_patch, y_patch = self.random_crop_downsample(scene_id)
                else:
                    scene = xr.open_dataset(os.path.join(
                        self.options['path_to_train_data'], self.ai4arctic_files[scene_id]), engine='h5netcdf')
                    ### VIIRS ###
                    scene_viirs = xr.open_dataset(os.path.join(
                        self.options['path_to_train_data'], self.viirs_files[scene_id]), engine='h5netcdf')
                    viirs_ist = self.mask_viirs(scene_viirs)
                    ### VIIRS ###                    
                    x_patch, y_patch = self.random_crop(scene, viirs_ist) 

            except Exception as e:
                if self.downsample:
                    print(f"Cropping in {self.ai4arctic_files[scene_id]} failed.")
                    print(f"Scene size: {self.scenes[scene_id][0].shape} for crop shape: \
                        ({self.options['patch_size']}, {self.options['patch_size']})")
                    print(f"CHECK FOR POTENTIAL DOWNSAMPLING VIIRS ISSUE WITH {self.viirs_files[scene_id]}")
                    print('Skipping scene.')
                    continue
                else:
                    print(f"Cropping in {self.ai4arctic_files[scene_id]} failed.")
                    print(f"Scene size: {scene['SIC'].values.shape} for crop shape: \
                        ({self.options['patch_size']}, {self.options['patch_size']})")
                    print(f"CHECK FOR POTENTIAL VIIRS ISSUE WITH {self.viirs_files[scene_id]}")
                    print('Skipping scene.')
                    continue

            if x_patch is not None:
                if self.do_transform:
                    x_patch, y_patch = self.transform(x_patch, y_patch)
                    
                # -- Stack the scene patches in patches
                x_patches[sample_n, :, :, :] = x_patch
                y_patches[sample_n, :, :, :] = y_patch
                sample_n += 1  # Update the index.

        if self.do_transform and torch.rand(1) < self.options['data_augmentations']['Cutmix_prob']:
            lam = np.random.beta(self.options['data_augmentations']['Cutmix_beta'],
                                  self.options['data_augmentations']['Cutmix_beta'])
            rand_index = torch.randperm(x_patches.size(0))
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)
            x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]

        # Prepare training arrays

        x, y = self.prep_dataset(x_patches, y_patches)

        return x, y

class TestDataset(Dataset):

    def __init__(self, options, ai4arctic_files, viirs_files, mode='test'):

        self.options = options
        self.ai4arctic_files = ai4arctic_files
        self.viirs_files = viirs_files

        # if mode not in ["train_val", "test_val", "test"]:
        if mode not in ["train", "test", "test_no_gt"]:
            raise ValueError("String variable must be one of 'train', 'test', or 'test_no_gt'")
        self.mode = mode    

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.ai4arctic_files) 

    def prep_scene(self, scene, viirs_ist):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches
        from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """

        x_feat_list = []

        sar_var_x = torch.from_numpy(
            scene[self.options['sar_variables']].to_array().values).unsqueeze(0)

        x_feat_list.append(sar_var_x)

        size = scene['nersc_sar_primary'].values.shape

        if len(self.options['amsrenv_variables']) > 0:
            asmr_env__var_x = torch.nn.functional.interpolate(input=torch.from_numpy(
                scene[self.options['amsrenv_variables']].to_array().values).unsqueeze(0),
                size=size,
                mode=self.options['loader_upsampling'])

            x_feat_list.append(asmr_env__var_x)

        # Only add auxiliary_variables if they are called

        if len(self.options['auxiliary_variables']) > 0:

            if 'aux_time' in self.options['auxiliary_variables']:
                # Get Scene time
                scene_id = scene.attrs['scene_id']
                # Convert Scene time to number data
                norm_time = get_norm_month(scene_id)

                time_array = torch.from_numpy(
                    np.full(scene['nersc_sar_primary'].values.shape, norm_time)).view(1, 1, size[0], size[1])

                x_feat_list.append(time_array,)

            if 'aux_lat' in self.options['auxiliary_variables']:
                # Get Latitude
                lat_array = scene['sar_grid2d_latitude'].values

                lat_array = (lat_array - self.options['latitude']['mean'])/self.options['latitude']['std']

                # Interpolate to size of original scene
                inter_lat_array = torch.nn.functional.interpolate(input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])), size=size,
                                                                  mode=self.options['loader_upsampling'])

                # Append to array
                x_feat_list.append(inter_lat_array)

            if 'aux_long' in self.options['auxiliary_variables']:
                # Get Longitude
                long_array = scene['sar_grid2d_longitude'].values

                long_array = (long_array - self.options['longitude']['mean'])/self.options['longitude']['std']

                # Interpolate to size of original scene
                inter_long_array = torch.nn.functional.interpolate(input=torch.from_numpy(long_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])), size=size,
                                                                   mode=self.options['loader_upsampling'])

                # Append to array
                x_feat_list.append(inter_long_array)
    
        ### VIIRS ###
        if len(self.options['viirs_variables']) > 0: # not really needed since the whole thing breaks if viirs isn't included :)
            viirs_ist_var_x =torch.from_numpy(viirs_ist).unsqueeze(0).unsqueeze(0) # gets shape (1,1,dim_x,dim_y)
            x_feat_list.append(viirs_ist_var_x)
        ### VIIRS ###

        x = torch.cat(x_feat_list, axis=1)

        # Downscale if needed
        if (self.options['down_sample_scale'] != 1):
            x = torch.nn.functional.interpolate(
                x, scale_factor=1/self.options['down_sample_scale'], mode=self.options['loader_downsampling'])

        # TODO: 
        if self.mode != 'test_no_gt': ### Q: gt = ground truth??
            y_charts = torch.from_numpy(scene[self.options['charts']].isel().to_array().values).unsqueeze(0)
            y_charts = torch.nn.functional.interpolate(
                y_charts, scale_factor=1/self.options['down_sample_scale'], mode='nearest')

            y = {}

            for idx, chart in enumerate(self.options['charts']):
                y[chart] = y_charts[:, idx].squeeze().numpy()

        else:
            y = None

        return x.float(), y

    def mask_viirs(self, scene_viirs):
        data = scene_viirs.variables['IST'].values
        flags = [0,1,2,3,4,5,6,33]
        mask_indices = np.full_like(data, False)
        mask_indices = np.isin(data,flags) | np.isnan(data)
        data[mask_indices] = 0

        data_no_zeros = data[mask_indices == False]

        # Calculate mean and standard deviation
        mean = np.mean(data_no_zeros)
        std_dev = np.std(data_no_zeros)

        norm_data = (data - mean)/std_dev
        norm_data[mask_indices] = 0
        
        return norm_data

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.

        """
        if self.mode == 'test' or  self.mode == 'test_no_gt':
            scene = xr.open_dataset(os.path.join(
                self.options['path_to_test_data'], self.ai4arctic_files[idx]), engine='h5netcdf')
            ### VIIRS ###
            scene_viirs = xr.open_dataset(os.path.join(
                self.options['path_to_test_data'], self.viirs_files[idx]), engine='h5netcdf')
            viirs_ist = self.mask_viirs(scene_viirs)
            ### VIIRS ###

        elif self.mode == 'train':
            scene = xr.open_dataset(os.path.join(
                self.options['path_to_train_data'], self.ai4arctic_files[idx]), engine='h5netcdf')
            ### VIIRS ###
            scene_viirs = xr.open_dataset(os.path.join(
                self.options['path_to_train_data'], self.viirs_files[idx]), engine='h5netcdf')
            viirs_ist = self.mask_viirs(scene_viirs)
            ### VIIRS ###

        x, y = self.prep_scene(scene, viirs_ist)
        name = self.ai4arctic_files[idx]

        if self.mode != 'vno_gt':
            cfv_masks = {}
            for chart in self.options['charts']:
                cfv_masks[chart] = (
                    y[chart] == self.options['class_fill_values'][chart]).squeeze()
        else:
            cfv_masks = None

        tfv_mask = (x.squeeze()[0, :, :] ==
                    self.options['train_fill_value']).squeeze()

        original_size = scene['nersc_sar_primary'].values.shape

        return x, y, cfv_masks, tfv_mask, name, original_size


def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with amsrenv options.
        Updated with correct true patch size
    """
    
    train_options['amsrenv_delta'] = train_options['amsrenv_pixel_spacing'] / \
        (train_options['pixel_spacing']*train_options['down_sample_scale'])

    train_options['amsrenv_patch'] = train_options['patch_size'] / \
        train_options['amsrenv_delta']
    train_options['amsrenv_patch_dec'] = int(
        train_options['amsrenv_patch'] - int(train_options['amsrenv_patch']))
    train_options['amsrenv_upsample_shape'] = (int(train_options['patch_size'] +
                                                   train_options['amsrenv_patch_dec'] *
                                                   train_options['amsrenv_delta']),
                                               int(train_options['patch_size'] +
                                                   train_options['amsrenv_patch_dec'] *
                                                   train_options['amsrenv_delta']))
    train_options['sar_variables'] = [variable for variable in train_options['train_variables']
                                      if 'sar' in variable or 'map' in variable]
    train_options['full_variables'] = np.hstack((train_options['charts'], train_options['sar_variables']))
    train_options['amsrenv_variables'] = [variable for variable in train_options['train_variables']
                                          if 'sar' not in variable and 'map' not in variable and 'aux' not in variable and 'viirs' not in variable] # VIIRS
    train_options['auxiliary_variables'] = [
        variable for variable in train_options['train_variables'] if 'aux' in variable]

    ### VIIRS ###
    train_options['viirs_variables'] = [variable for variable in train_options['train_variables'] 
                                        if 'viirs' in variable]
    ### VIIRS ###

    return train_options


def get_norm_month(file_name):
    #print("DEF get_norm_month")
    pattern = re.compile(r'\d{8}T\d{6}')

    # Search for the first match in the string
    match = re.search(pattern, file_name)

    first_date = match.group(0)

    # parse the date string into a datetime object
    date = datetime.datetime.strptime(first_date, "%Y%m%dT%H%M%S")

    # calculate the number of days between January 1st and the given date

    delta = relativedelta.relativedelta(date, datetime.datetime(date.year, 1, 1))

    months = delta.months
    norm_months = 2*months/11-1

    return norm_months