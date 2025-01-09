#!/usr/bin/env python
# -*-coding:utf-8 -*-


_base_ = ['./_base_/base.py']

SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    #'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m',
    # 'skt',
    'tcwv', 'tclw',

    # -- Auxilary Variables -- #
    'aux_time',
    'aux_lat',
    'aux_long',

    # -- VIIRS Variables -- #
    'viirs_ist'
]


train_options = {'train_variables': SCENE_VARIABLES,
                 'path_to_train_data': 'dataset',
                 'path_to_test_data': 'dataset',
                 'train_list_path': 'datalists/train_dataset_cross_validation.json', #test_train.json', #
                 #'val_path': 'datalists/validation_dataset.json', #
                 #'test_path': 'datalists/test_dataset.json', #
                 'val_path': 'datalists/test_dataset.json', #'datalists/validation_dataset.json', #
                 'test_path': 'datalists/validation_dataset.json', #'datalists/test_dataset.json', #

                 'train_viirs': 'datalists/train_dataset_viirs_cross_validation.json', #
                 'test_viirs': 'datalists/validation_dataset_viirs.json', #'datalists/test_dataset_viirs.json', #test_viirs.json',#
                 'validate_viirs': 'datalists/test_dataset_viirs.json', #'datalists/validation_dataset_viirs.json', #validate_viirs.json',#
                 #'train_viirs': 'datalists/train_dataset_viirs.json', #
                 #'test_viirs': 'datalists/test_dataset_viirs.json', #test_viirs.json',#
                 #'validate_viirs': 'datalists/validation_dataset_viirs.json', #validate_viirs.json',#


                 # p leave out cross val run
                 'cross_val_run': True,
                 'p-out': 12, # number of scenes taken from the TRAIN SET. Must change the datalist to move validation scenes to train if using
                 'p-fold': 48,
                 'compute_classwise_f1score': True,
                 'plot_confusion_matrix': True,
                 'save_nc_file': True,

                 'optimizer': {
                     'type': 'AdamW',
                     'lr': 0.0005,  # Optimizer learning rate.
                     'b1': 0.9,
                     'b2': 0.999,
                     'weight_decay': 0.01
                 },

                 'scheduler': {
                     'type': 'CosineAnnealingWarmRestartsLR',  # Name of the schedulers
                     'EpochsPerRestart': 10,  # Number of epochs for the first restart
                     # This number will be used to increase or descrase the number of epochs to restart after each restart.
                     'RestartMult': 2,
                     'lr_min': 0,  # Minimun learning rate
                 },

                 'batch_size': 4, #16,
                 'num_workers': 4,  # Number of parallel processes to fetch data.
                 'num_workers_val': 4,  # Number of parallel processes during validation.
                 'patch_size': 256,
                 'down_sample_scale': 10,

                 'data_augmentations': {
                     'Random_h_flip': 0.5,
                     'Random_v_flip': 0.5,
                     'Random_rotation_prob': 0.5,
                     'Random_rotation': 90,
                     'Random_scale_prob': 0.5,
                     'Random_scale': (0.9, 1.1),
                     'Cutmix_beta': 1.0,
                     'Cutmix_prob': 0.5,
                 },
                 # -- Model selection -- #
                 'model_selection': 'wnet-separate-viirs', #'wnet',#'unet_feature_fusion', #'unet_regression',
                 'unet_conv_filters': [32, 32, 64, 64],
                 'deconv_filters': [96, 128, 192, 192], # use if there's a mismatch with channels. corresponds with encoding [32,32,64,64]
                 'epochs': 300,  # Number of epochs before training stop.
                 'epoch_len': 500,  # Number of batches for each epoch.
                 # Size of patches sampled. Used for both Width and Height.
                 'task_weights': [1, 3, 3],
                 'chart_loss': {  # Loss for the task
                     'SIC': {
                         'type': 'MSELossWithIgnoreIndex',
                         'ignore_index': 255,
                     },
                     'SOD': {
                         'type': 'CrossEntropyLoss',
                         'ignore_index': 255,
                     },
                     'FLOE': {
                         'type': 'CrossEntropyLoss',
                         'ignore_index': 255,
                     },
                 }
                 }
