# -- Built-in modules -- #
import os

import os.path as osp

import json
import xarray as xr
# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torchmetrics.functional import r2_score, f1_score, accuracy
from tqdm import tqdm  # Progress bar
import torch.nn.functional as F
import math
# -- Proprietary modules -- #

from utils import ICE_STRINGS, GROUP_NAMES
from unet import UNet, UNet_feature_fusion, UNet_regression, UNet_regression_feature_fusion, UNet_regression_var
from wnet import WNet
from wnet_separate_decoders import WNet_Separate_Decoders
from wnet_separate_viirs_decoders import WNet_Separate_VIIRS_Decoders
from wnet_separate_viirs import WNet_Separate_VIIRS
from wnet_uncertainty import WNet_Uncertainty
from unet_uncertainty import UNet_regression_uncertainty
from wnet_separate_viirs_uncertainty import WNet_Separate_VIIRS_Uncertainty
from wnet_separate_decoders_uncertainty import WNet_Separate_Decoders_Uncertainty
from r2_replacement import r2_score_random 

#from ViT import SegmentationViT

def generate_pixels_per_class_bar_graph(chart, pixels_per_class, n_pixels, x_label, bar_width, title):
  labels = [GROUP_NAMES[chart][i] for i in range(len(pixels_per_class))]
  percent_class = [float((class_n/n_pixels)*100) for class_n in pixels_per_class]

  plt.figure(figsize=(6, 5))
  bars = plt.bar(labels, percent_class,width=bar_width)

  for bar, class_n in zip(bars, percent_class):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{class_n:.2f}%", ha='center', va='bottom', fontsize=10)
    
  plt.xlabel(x_label)
  plt.ylabel("Pixel Count Percentage [%]")
  plt.xticks(list(GROUP_NAMES[chart].values()), rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig(f"{title}-{chart}-BAR-CHART.png", dpi=150)
  plt.close('all')
 

def chart_cbar(ax, n_classes, chart, cmap='vridis'):

    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    arranged = np.arange(0, n_classes)
    cmap = plt.get_cmap(cmap, n_classes - 1)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the mask class.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label=ICE_STRINGS[chart], fontsize=12)
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))

def cbar_ice_classification(ax, n_classes, cmap='vridis'):

    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    LABELS = {0: 'Water', 1: 'Marginal\n Ice', 2: 'Consolidated\n Ice'}
    CLABEL = 'Ice Classification'
    arranged = np.arange(0, n_classes)
    cmap = plt.get_cmap(cmap, n_classes - 1)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the mask class.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label= CLABEL, fontsize=12)
    cbar.set_ticklabels(list(LABELS.values()))

def classify_from_SIC(SIC):
    SIC_classes = np.zeros_like(SIC, dtype=int)
    SIC_classes[(SIC >= 2) & (SIC <= 8)] = 1  # marginal ice
    SIC_classes[(SIC > 8) & (SIC <= 100)] = 2  # consolidated ice
    SIC_classes[SIC == 255] = 255
    return SIC_classes

def classify_SIC_tensor(data):
  class_data = torch.zeros_like(data, dtype=torch.int32)

  class_data[(data >= 2) & (data <= 8)] = 1     # 2 < x <= 8 -> 1
  class_data[(data > 8) & (data <= 100)] = 2   # 8 < x <= 100 -> 2
  class_data[data == 255] = 255

  return class_data

def accuracy_metric(preds,target):
  acc = accuracy(preds,target, task="multiclass", num_classes=3)
  acc = float(acc)*100

  return acc

def compute_metrics(true, pred, charts, metrics, num_classes):

    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores

def r2_metric_rand(true, pred, num_classes):

    """
    Calculate the r2 metric random.

     where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i_rand-f(x_i)_rand)^2` is total sum of squares. 
    Where y_i_rand is a random class and f(x_i)_rand is also a random class. 
    Therefore y_i_rand and f(x_i)_rand are random discrete variables with a uniform discrete distribution. 
    Calculating SS_tot can be expensive due to number of random number that need to be generated. 
    Thus instead it is calculated based on the expectation value of (y_rand-f(x)_rand).

    Futhermore, it is expected that all class inside the ground truth go from 0 to N-1
    where N is the number of classes.  Additionally all predictions should also be bound and go from 
    0 to N-1. 


    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  


    Returns
    -------
    r2_rand : float
        The calculated r2 score.

    """
    if (not len(pred)==0) and not (len(true)==0):
        r2 = r2_score_random(preds=pred, target=true,num_classes=num_classes)
    else:
        r2 = torch.tensor(float("nan"))

    return r2

def r2_metric(true, pred, num_classes=None):
    
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  


    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2

def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1

def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)

# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int):
    '''
    Saves the input model in the inside the directory "/work_dirs/"experiment_name"/
    The models with be save as best_model.pth.
    The following are stored inside best_model.pth
        model_state_dict
        optimizer_state_dict
        epoch
        train_options


    Parameters
    ----------
    cfg : mmcv.Config
        The config file object of mmcv
    train_options : Dict
        The dictory which stores the train_options from quickstart
    net :
        The pytorch model
    optimizer :
        The optimizer that the model uses.
    epoch: int
        The epoch number

    '''
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)

    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_options': train_options
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None):
    """
    Loads a PyTorch model from a checkpoint file and returns the model, optimizer, and scheduler.
    :param model: PyTorch model to load
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: PyTorch optimizer to load (optional)
    :param scheduler: PyTorch scheduler to load (optional)
    :return: If optimizer and scheduler are provided, return the model, optimizer, and scheduler.
    """

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    return epoch

def rand_bbox(size, lam):
    '''
    Given the 4D dimensions of a batch (size), and the ratio 
    of the spatial dimension (lam) to be cut, returns a bounding box coordinates
    used for cutmix

    Parameters
    ----------
    size : 4D shape of the batch (N, C, H, W)
    lam : Ratio (portion) of the input to be cutmix'd

    Returns 
    ----------
    Bounding box (x1, y1, x2, y2)
    '''
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(H)
    cy = np.random.randint(W)

    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2

def class_decider(output, train_options, chart):
    # normal
    if (train_options['binary_water_classifier'] == False):
        if output.size(3) == 1:
            output = torch.round(output.squeeze())
            output = torch.clamp(output, min=0, max=train_options
                                 ['n_classes'][chart])
        else:
            output = torch.argmax(output, dim=1).squeeze()
        return output

    # if regression head return output
    # class water
    else:
        probability = torch.nn.Softmax(dim=1)(output)
        water = probability[:, 0, :, :]
        not_water = torch.sum(probability, dim=1) - water
        class_output = water <= not_water
        without_water = probability[:, 1:, :, :]
        class_output_without_water = torch.argmax(without_water, dim=1) + 1
        class_output = class_output_without_water * class_output

        return class_output.squeeze()

def compute_classwise_f1score(true, pred, charts, num_classes):
    """ This function computes the classwise evaluation score for each task and stores them in a dic

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns score_dictionary
    """
    score = {}
    for chart in charts:
        print('True: ', true[chart].shape, torch.unique(true[chart]))
        print('Pred: ', pred[chart].shape, torch.unique(pred[chart]))

        score[chart] = f1_score(target=true[chart], preds=pred[chart], average='none',
                                task='multiclass', num_classes=num_classes[chart])
    return score


def create_train_validation_and_test_scene_list(train_options):
    '''
    Creates the a train and validation scene list. Adds these two list to the config file train_options

    '''

    # Train ------------
    with open(train_options['path_to_env'] + train_options['train_list_path']) as file:
        train_options['train_list'] = json.loads(file.read())

    # Convert the original scene names to the preprocessed names.
    train_options['train_list'] = [file[17:32] + '_' + file[77:80] +
                                   '_prep.nc' for file in train_options['train_list']]

    ### VIIRS ###
    with open(train_options['path_to_env'] + train_options['train_viirs']) as file:
        train_options['train_list_viirs'] = json.loads(file.read())
    ### VIIRS ###

    ##############################################################
    generate_train_bar_graph = False # only set to true if the train/val dataset changes  # add to config in future?

    if generate_train_bar_graph:
        with open(train_options['path_to_env'] + 'datalists/train_dataset_cross_validation.json') as file:
            bar_data = json.loads(file.read())
        bar_data = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in bar_data]
        
        print("TRAIN BAR CHARTS")
        x_axis_label = ["Sea Ice Concentration [%]", "Stage of Development", "Floe Size"]
        bar_widths = [5, 0.8, 0.8]

        n_pixels = np.zeros((3,1))
        pixels_per_class = np.zeros((3,train_options['n_classes']['SIC']-1))

        for data_file in bar_data:
            print(data_file)
            scene = xr.open_dataset('./dataset/' + data_file, engine='h5netcdf')            

            for idx, chart in enumerate(train_options['charts']):
                chart_data = scene.variables[chart].values

                for i in range(0, train_options['n_classes'][chart] - 1):
                    n_class = np.count_nonzero(chart_data == i) 
                    pixels_per_class[idx][i] += n_class
                    n_pixels[idx] += n_class

        title = './work_dir/opt_test_bar_charts/train_bar_charts' # note that this needs to change in future
        for idx, chart in enumerate(train_options['charts']):
            generate_pixels_per_class_bar_graph(chart, pixels_per_class[idx][0:train_options['n_classes'][chart] - 1], n_pixels[idx], x_axis_label[idx], bar_widths[idx], title)
    ##############################################################

    # Validation ---------
    if train_options['cross_val_run']:

        '''
        Three methods for selecting the validation set.
        Note: in future, could add a condition to the config to select one.
        '''

        ### Set validation set ###

        # Validation set is fixed for comparison across models
        # Conditional statement avoids file system error 
        if train_options['p-fold'] == 48:
            print('last file in fold 48 causes an error')
            train_options['validate_list'] = np.array(train_options['train_list'])[train_options['p-fold']:train_options['p-fold']+11] 
            ### VIIRS ###
            train_options['validate_list_viirs'] = np.array(train_options['train_list_viirs'])[train_options['p-fold']:train_options['p-fold']+11] 
            ### VIIRS ###
        else:
            train_options['validate_list'] = np.array(train_options['train_list'])[train_options['p-fold']:train_options['p-fold']+12]
            ### VIIRS ###
            train_options['validate_list_viirs'] = np.array(train_options['train_list_viirs'])[train_options['p-fold']:train_options['p-fold']+12] 
            ### VIIRS ###

        ### Set validation set ###
        
    else:
        # load validation list
        with open(train_options['path_to_env'] + train_options['val_path']) as file:
            train_options['validate_list'] = json.loads(file.read())
        # Convert the original scene names to the preprocessed names.
        train_options['validate_list'] = [file[17:32] + '_' + file[77:80] +
                                          '_prep.nc' for file in train_options['validate_list']]
        
        ### VIIRS ###
        with open(train_options['path_to_env'] + train_options['validate_viirs']) as file:
            train_options['validate_list_viirs'] = json.loads(file.read())       
        ### VIIRS ###

    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]
    ### VIIRS ###
    train_options['train_list_viirs'] = [scene for scene in train_options['train_list_viirs']
                                   if scene not in train_options['validate_list_viirs']]
    ### VIIRS ###

    # Test ----------
    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
        train_options['test_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc'
                                      for file in train_options['test_list']]

    ### VIIRS ###
    with open(train_options['path_to_env'] + train_options['test_viirs']) as file:
        train_options['test_list_viirs'] = json.loads(file.read())
    ### VIIRS ###

    print('Options initialised') # Allows for verification that there aren't repeated files across the data lists
    print('train')
    print(train_options['train_list'])
    print(train_options['train_list_viirs'])
    print('test')
    print(train_options['test_list'])
    print(train_options['test_list_viirs'])
    print('validate')
    print(train_options['validate_list'])
    print(train_options['validate_list_viirs'])

def pad_and_infer(model, image, train_size=256):
    N, C, H, W = image.shape
    
    # Calculate padding to make dimensions divisible by train_size (256)
    pad_h = (train_size - H % train_size) % train_size
    pad_w = (train_size - W % train_size) % train_size
    
    # Pad the image
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Calculate the number of 256x256 patches
    num_patches_h = math.ceil((H + pad_h) / train_size)
    num_patches_w = math.ceil((W + pad_w) / train_size)
    
    # Perform inference
    with torch.no_grad():
        # Process the image in 256x256 patches
        output_list = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = padded_image[:, :, i*train_size:(i+1)*train_size, j*train_size:(j+1)*train_size]
                patch_output = model(patch)
                output_list.append(patch_output)
        
        # Stitch the patches back together
        output = {}
        for task in output_list[0].keys():
            task_outputs = [patch[task] for patch in output_list]
            
            stitched_task = torch.cat([torch.cat(task_outputs[i*num_patches_w:(i+1)*num_patches_w], dim=3) for i in range(num_patches_h)], dim=2)
            
            output[task] = stitched_task
    
    # Unpad the outputs
    unpadded_output = {}
    for task, pred in output.items():

        unpadded_output[task] = pred[:, :, :H, :W]
    
    return {
        'SIC': unpadded_output['SIC'],  # TaskA renamed to SIC
        'SOD': unpadded_output['SOD'],  # TaskB renamed to SOD
        'FLOE': unpadded_output['FLOE']  # TaskC renamed to FLOE
    }

def get_scheduler(train_options, optimizer):
    if train_options['scheduler']['type'] == 'CosineAnnealingLR':
        T_max = train_options['epochs']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=train_options['scheduler']['lr_min'])
    elif train_options['scheduler']['type'] == 'CosineAnnealingWarmRestartsLR':
        T_0 = train_options['scheduler']['EpochsPerRestart']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,
                                                                         T_mult=train_options['scheduler']['RestartMult'],
                                                                         eta_min=train_options['scheduler']['lr_min'],
                                                                         last_epoch=-1,
                                                                         verbose=False)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5, last_epoch=- 1,
                                                        verbose=False)
    return scheduler


def get_optimizer(train_options, net):
    if train_options['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                     betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                     weight_decay=train_options['optimizer']['weight_decay'])

    elif train_options['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                      betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                      weight_decay=train_options['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(list(net.parameters()), lr=train_options['optimizer']['lr'],
                                    momentum=train_options['optimizer']['momentum'],
                                    dampening=train_options['optimizer']['dampening'],
                                    weight_decay=train_options['optimizer']['weight_decay'],
                                    nesterov=train_options['optimizer']['nesterov'])
    return optimizer

def get_loss(loss, chart=None, **kwargs):

    """_summary_

    Args:
        loss (str): the name of the loss
    Returns:
        loss: The corresponding
    """

    if loss == 'CrossEntropyLoss':
        kwargs.pop('type')
        loss = torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == 'BinaryCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = torch.nn.BCELoss(**kwargs)
    elif loss == 'OrderedCrossEntropyLoss':
        from losses import OrderedCrossEntropyLoss
        kwargs.pop('type')
        loss = OrderedCrossEntropyLoss(**kwargs)
    elif loss == 'MSELossFromLogits':
        from losses import MSELossFromLogits
        kwargs.pop('type')
        loss = MSELossFromLogits(chart=chart, **kwargs)
    elif loss == 'MSELoss':
        kwargs.pop('type')
        loss = torch.nn.MSELoss(**kwargs)
    elif loss == 'MSELossWithIgnoreIndex':
        from losses import MSELossWithIgnoreIndex
        kwargs.pop('type')
        loss = MSELossWithIgnoreIndex(**kwargs)
    elif loss == 'GaussianNLLLoss': # also used for beta-nll loss because that loss function builds off gnll
        kwargs.pop('type')
        loss = torch.nn.GaussianNLLLoss(**kwargs)
    else:
        raise ValueError(f'The given loss \'{loss}\' is unrecognized or Not implemented')

    return loss

def get_model(train_options, device):
    if train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device) # updated with additional argument
    elif train_options['model_selection'] == 'unet_feature_fusion':
        input_channels = len(train_options['train_variables'])
        net = UNet_feature_fusion(options=train_options, input_channels=input_channels).to(device) # updated with additional argument       
    elif train_options['model_selection'] == 'unet_regression':
        net = UNet_regression(options=train_options).to(device)
    elif train_options['model_selection'] == 'unet_regression_var':
        net = UNet_regression_var(options=train_options).to(device)
    elif train_options['model_selection'] == 'unet_regression_feature_fusion':
        input_channels = len(train_options['train_variables'])
        net = UNet_regression_feature_fusion(options=train_options, input_channels=input_channels).to(device)
    elif train_options['model_selection'] =='wnet':
        net = WNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-separate-decoders':
        net = WNet_Separate_Decoders(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-separate-viirs-decoders':
        net = WNet_Separate_VIIRS_Decoders(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-separate-viirs':
        net = WNet_Separate_VIIRS(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-uncertainty':
        net = WNet_Uncertainty(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-separate-decoders-uncertainty':
        net = WNet_Separate_Decoders_Uncertainty(options=train_options).to(device)
    elif train_options['model_selection'] == 'wnet-separate-viirs-uncertainty':
        net = WNet_Separate_VIIRS_Uncertainty(options=train_options).to(device)
    elif train_options['model_selection'] == 'unet-uncertainty':
        net = UNet_regression_uncertainty(options=train_options).to(device)
    elif train_options['model_selection'] == 'vit':
        net = SegmentationViT(options=train_options).to(device)
    else: 
        raise 'Unknown model selected'
    return net