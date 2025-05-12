# -- Built-in modules -- #
import json
import os
import os.path as osp

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import xarray as xr
import pandas as pd
from tqdm import tqdm
from mmcv import mkdir_or_exist
import wandb
# --Proprietary modules -- #
from functions import chart_cbar, compute_metrics, class_decider, cbar_ice_classification, classify_from_SIC, accuracy_metric, classify_SIC_tensor, generate_pixels_per_class_bar_graph #water_edge_plot_overlay, water_edge_metric,
from loaders import TestDataset, get_variable_options
#from functions_VIIRS import slide_inference, batched_slide_inference
from torchmetrics.functional.classification import multiclass_confusion_matrix
import seaborn as sns
from utils import GROUP_NAMES

def test(mode: str, net: torch.nn.modules, checkpoint: str, device: str, cfg, test_list, test_list_viirs, test_name):

    """_summary_

    Args:
        net (torch.nn.modules): The model
        checkpoint (str): The checkpoint to the model
        device (str): The device to run the inference on
        cfg (Config): mmcv based Config object, Can be considered dict
    """

    if mode not in ["val", "test"]:
        raise ValueError("String variable must be one of 'train_val', 'test_val', or 'train'")

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)
    weights = torch.load(checkpoint)['model_state_dict']
    net.load_state_dict(weights)

    print('Model successfully loaded.')

    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name+'_'+test_name, 'dataset')
    table = wandb.Table(columns=['ID', 'Image'])

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    output_class = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Stores the flat ouputs of only one scene.
    output_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Stores the flat outputs of all scene.
    outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Stores the flat ground truth of only one scene. 
    inf_y_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Stores the flat ground truth of all scenes
    inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Outputs mask by train fill values for only one scene
    output_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Outputs mask by train fill values fo all scenes
    outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}

    ### NEW ###
    preds_SIC_class = torch.Tensor().to(device)
    target_SIC_class = torch.Tensor().to(device)
    output_preds = torch.Tensor().to(device)
    inf_y_flat_target = torch.Tensor().to(device)
    ### NEW ###

    # ### Prepare the scene list, dataset and dataloaders

    if mode == 'test':

        train_options['test_list'] = test_list
        train_options['test_list_viirs'] = test_list_viirs

        # The test data is stored in a separate folder inside the training data.
        if train_options['save_nc_file']:
            upload_package = xr.Dataset()  # To store model outputs.
        dataset = TestDataset(
            options=train_options, ai4arctic_files=train_options['test_list'], viirs_files = train_options['test_list_viirs'], mode='test')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    elif mode == 'val':
        train_options['test_list'] = test_list
        train_options['test_list_viirs'] = test_list_viirs
        # The test data is stored in a separate folder inside the training data.
        if train_options['save_nc_file']:
            upload_package = xr.Dataset()  # To store model outputs.
        dataset = TestDataset(
            options=train_options, ai4arctic_files=train_options['test_list'], viirs_files = train_options['test_list_viirs'], mode='train')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    if mode == 'val':
        inference_name = 'inference_val'
    elif mode == 'test':
        inference_name = 'inference_test'

    os.makedirs(osp.join(cfg.work_dir, inference_name), exist_ok=True)

    # Store the scores obtained for each scene so we can see how each scene performs. 
    results_per_scene = []

    net.eval()
    for inf_x, inf_y, cfv_masks, tfv_mask, scene_name, original_size in tqdm(iterable=asid_loader,
                                                               total=len(train_options['test_list']), colour='green', position=0):
        scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
        torch.cuda.empty_cache()

        inf_x = inf_x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            if train_options['model_selection'] == 'swin':
                output = slide_inference(inf_x, net, train_options, 'test')
            else:
                output = net(inf_x)
                if train_options['uncertainty'] != 0:
                    sic_output_var = output['SIC']['variance'].unsqueeze(-1).to(device)  # Variance of SIC
                    output['SIC'] = output['SIC']['mean'].unsqueeze(-1).to(device)  # Mean of SIC
                    
                    #sic_output_var = output['SIC'][..., 1].unsqueeze(-1).to(device) # may not be needed
                    #output['SIC'] = output['SIC'][..., 0].unsqueeze(-1).to(device) 

            # Up sample the masks
            tfv_mask = torch.nn.functional.interpolate(tfv_mask.type(torch.uint8).unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            for chart in train_options['charts']:
                masks_int = cfv_masks[chart].to(torch.uint8)
                masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
                    0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
                cfv_masks[chart] = torch.gt(masks_int, 0)

            # Upsample data
            if train_options['down_sample_scale'] != 1:
                for chart in train_options['charts']:
                    # check if the output is regression output, if yes, permute the dimension
                    if output[chart].size(3) == 1:
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest')
                        output[chart] = output[chart].permute(0, 2, 3, 1)
                    else:
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest')

                    if chart == 'SIC' and sic_output_var.size(3) == 1:
                        print('PERMUTE SIC OUTPUT VARIANCE')
                        sic_output_var = sic_output_var.permute(0, 3, 1, 2)
                        sic_output_var = torch.nn.functional.interpolate(
                            sic_output_var, size=original_size, mode='nearest')
                        sic_output_var = sic_output_var.permute(0, 2, 3, 1)
                    elif chart == 'SIC':
                        print('DO NOT PERMUTE SIC OUTPUT VARIANCE')
                        sic_output_var = torch.nn.functional.interpolate(
                            sic_output_var, size=original_size, mode='nearest')
                    # upscale the output
                    # if not test:
                    inf_y[chart] = torch.nn.functional.interpolate(inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
                                                                   size=original_size, mode='nearest').squeeze()

        for chart in train_options['charts']:
            ### why is class decider used here?
            output_class[chart] = class_decider(output[chart], train_options, chart).detach()
            if train_options['save_nc_file']:
                upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}",
                                                                       data=output_class[chart].squeeze().cpu().numpy().astype('uint8'),
                                                                       dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))
            output_flat[chart] = output_class[chart][~cfv_masks[chart]] 
            outputs_flat[chart] = torch.cat((outputs_flat[chart], output_flat[chart]))
            output_tfv_mask[chart] = output_class[chart][~tfv_mask].to(device)
            outputs_tfv_mask[chart] = torch.cat((outputs_tfv_mask[chart], outputs_tfv_mask[chart]))
            inf_y_flat[chart] = inf_y[chart][~cfv_masks[chart]].to(device, non_blocking=True).float()
            inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y_flat[chart]))
        
        ## do I need?? ##
        sic_output_var = sic_output_var.squeeze(0).squeeze(-1) # = torch.flatten(sic_output_var)
        #sic_var_flat = sic_var_flat[~cfv_masks['SIC']]
        #print("OUTPUT CLASS SHAPE: ", output_class['SIC'].size())
        #print("SIC VAR SHAPE: ", sic_output_var.size())
        #print("CFV MASKS SHAPE: ", cfv_masks['SIC'].size())
        ### NEW ###
        output_preds = classify_SIC_tensor(output_flat['SIC'])
        preds_SIC_class = torch.cat((preds_SIC_class, output_preds)) 
        inf_y_flat_target = classify_SIC_tensor(inf_y_flat['SIC'])
        target_SIC_class = torch.cat((target_SIC_class, inf_y_flat_target))
        ### NEW ###

        for chart in train_options['charts']: 
            inf_y[chart] = inf_y[chart].cpu().numpy()
            output_class[chart] = output_class[chart].squeeze().cpu().numpy()

        fig, axs2d = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

        axs = axs2d.flat

        for j in range(0, 3): # don't know what to put in the spot that was the water edge overlay
            ax = axs[j]
            if j == 2:
                img = torch.squeeze(inf_x, dim=0).cpu().numpy()[4]
                ax.imshow(img)

                #ax.imshow(img, cmap='gnuplot2_r')
                #arranged = np.arange(0, len(np.unique(img)))
                #cmap = plt.get_cmap('gnuplot2_r', len(np.unique(img))-1)
                #norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
                #arranged = arranged[:-1]
                #cbar = plt.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax) #, ticks = arranged, fraction=0.0485, pad=0.049, ax=ax)
            else:
                img = torch.squeeze(inf_x, dim=0).cpu().numpy()[j]
                ax.imshow(img, cmap='gray')

            if j == 0:
                ax.set_title(f'Scene {scene_name}, HH')
            elif j == 1:
                ax.set_title(f'Scene {scene_name}, HV')
            else:
                ax.set_title(f'Scene {scene_name}, 18.7 H') #IST')
            ax.set_xticks([])
            ax.set_yticks([])
            


        for idx, chart in enumerate(train_options['charts']):

            ax = axs[idx+3]
            output_class[chart] = output_class[chart].astype(float)
            output_class[chart][cfv_masks[chart]] = np.nan

            ax.imshow(output_class[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            # removed for poster format:
            #ax.set_title([f'Scene {scene_name}, {chart}: Model Prediction'])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

        for idx, chart in enumerate(train_options['charts']):

            ax = axs[idx+6]
            inf_y[chart] = inf_y[chart].astype(float)
            inf_y[chart][cfv_masks[chart]] = np.nan
            ax.imshow(inf_y[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            # removed for poster format:
            #ax.set_title([f'Scene {scene_name}, {chart}: Ground Truth'])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

        # plt.suptitle(f"Scene: {scene_name}", y=0.65)
        # plt.suptitle(f"Scene: {scene_name}", y=0)
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.5, hspace=-0)
        fig.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png",
                    format='png', dpi=300, bbox_inches="tight")
                    #format='png', dpi=128, bbox_inches="tight")
        plt.close('all')

        #### Just Plot SIC ####
        fig_SIC, axs2d_SIC = plt.subplots(nrows=4, ncols=1, figsize=(5, 28))

        axs_SIC = axs2d_SIC.flat

        for j in range(0, 2):
            ax_SIC = axs_SIC[j]
            img_SIC = torch.squeeze(inf_x, dim=0).cpu().numpy()[j]
            #if j == 0:
            #    ax_SIC.set_title(f'Scene {scene_name}, HH')
            #else:
            #    ax.set_title(f'Scene {scene_name}, HV')
            ax_SIC.set_xticks([])
            ax_SIC.set_yticks([])
            ax_SIC.imshow(img_SIC, cmap='gray')

        #ax = axs[2]
        #ax.set_title('Water Edge SIC: Red, SOD: Green,Floe: Blue')
        #edge_water_output = water_edge_plot_overlay(output_class, tfv_mask.cpu().numpy(), train_options)

        #ax.imshow(edge_water_output, vmin=0, vmax=1, interpolation='nearest')
        ##where I stopped
        for idx, chart in enumerate(train_options['charts']):

            ax_SIC = axs_SIC[idx+2]
            output_class[chart] = output_class[chart].astype(float)
            # if test is False:
            output_class[chart][cfv_masks[chart]] = np.nan
            # else:
            #     output[chart][masks.cpu().numpy()] = np.nan
            ax_SIC.imshow(output_class[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax_SIC.set_xticks([])
            ax_SIC.set_yticks([])
            # removed for poster format:
            #ax.set_title([f'Scene {scene_name}, {chart}: Model Prediction'])
            chart_cbar(ax=ax_SIC, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')
            break

        for idx, chart in enumerate(train_options['charts']):

            ax_SIC = axs_SIC[idx+3]
            inf_y[chart] = inf_y[chart].astype(float)
            inf_y[chart][cfv_masks[chart]] = np.nan
            ax_SIC.imshow(inf_y[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax_SIC.set_xticks([])
            ax_SIC.set_yticks([])
            # removed for poster format:
            #ax.set_title([f'Scene {scene_name}, {chart}: Ground Truth'])
            chart_cbar(ax=ax_SIC, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')
            break

        # plt.suptitle(f"Scene: {scene_name}", y=0.65)
        # plt.suptitle(f"Scene: {scene_name}", y=0)
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=-0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.5, hspace=-0)
        fig_SIC.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Plot.png",
                    format='png', dpi=300, bbox_inches="tight")
                    #format='png', dpi=128, bbox_inches="tight")
        plt.close('all')

        ### SIC VARIANCE ###
        sic_output_var = sic_output_var.cpu().numpy().astype(float)
        sic_output_var[cfv_masks['SIC']] = np.nan

        fig_var, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(sic_output_var)
        plt.title(scene_name)
        cbar = plt.colorbar()
        cbar.set_label(label= "Variance [%$^2$]", fontsize=12)
        fig_var.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Variance.png",
            format='png', dpi=150, bbox_inches="tight")
        plt.close('all')

        ### ADD NEW CLASSIFICATION FOR SIC METRIC ###
        fig_SIC_classes, axs2d_SIC_classes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10)) # (5, 14)
        axs_SIC_classes = axs2d_SIC_classes.flat
        ## SIC CLASSIFICATION ##
        n_classes = 4
        ax_SIC = axs_SIC_classes[0]
        converted_SIC = classify_from_SIC(output_class['SIC'])
        #print('cfv_masks[SIC]: ', cfv_masks['SIC'])
        #output_class['SIC'] = output_class['SIC'].astype(float)
        converted_SIC = converted_SIC.astype(float)
        converted_SIC[cfv_masks['SIC']] = np.nan

        ax_SIC.imshow(converted_SIC, vmin=0, vmax=n_classes - 2, cmap='jet', interpolation='nearest')
        ax_SIC.set_xticks([])
        ax_SIC.set_yticks([])
        ax_SIC.set_title('Model Prediction') # removed square brackets
        cbar_ice_classification(ax=ax_SIC, n_classes=n_classes, cmap='jet')
        ## ICE CHART ##
        ax_SIC = axs_SIC_classes[2]
        converted_GT = classify_from_SIC(inf_y['SIC'])
        converted_GT = converted_GT.astype(float)
        converted_GT[cfv_masks['SIC']] = np.nan
        ax_SIC.imshow(converted_GT, vmin=0, vmax=n_classes - 2, cmap='jet', interpolation='nearest')
        ax_SIC.set_xticks([])
        ax_SIC.set_yticks([])        
        ax_SIC.set_title('Ground Truth')
        cbar_ice_classification(ax=ax_SIC, n_classes=n_classes, cmap='jet')               

        # SIC Prediction
        ax_SIC = axs_SIC_classes[1]
        output_class['SIC'] = output_class['SIC'].astype(float)
        output_class['SIC'][cfv_masks['SIC']] = np.nan
        ax_SIC.imshow(output_class['SIC'], vmin=0, vmax=train_options['n_classes']['SIC'] - 2, cmap='jet', interpolation='nearest')
        ax_SIC.set_xticks([])
        ax_SIC.set_yticks([])
        ax_SIC.set_title('Model Prediction')
        chart_cbar(ax=ax_SIC, n_classes=train_options['n_classes']['SIC'], chart=chart, cmap='jet')

        # SIC Ice Chart
        ax_SIC = axs_SIC_classes[3]
        inf_y['SIC'] = inf_y['SIC'].astype(float)
        inf_y['SIC'][cfv_masks['SIC']] = np.nan
        ax_SIC.imshow(inf_y['SIC'], vmin=0, vmax=train_options['n_classes']['SIC'] - 2, cmap='jet', interpolation='nearest')
        ax_SIC.set_xticks([])
        ax_SIC.set_yticks([])
        ax_SIC.set_title('Ground Truth')
        chart_cbar(ax=ax_SIC, n_classes=train_options['n_classes']['SIC'], chart=chart, cmap='jet')

        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.25, hspace=0.15)
        fig_SIC_classes.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Classification-R2.png",
                    format='png', dpi=150, bbox_inches="tight")
        plt.close('all')
        ### ADD NEW CLASSIFICATION FOR SIC METRIC ###

        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png"))

        # Saving results per scene

        
        # Get the scores per scene
        #print("ACTUAL METRIC (R2 RAND)")
        scene_combined_score, scene_scores = compute_metrics(true=inf_y_flat, pred=output_flat,
                                                             charts=train_options['charts'],
                                                             metrics=train_options['chart_metric_individual_scenes'],
                                                             num_classes=train_options['n_classes'])
        ### NEW ###
        scene_accuracy = accuracy_metric(output_preds, inf_y_flat_target)
        ### NEW ###        
        ##TEST USING R2 METRIC ###
        #print("TEST METRIC (R2)")
        #scene_combined_score_TEST, scene_scores_TEST = compute_metrics(true=inf_y_flat, pred=output_flat,
        #                                                     charts=train_options['charts'],
        #                                                     metrics=train_options['chart_metric'],
        #                                                     num_classes=train_options['n_classes'])        
        #scene_water_edge_accuarcy = water_edge_metric(output_tfv_mask, train_options)
        
        # Create table with results and log it into wandb b. 
        # Add all the scores into a list and append it to results per scene. 
        # This list with be the data for the table
        scene_results = [x.item() for x in scene_scores.values()]
        #print("INDIVIDUAL SCENE SCORES")
        #print(scene_results)
        scene_results.insert(0, scene_combined_score.item())
        scene_results.insert(0, scene_name)
        #scene_results.append(scene_water_edge_accuarcy.item())
        ### NEW ###
        scene_results.append(scene_accuracy)
        ### NEW ###
        results_per_scene.append(scene_results)

        # Saving scene results on summary if  mode == 'test'

        if mode == 'test':

            wandb.run.summary[f"{'Test '+scene_name}/Best Combined Score"] = scene_combined_score

            for chart in train_options['charts']:
                wandb.run.summary[f"{'Test '+scene_name}/{chart} {train_options['chart_metric_individual_scenes'][chart]['func'].__name__}"] = scene_scores[chart]
            
            ### NEW ###
            wandb.run.summary[f"{'Test '+scene_name}/SIC Accuracy"] = scene_accuracy
            ### NEW ###
            #wandb.run.summary[f"{'Test '+scene_name}/Water Consistency Accuarcy"] = scene_water_edge_accuarcy

    print('inference done')
    # Create wandb table to store results
    #scenes_results_table = wandb.Table(columns=['Scene', 'Combine Score', 'SIC', 'SOD', 'FLOE', 'Water Consistency Acccuracy'],

    #scenes_results_table = wandb.Table(columns=['Scene', 'Combine Score', 'SIC', 'SOD', 'FLOE'],
    scenes_results_table = wandb.Table(columns=['Scene', 'Combine Score', 'SIC', 'SOD', 'FLOE', "SIC Accuracy"],
                                       data=results_per_scene)
    # Log table into wandb
    wandb.run.log({mode+' results table': scenes_results_table})
    print('done saving result per scene on wandb table')

    # compute combine score   
    combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                             metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])
    
    accuracy = accuracy_metric(preds_SIC_class, target_SIC_class)

    ### BAR CHART ###
    print("BAR CHARTS")
    x_axis_label = ["Sea Ice Concentration [%]", "Stage of Development", "Floe Size"]
    bar_widths = [5, 0.8, 0.8]
    for idx, chart in enumerate(train_options['charts']):
        chart_data = inf_ys_flat[chart]
        n_pixels = 0
        pixels_per_class = np.zeros(train_options['n_classes'][chart] - 1)

        for i in range(0, train_options['n_classes'][chart] - 1):
            n_class = torch.sum(chart_data == i).item()
            pixels_per_class[i] += n_class
            n_pixels += n_class

        title = osp.join(cfg.work_dir,inference_name,inference_name)
        generate_pixels_per_class_bar_graph(chart, pixels_per_class, n_pixels, x_axis_label[idx], bar_widths[idx], title)
    ### BAR CHART ###

    #scene_results2 = [x.item() for x in scores.values()]
    #print("FULL DATASET SCENE SCORES")
    #print(scene_results2)    
    # Release 
    torch.cuda.empty_cache()

    print('done calculating overall results. ')
    # compute water edge metric
    #water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)
    if train_options['compute_classwise_f1score']:
        from functions import compute_classwise_f1score
        classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                     charts=train_options['charts'], num_classes=train_options['n_classes'])
        # Release memory
        torch.cuda.empty_cache()

    print('done computing class wise scores.')
    if train_options['plot_confusion_matrix']:


        for chart in train_options['charts']:
            cm = multiclass_confusion_matrix(
                preds=outputs_flat[chart], target=inf_ys_flat[chart], num_classes=train_options['n_classes'][chart])
            # Release memory
            torch.cuda.empty_cache()
            # Calculate percentages
            cm = cm.cpu().numpy()
            cm_percent = np.round(cm / cm.sum(axis=1)[:, np.newaxis] * 100, 2)
            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_percent, annot=True, cmap='Blues')
            # Customize the plot
            class_names = list(GROUP_NAMES[chart].values())
            class_names = [str(obj) for obj in class_names]
            class_names.append('255')
            tick_marks = np.arange(len(class_names)) + 0.5
            plt.xticks(tick_marks, class_names, rotation=45)
            if chart in ['FLOE', 'SOD']:
                plt.yticks(tick_marks, class_names, rotation=45)
            else:
                plt.yticks(tick_marks, class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('Actual Labels')
            plt.title(chart+" Confusion Matrix "+test_name)
            cbar = ax.collections[0].colorbar
            # cbar.set_ticks([0, .2, .75, 1])
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            mkdir_or_exist(f"{osp.join(cfg.work_dir)}/{test_name}")
            plt.savefig(f"{osp.join(cfg.work_dir)}/{test_name}/{chart}_confusion_matrix.png",
                        format='png', dpi=300, bbox_inches="tight")
            
            # Save figure in wandb. 
            image = wandb.Image(plt)
            wandb.log({chart+" Confusion Matrix "+test_name: image})

            # Create a dataframe
            df_cm = pd.DataFrame(cm_percent
                                 
                                 ).astype("float")

            # Name the columns
            df_cm.columns = class_names

            # Name the rows
            df_cm.index = class_names

            print(df_cm)

            tbl_cm = wandb.Table(data=df_cm)

            # Wandb save into artifact
            artifact_cm = wandb.Artifact('Confusion_Matrix_'+experiment_name+'_'+test_name+'_'+chart, 'confusion_matrix')
            artifact_cm.add(tbl_cm, chart+'_Confusion_Matrix_'+experiment_name+'_'+test_name)
            wandb.log_artifact(artifact_cm)

    print('done ploting confusion matrix')
    wandb.run.summary[f"{test_name}/Best Combined Score"] = combined_score
    print(f"{test_name}/Best Combined Score = {combined_score}")
    for chart in train_options['charts']:
        wandb.run.summary[f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
        print(
            f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__} = {scores[chart]}")
        if train_options['compute_classwise_f1score']:
            wandb.run.summary[f"{test_name}/{chart}: classwise score:"] = classwise_scores[chart]
            print(
                f"{test_name}/{chart}: classwise score: = {classwise_scores[chart]}")
    ### NEW ###
    wandb.run.summary[f"{test_name}/SIC Accuracy"] = accuracy
    print(f"SIC Accuracy: {accuracy:.3f}%")
    ### NEW ###
    #wandb.run.summary[f"{test_name}/Water Consistency Accuarcy"] = water_edge_accuarcy
    #print(
    #    f"{test_name}/Water Consistency Accuarcy = {water_edge_accuarcy}")

    if mode == 'test':
        artifact.add(table, experiment_name+'_test')
    elif mode == 'val':
        artifact.add(table, experiment_name+'_val')
    
    wandb.log_artifact(artifact)
  
    # # - Save upload_package with zlib compression.
    if train_options['save_nc_file']:
        print('Saving upload_package. Compressing data with zlib.')
        compression = dict(zlib=True, complevel=1)
        encoding = {var: compression for var in upload_package.data_vars}
        upload_package.to_netcdf(osp.join(cfg.work_dir, f'{experiment_name}_{test_name}_upload_package.nc'),
                                 # f'{osp.splitext(osp.basename(cfg))[0]}
                                 mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
        print('Testing completed.')
        print("File saved succesfully at", osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
        wandb.save(osp.join(cfg.work_dir, f'{experiment_name}_{test_name}_upload_package.nc'))
