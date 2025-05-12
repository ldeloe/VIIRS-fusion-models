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

    # Stores the flat ouputs of only one scene.
    output_flat = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}
    # Stores the flat outputs of all scene.
    outputs_flat = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}
    # Stores the flat ground truth of only one scene. 
    inf_y_flat = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}
    # Stores the flat ground truth of all scenes
    inf_ys_flat = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}
    # Outputs mask by train fill values for only one scene
    output_tfv_mask = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}
    # Outputs mask by train fill values fo all scenes
    outputs_tfv_mask = {chart: torch.Tensor().to("cpu") for chart in train_options['charts']}

    ### NEW ###
    preds_SIC_class = torch.Tensor().to("cpu")
    target_SIC_class = torch.Tensor().to("cpu")
    output_preds = torch.Tensor().to("cpu")
    inf_y_flat_target = torch.Tensor().to("cpu")
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

            inf_x = inf_x.to("cpu")

            # Up sample the masks
            tfv_mask = torch.nn.functional.interpolate(tfv_mask.type(torch.uint8).unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            for chart in train_options['charts']:
                masks_int = cfv_masks[chart].to(torch.uint8)
                masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
                    0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
                cfv_masks[chart] = torch.gt(masks_int, 0)
                cfv_masks[chart] = cfv_masks[chart].to("cpu")

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
                    output[chart] = output[chart].to("cpu")
                    
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
                    inf_y[chart] = torch.nn.functional.interpolate(inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
                                                                   size=original_size, mode='nearest').squeeze()
                    inf_y[chart] = inf_y[chart].to("cpu")  

                sic_output_var = sic_output_var.squeeze(0).squeeze(-1).cpu().numpy().astype(float)
                                                             

        output_class = {}
        for chart in train_options['charts']:
            ### why is class decider used here?
            output_class[chart] = class_decider(output[chart].to(torch.float32), train_options, chart).detach()
            if train_options['save_nc_file']:
                upload_package[f"{scene_name}_{chart}"] = xr.DataArray(name=f"{scene_name}_{chart}",
                                                                       data=output_class[chart].squeeze().cpu().numpy().astype('uint8'),
                                                                       dims=(f"{scene_name}_{chart}_dim0", f"{scene_name}_{chart}_dim1"))
                if chart == 'SIC':
                    sic_output_var[cfv_masks['SIC']] = np.nan
                    upload_package[f"{scene_name}_VAR"] = xr.DataArray(name=f"{scene_name}_VAR", data=sic_output_var, dims=(f"{scene_name}_VAR_dim0", f"{scene_name}_VAR_dim1"))
                    upload_package[f"{scene_name}_STD_DEV"] = xr.DataArray(name=f"{scene_name}_STD_DEV", data=np.sqrt(sic_output_var), dims=(f"{scene_name}_STD_DEV_dim0", f"{scene_name}_STD_DEV_dim1"))

            output_flat[chart] = output_class[chart][~cfv_masks[chart]] 
            outputs_flat[chart] = torch.cat((outputs_flat[chart], output_flat[chart]))
            output_tfv_mask[chart] = output_class[chart][~tfv_mask].to(device)
            outputs_tfv_mask[chart] = torch.cat((outputs_tfv_mask[chart], outputs_tfv_mask[chart]))
            inf_y_flat[chart] = inf_y[chart][~cfv_masks[chart]].to("cpu").float()
            inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y_flat[chart]))
        
        ### SIC Accuracy ###
        output_preds = classify_SIC_tensor(output_flat['SIC'])
        preds_SIC_class = torch.cat((preds_SIC_class, output_preds)) 
        inf_y_flat_target = classify_SIC_tensor(inf_y_flat['SIC'])
        target_SIC_class = torch.cat((target_SIC_class, inf_y_flat_target))
        ### SIC Accuracy ###

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
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

        for idx, chart in enumerate(train_options['charts']):

            ax = axs[idx+6]
            inf_y[chart] = inf_y[chart].astype(float)
            inf_y[chart][cfv_masks[chart]] = np.nan
            ax.imshow(inf_y[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')

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
            ax_SIC.set_xticks([])
            ax_SIC.set_yticks([])
            ax_SIC.imshow(img_SIC, cmap='gray')

        for idx, chart in enumerate(train_options['charts']):

            ax_SIC = axs_SIC[idx+2]
            output_class[chart] = output_class[chart].astype(float)

            output_class[chart][cfv_masks[chart]] = np.nan

            ax_SIC.imshow(output_class[chart], vmin=0, vmax=train_options['n_classes']
                      [chart] - 2, cmap='jet', interpolation='nearest')
            ax_SIC.set_xticks([])
            ax_SIC.set_yticks([])
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

            chart_cbar(ax=ax_SIC, n_classes=train_options['n_classes'][chart], chart=chart, cmap='jet')
            break

        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.5, hspace=-0)
        fig_SIC.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Plot.png",
                    format='png', dpi=300, bbox_inches="tight")
                    #format='png', dpi=128, bbox_inches="tight")
        plt.close('all')

        ### SIC VARIANCE ###
        #sic_output_var[cfv_masks['SIC']] = np.nan

        fig_var, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.imshow(sic_output_var)
        plt.imshow(np.sqrt(sic_output_var))
        plt.title(scene_name)
        cbar = plt.colorbar()
        #cbar.set_label(label= "Variance [%$^2$]", fontsize=12)
        cbar.set_label(label= "Standard Deviation [%]", fontsize=12)
        #fig_var.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Variance.png",
        fig_var.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}-SIC-Std-Dev.png",
            format='png', dpi=150, bbox_inches="tight")
        plt.close('all')

        ### ADD NEW CLASSIFICATION FOR SIC METRIC ###
        fig_SIC_classes, axs2d_SIC_classes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10)) # (5, 14)
        axs_SIC_classes = axs2d_SIC_classes.flat
        ## SIC CLASSIFICATION ##
        n_classes = 4
        ax_SIC = axs_SIC_classes[0]
        converted_SIC = classify_from_SIC(output_class['SIC'])
        converted_SIC = converted_SIC.astype(float)
        converted_SIC[cfv_masks['SIC']] = np.nan

        ax_SIC.imshow(converted_SIC, vmin=0, vmax=n_classes - 2, cmap='jet', interpolation='nearest')
        ax_SIC.set_xticks([])
        ax_SIC.set_yticks([])
        ax_SIC.set_title('Model Prediction') 
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
        scene_combined_score, scene_scores = compute_metrics(true=inf_y_flat, pred=output_flat,
                                                             charts=train_options['charts'],
                                                             metrics=train_options['chart_metric_individual_scenes'],
                                                             num_classes=train_options['n_classes'])

        scene_accuracy = accuracy_metric(output_preds, inf_y_flat_target)

        
        # Create table with results and log it into wandb b. 
        # Add all the scores into a list and append it to results per scene. 
        # This list with be the data for the table
        scene_results = [x.item() for x in scene_scores.values()]

        scene_results.insert(0, scene_combined_score.item())
        scene_results.insert(0, scene_name)

        scene_results.append(scene_accuracy)
      
        results_per_scene.append(scene_results)

        # Saving scene results on summary if  mode == 'test'

        if mode == 'test':

            wandb.run.summary[f"{'Test '+scene_name}/Best Combined Score"] = scene_combined_score

            for chart in train_options['charts']:
                wandb.run.summary[f"{'Test '+scene_name}/{chart} {train_options['chart_metric_individual_scenes'][chart]['func'].__name__}"] = scene_scores[chart]
            
            wandb.run.summary[f"{'Test '+scene_name}/SIC Accuracy"] = scene_accuracy

    print('inference done')
    # Create wandb table to store results
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
    # Release 

    del inf_x, output
    torch.cuda.empty_cache()

    print('done calculating overall results. ')

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
    
    wandb.run.summary[f"{test_name}/SIC Accuracy"] = accuracy
    print(f"SIC Accuracy: {accuracy:.3f}%")

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
                                 mode='w', format='netcdf4', engine='h5netcdf', encoding=encoding)
        print('Testing completed.')
        print("File saved succesfully at", osp.join(cfg.work_dir, f'{experiment_name}_upload_package.nc'))
        wandb.save(osp.join(cfg.work_dir, f'{experiment_name}_{test_name}_upload_package.nc'))
