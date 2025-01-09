import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib
import warnings
import numpy as np
import torch
from tqdm import tqdm  # Progress bar
import wandb
from mmcv import Config, mkdir_or_exist

from functions import compute_metrics, save_best_model, load_model, class_decider, \
create_train_validation_and_test_scene_list, get_scheduler, get_optimizer, get_loss, get_model, accuracy_metric, classify_SIC_tensor

from early_stopping import EarlyStopping

# Note: may need in future
#\ slide_inference, batched_slide_inference

from loaders import get_variable_options, TrainValDataset, TestDataset

from utils import colour_str
from test_function_uncertainty import test

def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None,
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args

def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0):
    '''
    Trains the model.

    '''
    best_combined_score = -np.Inf  # Best weighted model score.

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart])
                         for chart in train_options['charts']}

    early_stopping = EarlyStopping(patience=15) ### EARLY STOPPING
    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        cross_entropy_loss_sum = torch.tensor([0.])  # To sum the training cross entropy batch losses during the epoch.
        # To sum the training edge consistency batch losses during the epoch.
        #edge_consistency_loss_sum = torch.tensor([0.])

        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        # To sum the validation cross entropy batch losses during the epoch.
        val_cross_entropy_loss_sum = torch.tensor([0.])
        # To sum the validation cedge consistency batch losses during the epoch.
        #val_edge_consistency_loss_sum = torch.tensor([0.])
        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            train_loss_batch = torch.tensor([0.]).to(device)  # Reset from previous batch.
            #edge_consistency_loss = torch.tensor([0.]).to(device)
            cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
                # - Forward pass.
                output = net(batch_x)
                # - Calculate loss.
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    if train_options['uncertainty'] != 0 and chart == 'SIC':

                        # COULD REMOVE UNSQUEEZE IF REMOVE SQUEEZE FROM FUNCTION
                        sic_mean = output[chart]['mean']  # Mean of SIC
                        sic_variance = output[chart]['variance']  # Variance of SIC
                        loss = loss_ce_functions[chart](sic_mean.to(device), batch_y[chart].to(device), sic_variance.to(device)) 
                        mask = (batch_y[chart] != 255).type_as(sic_mean)
                        masked_loss = loss * mask
                        # Reduce the masked loss (e.g., mean over valid elements)
                        final_loss = masked_loss.sum() / mask.sum() #masked_loss.nansum() / mask.nansum() # was just .sum()
                        #if np.isnan(final_loss):
                        #    final_loss = 0.0
                        cross_entropy_loss += weight * final_loss
                        #cross_entropy_loss += weight * loss_ce_functions[chart](
                        #    sic_mean.unsqueeze(-1).to(device), batch_y[chart].to(device), sic_variance.unsqueeze(-1).to(device)) 
                        #cross_entropy_loss += weight * loss_ce_functions[chart](
                        #    output[chart][..., 0].unsqueeze(-1).to(device), batch_y[chart].to(device), output[chart][..., 1].unsqueeze(-1).to(device))      #added to device                   
                    else:
                        cross_entropy_loss += weight * loss_ce_functions[chart](
                            output[chart], batch_y[chart].to(device))

            # Note: removed water edge loss; this conditional is a renment and could be removed in future
            #if train_options['edge_consistency_loss'] != 0:
            #    a = train_options['edge_consistency_loss']
            #    train_loss_batch = cross_entropy_loss 
            #else:
            #    train_loss_batch = cross_entropy_loss
            train_loss_batch = cross_entropy_loss

            ###if not torch.isnan(train_loss_batch): #freezes learning rate
            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            train_loss_batch.backward()

            # - Prevent exploding gradient with GaussianNLLLoss
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=train_options['max_norm']) #max_norm=1.0) 

            # - Optimizer step
            optimizer.step()

            # - Scheduler step
            scheduler.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()
            #edge_consistency_loss_sum += edge_consistency_loss.detach().item()

        # - Average loss for displaying
        train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, i + 1).detach().item()
        #edge_consistency_epoch = torch.true_divide(edge_consistency_loss_sum, i + 1).detach().item()

        # -- Validation Loop -- #
        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}

        ### NEW ###
        preds_SIC_class = torch.Tensor().to(device)
        target_SIC_class = torch.Tensor().to(device)
        ### NEW ###

        # Outputs mask by train fill values
        outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
        net.eval()  # Set network to evaluation mode.
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            val_loss_batch = torch.tensor([0.]).to(device)
            #val_edge_consistency_loss = torch.tensor([0.]).to(device)
            val_cross_entropy_loss = torch.tensor([0.]).to(device)

            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                if train_options['model_selection'] == 'swin':
                    output = slide_inference(inf_x, net, train_options, 'val')
                else:
                    output = net(inf_x)

                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    if train_options['uncertainty'] != 0 and chart == 'SIC':
                        sic_mean = output[chart]['mean']  # Mean of SIC
                        sic_variance = output[chart]['variance']  # Variance of SIC
                        loss = loss_ce_functions[chart](sic_mean.to(device), inf_y[chart].unsqueeze(0).long().to(device), sic_variance.to(device))
                        mask = (inf_y[chart].unsqueeze(0).long() != 255).type_as(sic_mean)
                        masked_loss = loss * mask
                        # Reduce the masked loss (e.g., mean over valid elements)
                        final_loss = masked_loss.sum() / mask.sum()
                        val_cross_entropy_loss += weight * final_loss
                        #val_cross_entropy_loss += weight * loss_ce_functions[chart](
                        #    output[chart][..., 0].unsqueeze(-1).to(device), inf_y[chart].unsqueeze(0).long().to(device), output[chart][..., 1].unsqueeze(-1).to(device))                        
                    else:
                        val_cross_entropy_loss += weight * loss_ce_functions[chart](output[chart],
                                                                                inf_y[chart].unsqueeze(0).long().to(device))

                # Note: again, a remnent from removing the water loss
                #if train_options['edge_consistency_loss'] != 0:
                #    a = train_options['edge_consistency_loss']

            val_loss_batch = val_cross_entropy_loss

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                #print("CHART: ", chart)
                #output[chart] = class_decider(output[chart], train_options, chart)
                if train_options['uncertainty'] != 0 and chart == 'SIC':
                    #print("UNIQUE BEFORE CLASS DECIDER")
                    #print(torch.unique(output[chart]))

                    sic_mean = output[chart]['mean'].unsqueeze(-1)
                    output[chart] = class_decider(sic_mean.to(device), train_options, chart)
                    #output[chart] = class_decider(output[chart][..., 0].unsqueeze(-1).to(device), train_options, chart)
                    #print(output[chart].size())
                    #print("UNIQUE CLASS DECIDER VALS")
                    #print(torch.unique(output[chart]))
                else:
                    #print("UNIQUE OUTPUT")
                    #print(torch.unique(output[chart]))
                    output[chart] = class_decider(output[chart], train_options, chart)
                    #print(output[chart].size())

                outputs_flat[chart] = torch.cat((outputs_flat[chart], output[chart][~cfv_masks[chart]]))
                #print(outputs_flat[chart].size())

                outputs_tfv_mask[chart] = torch.cat((outputs_tfv_mask[chart], output[chart][~tfv_mask]))
                inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y[chart]
                                                [~cfv_masks[chart]].to(device, non_blocking=True)))
                #print(inf_ys_flat[chart].size())
            
            #print(outputs_flat.size())
            #print(inf_ys_flat.size())
            ### NEW ###
            #output_preds = classify_SIC_tensor(output['SIC'][~cfv_masks['SIC']])
            preds_SIC_class = classify_SIC_tensor(outputs_flat['SIC']) #torch.cat((preds_SIC_class, output_preds))#[~cfv_masks['SIC']])) 
            #inf_ys_flat_target = classify_SIC_tensor(inf_ys_flat['SIC'][~cfv_masks['SIC']])
            target_SIC_class = classify_SIC_tensor(inf_ys_flat['SIC']) #torch.cat((target_SIC_class, inf_ys_flat_target)) #[~cfv_masks['SIC']]))
            ### NEW ###  

            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()
            val_cross_entropy_loss_sum += val_cross_entropy_loss.detach().item()
            #val_edge_consistency_loss_sum += val_edge_consistency_loss.detach().item()

        # - Average loss for displaying
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()
        #val_edge_consistency_epoch = torch.true_divide(val_edge_consistency_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        print('Computing Metrics on Val dataset')
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                 metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

        #water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)

        if train_options['compute_classwise_f1score']:
            from functions import compute_classwise_f1score
            # dictionary key = chart, value = tensor; e.g  key = SOD, value = tensor([0, 0.2, 0.4, 0.2, 0.1])
            classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'], num_classes=train_options['n_classes'])
        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options['charts']:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            # if classwise_f1score is True,
            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])
        
        ### NEW ###
        accuracy = accuracy_metric(preds_SIC_class, target_SIC_class)
        wandb.log({f"SIC Accuracy": accuracy}, step=epoch)
        print(f"SIC Accuracy: {accuracy:.3f}%")
        ### NEW ###

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Learning Rate": optimizer.param_groups[0]["lr"]}, step=epoch)

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.

        ### IMPLEMENTATION OF EARLY STOPPING ###
        if combined_score > best_combined_score:
            best_combined_score = combined_score

            # Log the best combine score, and the metrics that make that best combine score in summary in wandb.
            wandb.run.summary[f"While training/Best Combined Score"] = best_combined_score
            for chart in train_options['charts']:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, scheduler, epoch)

            wandb.save(model_path)
        
        #if early_stopping(val_loss_epoch, net):
        if early_stopping(val_loss_epoch):
            break
        ### IMPLEMENTATION OF EARLY STOPPING ###

    del inf_ys_flat, outputs_flat  # Free memory.
    return model_path

def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = TrainValDataset(ai4arctic_files=train_options['train_list'], viirs_files = train_options['train_list_viirs'], options=train_options, do_transform=True)

    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'], pin_memory=True)

    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    dataset_val = TestDataset(options=train_options, ai4arctic_files=train_options['validate_list'], viirs_files = train_options['validate_list_viirs'], mode='train')

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

    return dataloader_train, dataloader_val

def main():
    print("Entered main")
    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)

    
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    # Set the seed if not -1
    if train_options['seed'] != -1 and args.seed == None:
        # set seed for everything
        if args.seed != None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run_name)

    ic(cfg.work_dir)
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
  
    # ### CUDA / GPU Setup
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        
        # Check if NVIDIA V100, A100, or H100 is available for torch compile speed up
        if train_options['compile_model']:
            gpu_ok = False
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                gpu_ok = True
            
            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        # Setup device to be used
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    net = get_model(train_options, device)

    if train_options['compile_model']:
        net = torch.compile(net)

    optimizer = get_optimizer(train_options, net)

    scheduler = get_scheduler(train_options, optimizer)

    if args.resume_from is not None:
        print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
        epoch_start = load_model(net, args.resume_from, optimizer, scheduler)
    elif args.finetune_from is not None:
        print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
        _ = load_model(net, args.finetune_from)

    # initialize wandb run
    print("start initializing wandb run")
    if not train_options['cross_val_run']:
        wandb.init(name=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="sic-data-fusion", config=train_options, id=id, resume="allow")
    else:
        wandb.init(name=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="sic-data-fusion", config=train_options, id=id, resume="allow")

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Validation Epoch Loss", summary="none")
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Combined score", summary="none")
    wandb.define_metric("SIC r2_metric", summary="none")
    wandb.define_metric("SOD f1_metric", summary="none")
    wandb.define_metric("FLOE f1_metric", summary="none")
    wandb.define_metric("Learning Rate", summary="none")
    wandb.define_metric("SIC Accuracy", summary="none")

    wandb.save(str(args.config))
    print(colour_str('Save Config File', 'green'))

    
    create_train_validation_and_test_scene_list(train_options)

    dataloader_train, dataloader_val = create_dataloaders(train_options)

    # Update Config
    wandb.config['validate_list'] = train_options['validate_list']


    print('Data setup complete.')


    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)


    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')



    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    # this is for valset 1 visualization along with gt
    test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], train_options['validate_list_viirs'], 'CrossValidation')


    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')




    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    # this is for test path along with gt after the gt has been released
    test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], train_options['test_list_viirs'], 'Test')

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')


    # finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
