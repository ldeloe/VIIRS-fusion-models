import torch
from torch import nn
import torch.nn.functional as F
import math


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss


class MSELossFromLogits(nn.Module):
    def __init__(self, chart, ignore_index=-100):
        super(MSELossFromLogits, self).__init__()
        self.ignore_index = ignore_index
        self.chart = chart
        if self.chart == 'SIC':
            self.replace_value = 11
            self.num_classes = 12
        elif self.chart == 'SOD':
            self.replace_value = 6
            self.num_classes = 7
        elif self.chart == 'FLOE':
            self.replace_value = 7
            self.num_classes = 8
        else:
            raise NameError('The chart \'{self.chart} \'is not recognized')

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        # replace ignore index value(for e.g 255) with a number 11. Becuz one hot encode requires
        # continous numbers (you cant one hot encode 255)
        target = torch.where(target == self.ignore_index,
                             torch.tensor(self.replace_value, dtype=target.dtype,
                                          device=target.device), target)
        # do one hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)

        # apply softmax on logits
        softmax = torch.softmax(output, dim=1, dtype=output.dtype)

        criterion = torch.nn.MSELoss(reduction='none')

        # calculate loss between softmax and one hot encoded target
        loss = criterion(softmax, target_one_hot.to(softmax.dtype))

        # drop the last channel since it belongs to ignore index value and should not
        # contribute to the loss

        loss = loss[:, :-1, :, :]
        loss = loss.mean()
        return loss

#class WaterConsistencyLoss(nn.Module):

    #def __init__(self):
    #    super().__init__()
    #    self.keys = ['SIC', 'SOD', 'FLOE']
    #    self.activation = nn.Softmax(dim=1)
    
    #def forward(self, output):
    #    sic = self.activation(output[self.keys[0]])[:, 0, :, :]
    #    sod = self.activation(output[self.keys[1]])[:, 0, :, :]
    #    floe = self.activation(output[self.keys[2]])[:, 0, :, :]
    #    return torch.mean((sic-sod)**2 + (sod-floe)**2 + (floe-sic)**2)

# only applicable to regression outputs



class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum() # / mask.sum() because mean reduction
        return loss

class GaussianNLLLossWithIgnoreIndex3(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', eps=1e-3): # originally 1e-6 but ray used 1e-3
        super(GaussianNLLLossWithIgnoreIndex3, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps  # small value to avoid log(0)

    def forward(self, input, target, var):
        # Remove the last channel of input and var to match target shape
        #var = torch.nan_to_num(var, nan=self.eps)

        # Check validity of reduction mode
        #if reduction != "none" and reduction != "mean" and reduction != "sum":
        #    raise ValueError(reduction + " is not valid")

        # Clamp for stability
        var = var.clone()
        with torch.no_grad():
            input = input.squeeze(-1)
            var = var.squeeze(-1)
            var.clamp_(min=self.eps)

            # Apply mask to input and var
            mask = (target != self.ignore_index).type_as(input)
            diff = input - target
            diff = diff * mask

            log_var = torch.log(var)
            masked_var = var * mask
            masked_log_var = log_var * mask

        # Entries of var must be non-negative
        if torch.any(masked_var < 0):
            raise ValueError("masked var has negative entry/entries")

        # Calculate the loss
        loss = 0.5 * (masked_log_var + diff ** 2 / masked_var)
        #loss = 0.5 * (masked_log_var.sum() + torch.sum(diff ** 2) / masked_var.sum())
        #loss = 0.5 * (masked_log_var.sum() + (diff ** 2 / masked_var).sum())
        #if full:
        #    loss += 0.5 * math.log(2 * math.pi)
        #print("Loss before nan_to_num: ", loss)
        # Replace NaN values with 0
        #loss = torch.nan_to_num(loss, nan=0.0)
        print("Total loss: ", loss.sum() / mask.sum())

        if self.reduction == "mean":
            return loss.sum() / mask.sum()
        #elif reduction == "sum":
        #    return loss.sum()
        #else:
        #    return loss

class GaussianNLLLossWithIgnoreIndex2(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', eps=1e-3): # originally 1e-6 but ray used 1e-3
        super(GaussianNLLLossWithIgnoreIndex2, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps  # small value to avoid log(0)

    def forward(self, input, target, var):

        mask = (target != self.ignore_index).type_as(input)  # Binary mask for valid targets
        #print("Unique values in mask: ", torch.unique(mask))
        diff = input.squeeze(-1) - target
        diff = diff * mask

        var = var.clone()
        var = var.squeeze(-1)
        with torch.no_grad():
            #max_var = torch.clamp(var.squeeze(-1), min=self.eps)
            var.clamp_(min=self.eps)
        
        #var_data = var.squeeze(-1)
        #print("VAR SQUEEZED attempt 2!")
        masked_var = var * mask #.squeeze(-1) 
        #print("Unique values in masked var: ", torch.unique(masked_var))

        ## SEPARATE TO PREVENT LOG(0)
        log_var = torch.log(var) #.squeeze(-1)) 
        masked_log_var = log_var * mask

        if torch.any(masked_var < 0):
            raise ValueError("var after masking has negative entry/entries")
        
        loss = 0.5*(torch.nansum(masked_log_var) + torch.nansum(diff**2)/torch.nansum(masked_var))
        #loss = 0.5 * (masked_log_var + diff**2 / masked_var)

        #term = diff**2 / masked_var
        #if torch.any(torch.isnan(term)):
        #    print("NaN in diff**2 / masked_var")
        #if torch.any(torch.isinf(term)):
        #    print("Inf in diff**2 / masked_var")

        #if torch.any(torch.isnan(loss)):
        #    print("NaN Loss")
            #loss += 0.5 * math.log(2 * math.pi)

        # Reduce loss
        if self.reduction == 'mean':
            loss = loss / mask.sum()
            #loss = torch.nansum(loss) / mask.sum() # omit masked pixels from mean calculation

        #if torch.any(torch.isnan(var.squeeze(-1))):
        #    print("NaN in clamped var")
        #if torch.any(torch.isnan(masked_var)):
        #    print("NaN in masked var")
        #if torch.any(torch.isnan(masked_log_var)):
        #    print("NaN masked log var")
        #denominator = torch.sum(masked_var)
        #if denominator == 0:
        #    print("Masked variance has zero sum.")
        if torch.any(torch.isnan(loss)):
            print("Loss is still somehow NaN after mean")

        return loss

class GaussianNLLLossWithIgnoreIndex(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', eps=1e-3): # originally 1e-6 but ray used 1e-3
        super(GaussianNLLLossWithIgnoreIndex, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps  # small value to avoid log(0)

    def forward(self, input, target, var):

        mask = (target != self.ignore_index).type_as(input)  # Binary mask for valid targets
        diff = input.squeeze(-1) - target
        diff = diff * mask
        max_var = torch.clamp(var.squeeze(-1), min=self.eps) # avoid log(0), divide by 0
        masked_var = max_var * mask
        #print("Unique values in masked var: ", torch.unique(masked_var))

        ## SEPARATE TO PREVENT LOG(0)
        log_var = torch.log(max_var) 
        masked_log_var = log_var * mask

        #loss = 0.5*(torch.sum(masked_log_var) + torch.sum(diff**2)/torch.sum(masked_var)) # this works but element-wise might be wrong
        loss = 0.5*(torch.sum(masked_log_var) + torch.sum(diff**2/masked_var)) # results in nan

        # Reduce loss
        if self.reduction == 'mean':
            loss = loss / mask.sum()

        print("Total loss: ", loss)

        return loss
