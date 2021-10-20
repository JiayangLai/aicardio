from torchvision import models
import net_lib.modifiedmodels as modifiedmodels

from net_lib.classifation_end import CLS_end



def isintorchvision(netname_in):
    return netname_in in ['resnet18','resnet50']

def isinmodifiedmodels(netname_in):
    return netname_in in ['resnet18_2332']

def model_builder_classification(net_name, num_classes):
    if isintorchvision(net_name):
        model_out = eval("models." + net_name)(pretrained=False)
    elif isinmodifiedmodels(net_name):
        model_out = eval("modifiedmodels." + net_name)(pretrained=False)
    model_out = CLS_end(model_out, num_classes)
    return model_out