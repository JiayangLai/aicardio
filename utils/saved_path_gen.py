
def savedpathgen(pretrained,net_name):
    if pretrained:
        model_saved_path = net_name+'_pretrained'
    else:
        model_saved_path = net_name
    return model_saved_path