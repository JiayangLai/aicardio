import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.saved_path_gen import savedpathgen


net_name = 'resnext50_32x4d'
pretrained = True
model_saved_path = savedpathgen(pretrained, net_name)
mod_name = 'TL0.0001226_TA100.000_VL0.01551_VA99.6_EP59'
model_now = torch.load('../models_saved/' + model_saved_path + '/' + mod_name + '.pt')
model_now = model_now.to('cpu')
data_path = 'infer_pic'

image_transforms = {
    'valid': transforms.Compose([
        transforms.Resize(size=(256)),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
    ])
}
dataset = {
    'valid': datasets.ImageFolder(root=data_path, transform=image_transforms['valid'])
}

valid_data = DataLoader(dataset['valid'], batch_size=1, shuffle=True)


classes = {0:'ch2',1:'ch3',2:'ch4',3:'lap',4:'salv'}

with torch.no_grad():
    model_now.eval()

    for j, (inputs, labels) in enumerate(valid_data):


        outputs = model_now(inputs)
        print(outputs)
        ret, predictions = torch.max(outputs.data, dim=1)
        print(classes[predictions.numpy()[0]])