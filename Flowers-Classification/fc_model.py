import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            hidden_layers: list of integers, the sizes of the hidden layers
            output_size: integer, size of the output layer
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
    
def save_checkpoint(model, optimizer, input_size, output_size, drop_p, train_datasets, epochs, lr, arch, file_path):
    print('Start saving in {}'.format(file_path))
    model.to('cpu')
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'drop_p': drop_p,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'lr': lr,
                  'arch': arch}
    torch.save(checkpoint, file_path)
    print('Model saved.')
    return checkpoint

def load_checkpoint(file_path, device='cpu'):
    checkpoint = torch.load(file_path, map_location=device)
    arch = checkpoint['arch']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)

    classifier = Classifier(input_size = checkpoint['input_size'],
                            hidden_layers = checkpoint['hidden_layers'],
                            output_size = checkpoint['output_size'],
                            drop_p = checkpoint['drop_p'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device);
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Model loaded.')
    return model, optimizer, checkpoint

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    image_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    with Image.open(image_path) as im:
        im_resize = im.resize((256, 256))
        center = 256 / 2
        im_crop = im_resize.crop((center - 112 , center - 112, center + 112, center + 112))
        im_processed = image_transforms(im_crop)
    return im_processed