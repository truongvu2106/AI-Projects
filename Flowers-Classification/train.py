# Imports python modules
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time

from get_input_args import get_train_input_args
from fc_model import Classifier, save_checkpoint, load_checkpoint

def create_data_loader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
    return trainloader, validloader, testloader, train_image_datasets

def train(model, optimizer, criterion, trainloader, validloader, device, epochs=10):
    
    model.to(device);

    print_every = 10
    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        print(f"Epoch {epoch+1}/{epochs}..")
        for inputs, labels in trainloader:
            start = time.time()
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print(f"    Batch: {steps}.. "
                      f"    Time per batch: {(time.time() - start):.3f}.. "
                      f"    Training loss: {loss.item():.3f}.. ")
        else:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"    Training loss: {running_loss/len(trainloader):.3f}.. "
                  f"    Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"    Accuracy: {accuracy/len(validloader):.3f}")

            model.train()
    

def main():
    in_arg = get_train_input_args()
    data_dir = in_arg.data_dir
    learning_rate = in_arg.learning_rate
    epochs = in_arg.epochs
    arch = in_arg.arch
    gpu = in_arg.gpu
    file_path = in_arg.save_dir
    hidden_layers = in_arg.hidden_units
    drop_p = 0.2
    checkpoint_path = in_arg.checkpoint
    checkpoint = None

    device = 'cpu'
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if not checkpoint_path:
        if type(hidden_layers) == int:
            hidden_layers = [hidden_layers]
    
        output_size = 102
    
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        else:
            model = models.alexnet(pretrained=True)
            input_size = 9216
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = Classifier(input_size, hidden_layers, output_size, drop_p)

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        model, optimizer, checkpoint = load_checkpoint(checkpoint_path, device)
    
    criterion = nn.NLLLoss()

    trainloader, validloader, testloader, train_image_datasets = create_data_loader(data_dir)

    epochs_trained = 0
    if checkpoint:
        arch = checkpoint['arch']
        learning_rate = checkpoint['lr']
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        drop_p = checkpoint['drop_p']
        epochs_trained = checkpoint['epochs']

    print('Start training...')
    print('  Arch: {}'.format(arch))
    print('  Epochs: {}'.format(epochs))
    print('  Device: {}'.format(device))
    print('  Learning rate: {}'.format(learning_rate))
 
    train(model, optimizer, criterion, trainloader, validloader, device, epochs)

    save_checkpoint(model, optimizer, input_size, output_size, drop_p, train_image_datasets, (epochs + epochs_trained), learning_rate, arch, file_path)

if __name__ == "__main__":
    main()