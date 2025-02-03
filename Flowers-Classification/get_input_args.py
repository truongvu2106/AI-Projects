import argparse

def get_train_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('--save_dir', default='checkpoint.pth', help='Checkpoint saving path')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--hidden_units', default=512, type=int, nargs='+', help='Hidden units')
    parser.add_argument('--epochs', default=5, type=int, help='Epochs')
    parser.add_argument('--gpu', action='store_true', help='Check using gpu if it is available')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint file')

    return parser.parse_args()

def get_predict_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Predict model.')
    parser.add_argument('input', help='Path to image')
    parser.add_argument('checkpoint', default='checkpoint.pth', help='Path to checkpoint file')
    parser.add_argument('--top_k', default=3, type=int, help='Top K')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Category names')
    parser.add_argument('--gpu', action='store_true', help='Check using gpu if it is available')

    return parser.parse_args()