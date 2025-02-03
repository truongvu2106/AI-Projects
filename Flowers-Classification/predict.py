# Imports python modules
import torch
import numpy as np
import json
from PIL import Image
from get_input_args import get_predict_input_args
from fc_model import Classifier, load_checkpoint, process_image

def sanity_checking(probs, classes_idx, cat_to_name):
    class_names = [cat_to_name[i] for i in classes_idx]
    probs = probs.cpu().data.numpy();
    flower_name = class_names[np.argmax(probs)]
    print('Top Class names: {}'.format(class_names))
    print('Top Probability: {}'.format(probs))
    print('Classify flower: {}'.format(flower_name))
    
def predict(image_path, model, topk=3, device='cpu'):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    img_processed = process_image(image_path)
    image = img_processed.unsqueeze(0)

    classes = []
    idx_to_class = {y: x for x, y in model.class_to_idx.items()}

    model.to(device)
    model.eval()
    image = image.to(device)
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model(image)

    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    for i in top_class[0]:
        classes.append(idx_to_class[i.item()])
    
    return top_p[0], classes
    

def main():
    in_arg = get_predict_input_args()
    image_path = in_arg.input
    checkpoint_path = in_arg.checkpoint
    gpu = in_arg.gpu
    top_k = in_arg.top_k
    category_names = in_arg.category_names

    device = 'cpu'
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model, optimizer, checkpoint = load_checkpoint(checkpoint_path, device)

    probs, classes_idx = predict(image_path, model, top_k, device)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    sanity_checking(probs, classes_idx, cat_to_name)

if __name__ == "__main__":
    main()