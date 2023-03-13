import argparse
import torch
import torchvision 
import json
import dataloaders
import models
from collections import OrderedDict
import os
from glob import glob
import tqdm
import PIL

def pth_to_onnx():
    pass


def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K']
    if dataset_type == 'CityScapes': 
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    model_name = config['arch']['type']
    print("model {} load successfully: {}\n".format(model_name ,model))
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    print("the model after load the state_dict: {}".format(model))
    model.to(device)
    model.eval()
    
    

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    print(image_files)
    with torch.no_grad():
        # tbar = tqdm.tqdm(image_files, ncols=100)   #进度条
        # for img_file in tbar:
        img_file = image_files[0]
        image = PIL.Image.open(img_file).convert('RGB')
        input = normalize(to_tensor(image)).unsqueeze(0)
        print("input type: {}, input_size: {}\n".format(type(input), input.size()))
        
        # if args.mode == 'multiscale':
        #     prediction = multi_scale_predict(model, input, scales, num_classes, device)
        # elif args.mode == 'sliding':
        #     prediction = sliding_predict(model, input, num_classes)
        # else:
        #     prediction = model(input.to(device))
        #     prediction = prediction.squeeze(0).cpu().numpy()
        # prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        # save_images(image, prediction, args.output, img_file, palette)
        # input to the model
        output = model(input.to(device))
        # torch.nn.DataParallel is not supported by ONNX exporter, please use 'attribute' module 
        # to unwrap model from torch.nn.DataParallel. Try torch.onnx.export(model.module, ...)
        # torch.onnx.export(model.module, input.to(device), "{}.onnx".format(model_name), export_params=True, opset_version=11, \
        #                   do_constant_folding=True, input_names=['input'], output_names=['output'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()   
