import numpy as np
import argparse
import torch
from torchvision.ops import nms
import os 
# from dataloader import dataloader
from PIL import ImageDraw, ImageFont, Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from util import *


def get_args_parser(known=False):
    
    parser = argparse.ArgumentParser('FRCNN detect and validation process ')

    # validation setting 
    # parser.add_argument('--batchsize', default=1, type=int, help='batch size of validation data')

    # model, backbone setting
    parser.add_argument('--weights', default='train_ALL_VOC2007', help='fine-tuned weights filename. train_ALL_VOC2007.cpu.pt bu default')
    parser.add_argument('--path', default='./model', help='place for storing the fine-tuned weights') 
    parser.add_argument('--exp', default='1', type=int, help='experiment number', required=True)     
    parser.add_argument('--device', default='cpu', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  
    parser.add_argument('--plot_result', default=False, type=bool, help='plot the detection results.')  

    # used data path
    parser.add_argument('--dataset_name', default='VOC2007', help='training data format. currently only support VOC2007')    
    parser.add_argument('--label', default='./dataset/VOCdevkit/VOC2007/Annotations/', help='data path')                  
    parser.add_argument('--data', default='./dataset/VOCdevkit/VOC2007/JPEGImages/', help='label path')  
    parser.add_argument('--sample', default=10, type=int, help='sampling the dataset')
    parser.add_argument('--eval_output', default='./eval', type=str, help='path for output evaluation results')
    parser.add_argument('--detect_output', default='./detect', type=str, help='path for output detection results')
    
    
    # others
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    # opt = parser.parse_known_args()[0]
    return opt




def evaluate(val_dataloader, model, plot_result=False, device='cpu'):

    # create a list for calculating mAP
    pred        = [None]*val_dataloader.dataset.__len__()
    target_gt   = [None]*val_dataloader.dataset.__len__()

    for i, (image, target, image_id) in enumerate(val_dataloader.dataset): 
        # image, target, image_id = next(iter(val_dataloader.dataset))

        # move the image arrays to the specified device one by one
        # images = list(img.to(device) for img in images)
        image = image.to(device)
        outputs = model([image])
        # print(outputs)

        # turns image array into PIL Image module
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))

        # prediction 
        boxes   = outputs[0]["boxes"].data.cpu().numpy()
        scores  = outputs[0]["scores"].data.cpu().numpy()
        labels  = outputs[0]["labels"].data.cpu().numpy()

        # using threshold 0.5 to filter out the boxes 
        conf = 0.5
        boxes   = boxes[scores >= conf].astype(np.int32)
        labels  = labels[scores >= conf]
        scores  = scores[scores >= conf]

        # apply nms on the prediction
        indices_keep = nms(torch.tensor(boxes.astype(float)),  
                torch.tensor(scores.astype(float)), iou_threshold=0.5).cpu().numpy()

        # ground truth
        boxes_gt    = target["boxes"].data.cpu().numpy()
        labels_gt   = target["labels"].data.cpu().numpy()
        image_gt    = image.copy()

        print('now processing image: {}'.format(image_id))
        
        pred[i]   = dict(
                    boxes=torch.tensor(boxes[indices_keep].astype(float)), 
                    scores=torch.tensor(scores[indices_keep].astype(float)), 
                    labels=torch.tensor(labels[indices_keep].astype(float))
                    )
        target_gt[i]  = dict(
                    boxes=torch.tensor(boxes_gt.astype(float)),
                    labels=torch.tensor(labels_gt.astype(float))
                    )

        if plot_result : 
            # plot the predictions
            plot_results(image, boxes[indices_keep], labels[indices_keep], image_id, 
                scores=scores[indices_keep], gt=False, exp_number=exp_number)

            # plot the ground truth
            plot_results(image_gt, boxes_gt, labels_gt, image_id, 
                gt=True, exp_number=exp_number)


    metric  = MeanAveragePrecision()
    metric.update(pred, target_gt)
    mAP = metric.compute()['map']
    print('mAP over {} images: {:4.3f}'.format(val_dataloader.dataset.__len__(), mAP) )
            
    return metric, mAP


def output_result_dict(metric, args): 
    result_metric = metric.compute().copy()

    path = args.path
    exp_number = args.exp
    
    if os.path.exists('{}/exp{:0>2}/var.dat'.format(path, exp_number)): 
        with open('{}/exp{:0>2}/var.dat'.format(path, exp_number), 'r') as f : 
            for line in f: 
                item, value = line.split(":")
                result_metric[item] = value.rstrip()


    output_path = '{}/exp{:0>2}/'.format(args.eval_output, exp_number)
    if not os.path.isdir(output_path): 
        os.mkdir(output_path)

    np.savez(output_path+'result_metric.npz', result_metric=result_metric)
        
    return result_metric 

def out_result_dat(exp_number):
    
    fname = 'eval/exp{:0>2}/result_metric.npz'.format(exp_number)
    result_metric = np.load(fname, allow_pickle=True)['result_metric'].item()

    with open('eval/exp{:0>2}/result_metric.dat'.format(exp_number), 'w') as f:
        for name in result_metric.keys():
            if name[0:3] == 'map': 
                value = float(result_metric[name])*100
                f.write('{}: {:3.2f}\n'.format(name,value))
            elif name[0:3] == 'mar': 
                pass
            else : 
                f.write('{}: {}\n'.format(name, result_metric[name]))
 
    return


def load_data_val(annotation_path, img_path, batch_size_val=1, sample=10):
    """ (1)データの準備  """
    # datasetの読み込み
    xml_dir     = annotation_path  #"dataset/VOCdevkit/VOC2007/Annotations/"
    image_dir   = img_path #"dataset/VOCdevkit/VOC2007/JPEGImages/"

    # debug 
    if len(glob(xml_dir)) == 0 or len(glob(image_dir)) == 0 : 
        print('cannot load the file from the specified folder. please confirm the data again. ')

    print('loading data from {}'.format(image_dir))

    # データをロード
    # batch_size_train = 5
    # batch_size_val = 1
    _,val_dataloader= dataloader(
        xml_dir,
        image_dir,
        batch_size_train=1, 
        batch_size_val=batch_size_val, 
        sample=sample)
    # train_batches = train_dataloader.__len__()
    # val_batches = val_dataloader.__len__()
    return val_dataloader 

if __name__ == '__main__': 

    
    """ fetch the areguments """
    args = get_args_parser()


    """ load training dataloader """
    # get args
    data_path   = args.data
    label_path  = args.label
    sample      = args.sample

    # load validation data
    val_dataloader = load_data_val(label_path, data_path, sample=sample)  

    
    """ load the fine-tuned model"""
    # get args
    device      = args.device           # cpu or cuda
    weights     = args.weights          # e.g., train_ALL_VOC2007.cpu.pt
    exp_number  = args.exp              # e.g., 29
    path        = args.path             # e.g., ./model
    plot_result = args.plot_result
    
    model = load_model_eval(exp_number, weights=weights, device=device)


    """ evaluate the mAP """
    print('loading {}/exp{:0>2}/{}'.format(path,exp_number,weights))
    metric, mAP = evaluate(val_dataloader, model, plot_result=plot_result, device=device)


    """ output the evaluation result """
    output_result_dict(metric, args)
    out_result_dat(exp_number)
    
