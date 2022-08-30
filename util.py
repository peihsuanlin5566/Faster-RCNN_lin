import numpy as np 
from glob import glob
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET 
from PIL import ImageDraw, ImageFont, Image
import os

 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def model_1():
    """
    we will be using Mask R-CNN, which is based on top of Faster R-CNN. 
    Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.
    
    Finetuning from a pretrained model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    # num_classes=len(dataset_class)+1
    num_classes = 21
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def find_class(xml_dir): 
    """ fetch the class names on the dataset"""
    class_name = []

    xml_files = glob(xml_dir+'/*.xml')
    file_num = np.array([np.int64(x[-10:-4]) for x in xml_files ])
    xml_files = np.array(xml_files)[np.argsort(file_num)]
    for xml_file in xml_files: 
        xml = ET.parse(xml_file)
        for i in xml.iter('object'): 
            obj_name = i.find('name').text 
            if obj_name not in class_name: 
                class_name.append(obj_name)

    return class_name



class xml2list(object):
    """ 
    return: [width, height, xmin, ymin, xamx, ymax, label_idx] 
    
    """
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        xml = ET.parse(xml_path).getroot()
        
        for size in xml.iter("size"):     
            width = float(size.find("width").text)
            height = float(size.find("height").text)
                
        for obj in xml.iter("object"):
            # なんかdifficultなかったから消す
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            #     continue          
            bndbox = [width, height]        
            name = obj.find("name").text.lower().strip() 
            bbox = obj.find("bndbox")            
            pts = ["xmin", "ymin", "xmax", "ymax"]     
            for pt in pts:        
                cur_pixel =  np.float32(bbox.find(pt).text)
                ###########エラー対策 
                #cur_pixel =  int(bbox.find(pt).text)               
                bndbox.append(cur_pixel)           
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)    
            ret += [bndbox]
            
        return np.array(ret) 

class MyDataset(torch.utils.data.Dataset):
        
        def __init__(self, df, image_dir):
            """ create a dataset for training and validation 
                input: 
                    df: dataframe containing the annotation information ([width, height, xmin, ymin, xamx, ymax, label_idx] )
                    image_dir: folder for placing the images
            """
            
            super().__init__()
            
            self.image_ids = df["image_id"].unique()
            self.df = df
            self.image_dir = image_dir
            
        def __getitem__(self, index):
    
            transform = transforms.Compose([
                                    transforms.ToTensor()
            ])
    
            # load the images under the specified folder
            image_id = self.image_ids[index]
            image = Image.open(f"{self.image_dir}/{image_id}.jpg")
            image = transform(image)
            
            # load the annotations
            records = self.df[self.df["image_id"] == image_id]
            boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
            # boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.int32)
            area    = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area    = torch.as_tensor(area, dtype=torch.float32)
            labels  = torch.tensor(records["class"].values, dtype=torch.int64)
            iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"]= labels
            target["image_id"] = torch.tensor([index])
            target["area"] = area
            target["iscrowd"] = iscrowd
            
            return image, target, image_id
        
        def __len__(self):
            return self.image_ids.shape[0]



def dataloader (xml_dir, image_dir, batch_size_train=1, batch_size_val=None, sample=10):
    
    """ generate two data, split it into two datasets, create batches
        input: 
            xml_dir: path for placing annotations  
            image_dir: path for placing images 
            
        output: 
            train_dataloader 
            val_dataloader
            
    """

    # read the annotations 
    classes = find_class(xml_dir) 
    transform_anno = xml2list(classes)

    # output the category information
    category = {}
    category[0] = 'background'
    for i, class_name in enumerate(classes):
        category[i+1] = class_name
    
    with open('category_VOC2007.dat', 'w') as f:
        for key in category.keys():
            f.write('{}  {}\n'.format(key, category[key]))
 
    i = 0
    df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
    for path in  glob(xml_dir+'/*.xml'):
        image_id = path.split("/")[-1].split(".")[0]
        bboxs = transform_anno(path)
        for bbox in bboxs:
            tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            # df = df.append(tmp, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([tmp])], ignore_index=True)


    # the amount of classes + 1 (background)
    df["class"] = df["class"] + 1

    # sort the rows acctoding to image id 
    df = df.sort_values(by="image_id", ascending=True)
    
    # create torch dataset 
    dataset  = MyDataset(df, image_dir)

    # subdataset
    drawing_indices = list(range(0, len(dataset), sample))
    subset = torch.utils.data.Subset(dataset, drawing_indices)

    # split tht dataset   
    torch.manual_seed(2020)
    n_train = int(len(subset) * 0.7)
    n_val = len(subset) - n_train 
    train, val = torch.utils.data.random_split(subset, [n_train, n_val])

    def collate_fn(batch):
        return tuple(zip(*batch))
        
    
    train_dataloader  = torch.utils.data.DataLoader(train, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    if batch_size_val != None: 
        val_dataloader    = torch.utils.data.DataLoader(val, batch_size=batch_size_val, shuffle=True, collate_fn=collate_fn)
    else: 
        val_dataloader    = torch.utils.data.DataLoader(val, shuffle=True, collate_fn=collate_fn)


    return train_dataloader, val_dataloader

def load_model_eval(exp_number, weights='train_ALL_VOC2007.cpu.pt', device='cpu'): 
    # model_filename = './model/exp{:0>2}/{}.{}.pt'.format(exp_number,weights,device)
    model_filename = glob('./model/exp{:0>2}/*.pt'.format(exp_number))[0]
    if torch.cuda.is_available():  
        model = torch.load(model_filename)
    else: 
        model = torch.load(model_filename, map_location=torch.device('cpu'))


    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    device = torch.device(device)    
    model.to(device)

    # evaluation mode
    return model.eval()

def plot_results(image, boxes, labels, image_id, exp_number=None, scores=None, gt=False): 
    # plot the prediction result
    # fetch the category
    if exp_number == None: 
        print('since no exp number is input, images would be output to detect/test/')

    category = {}
    with open('category_VOC2007.dat', 'r') as f: 
        for line in f: 
            label_num, key = line.split()
            category[int(label_num)] = key


    if np.size(labels) == 1: 
        label = labels[0]
        box = boxes[0]
        draw = ImageDraw.Draw(image)
        label_name = category[label]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        # ラベルの表示
        fnt = ImageFont.truetype("arial.ttf", 10)#40
        if not gt: 
            label_text = "{} {:3.2f}".format(label_name,scores[0] )
        else: 
            label_text = "{}".format(label_name)

        text_w, text_h = fnt.getsize(label_text)
        draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
        draw.text((box[0], box[1]),label_text , font=fnt, fill='white')

    else: 
        for i, (label, box) in enumerate(zip(labels, boxes)):
            draw = ImageDraw.Draw(image)
            label_name = category[label]
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

            # ラベルの表示
            fnt = ImageFont.truetype("arial.ttf", 10)#40
            if not gt: 
                label_text = "{} {:3.2f}".format(label_name,scores[i] )
            else: 
                label_text = "{}".format(label_name)

            text_w, text_h = fnt.getsize(label_text)
            draw.rectangle([box[0], box[1], box[0]+text_w, box[1]+text_h], fill="red")
            draw.text((box[0], box[1]),label_text , font=fnt, fill='white')
            

    if exp_number != None: 
        output_path = 'detect/exp{:0>2}'.format(exp_number)
    else:     
        output_path = 'detect/test'

    if not os.path.isdir(output_path): 
        os.mkdir(output_path)
        

    if not gt: 
        image_name = "{}/detection_id{}.png".format(output_path,image_id)
    else: 
        image_name = "{}/detection_id{}_gt.png".format(output_path,image_id)
    
    image.save(image_name)
    print(image_name)

    return 