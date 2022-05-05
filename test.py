import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
from utils import select_device, natural_keys, gazeto3d, angular
from nets.PerimetryNet import PerimetryNet


cfg_re50 = {
    'name'              : 'Resnet50',
    'return_layers'     : {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel'        : 256,
    'out_channel'       : 64    
}
backbone = "resnet50"


cfg = cfg_re50

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='/MPIIFaceGaze_phi/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='/MPIIFaceGaze_phi/Label', type=str)
    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='eyediap, mpiigaze',
        default= "mpiigaze", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=256, type=int)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

  
    if data_set=="mpiigaze":
        model_used =  PerimetryNet(cfg=cfg, pretrained=False)

        for fold in range(0,15):
            folder = os.listdir(args.gazeMpiilabel_dir)
            folder.sort()
            testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] 
            gaze_dataset=datasets.Mpiigaze(testlabelpathombined,args.gazeMpiimage_dir, transformations, False, 180, fold)

            test_loader = torch.utils.data.DataLoader(
                dataset=gaze_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            
            
            if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):
                os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))

            # list all epochs for testing
            folder = os.listdir(os.path.join(snapshot_path,"fold"+str(fold)))
            folder.sort(key=natural_keys)
            
            softmax = nn.Softmax(dim=1)
            with open(os.path.join(evalpath, os.path.join("fold"+str(fold), data_set+".log")), 'w') as outfile:
                configuration = f"\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}, fold={fold}---------------------------------------\n"
                print(configuration)
                outfile.write(configuration)
                epoch_list=[]
                avg_MAE=[]
                avg_class_MAE = []
                avg_all_MAE=[]
                for epochs in folder: 
                    model=model_used
                    saved_state_dict = torch.load(os.path.join(snapshot_path+"/fold"+str(fold),epochs))# epochs
                    model= nn.DataParallel(model,device_ids=[0])
                    model.load_state_dict(saved_state_dict)  # ['model']
                    model.cuda(gpu)
                    model.eval()
                    total = 0


                    avg_error = .0

                    

                    with torch.no_grad():
                        for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                            images = Variable(images).cuda(gpu)
                            total += cont_labels.size(0)

                            label_yaw = cont_labels[:, 0].float() * np.pi / 180  
                            label_pitch = cont_labels[:, 1].float() * np.pi / 180

                            yaw_class, pitch_class, gaze_regre = model(images)

                            yaw_predicted = (gaze_regre[:, 0]).cpu()
                            pitch_predicted = (gaze_regre[:, 1]).cpu()
                            yaw_predicted =yaw_predicted * np.pi / 180  
                            pitch_predicted = pitch_predicted * np.pi / 180

                            for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                                avg_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))
                               

                    x = ''.join(filter(lambda i: i.isdigit(), epochs))
                    epoch_list.append(x)
                    avg_MAE.append(avg_error/ total)
                    loger = f"[{epochs}---{args.dataset}] Total Num:{total},MAE:{avg_error/total} \n"
                    outfile.write(loger)
                    print(loger)
