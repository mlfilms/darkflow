from darkflow.net.build import TFNet
import cv2
import json
import os
import shutil
import sys


def trainFlowCFG(cfg):
    ckptPath = 'ckpt'
    checkpointPath = 'ckpt/checkpoint'
    
    if not os.path.exists(ckptPath):
        os.makedirs(ckptPath)

    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    #dataset is the image folder
    options = {"model": cfg['darkflow']["model"], 
            "load": cfg['darkflow']["starting_weights"],
            "batch": cfg['darkflow']["batch_size"],
            "epoch": cfg['darkflow']["epoch"],
            "gpu": cfg['darkflow']["gpu_usage"],
            "train": True,
            "lr": float(cfg['darkflow']["learning_rate"]),
            "annotation": cfg['darkflow']["training_annotations"],
            "labels": cfg['darkflow']["labels_file"],
            "dataset": cfg['darkflow']["training_images"]}
            



    
    tfnet = TFNet(options)
    tfnet.train()
    tfnet.savepb()


    modelPath = cfg['darkflow']['model']
    path, filename = os.path.split(modelPath)
    name, ext = os.path.splitext(filename)


    pathPB = os.path.join('built_graphs', name+'.pb')
    pathMeta = os.path.join('built_graphs', name+'.meta')
    savePathPB = os.path.join(cfg['temp']['rootDir'], cfg['meta']['runName'],[cfg['meta']['runName']+'.pb'])
    savePathMeta = os.path.join(cfg['temp']['rootDir'], cfg['meta']['runName'],[cfg['meta']['runName']+'.meta'])
    shutil.copyfile(pathPB,savePathPB)
    shutil.copyfile(pathMeta,savePathMeta)





if __name__ == "__main__":
    print("Failure")
    print(os.getcwd())
    '''print(os.getcwd())
    with open('config.json') as json_config:
        config = json.load(json_config)
    #dataset is the image folder
    options = {"model": config["model"], 
            "load": config["starting_weights"],
            "batch": config["batch_size"],
            "epoch": config["epoch"],
            "gpu": config["gpu_usage"],
            "train": True,
            "lr": config["learning_rate"],
            "annotation": config["training_annotations"],
            "labels": config["labels_file"],
            "dataset": config["training_images"]}
            
    tfnet = TFNet(options)
    tfnet.train()
    tfnet.savepb()
'''