from darkflow.net.build import TFNet
import cv2
import json

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