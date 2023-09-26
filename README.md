# Med_RetinaNet
## Model can use in RetinaNet
 - ResNet18
 - ResNet34
 - ResNet50
 - ResNet101
 - ResNet152

 - ResNeXt101
 - ResNeXt101_32x8d
 - SE_ResNeXt101

 - ResNeSt50
 - ResNeSt101
 - ResNeSt200
 - ResNeSt269

### Training
The network can be trained using the train.py script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

    python train.py --dataset coco --coco_path ../coco --depth 50
For training using a custom dataset, with annotations in CSV format (see below), use

    python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>

### How to easy use IoU_v5 ###
    from IoU_v5 import *
    ground_truth = read_json('path for ground truth .json')
    predict = read_json('path for predict .json')
Example for ground_truth .json

    {"image_name":[[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]}
Example for predict .json

    {"image_name":[[xmin, ymin, xmax, ymax, conf_score], [xmin, ymin, xmax, ymax, conf_score]]}

