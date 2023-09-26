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
### How to easy use IoU_v5 ###
    from IoU_v5 import *
    ground_truth = read_json('path for ground truth .json')
    predict = read_json('path for predict .json')
Example for ground_truth .json

    {"image_name":[[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]}
Example for predict .json

    {"image_name":[[xmin, ymin, xmax, ymax, conf_score], [xmin, ymin, xmax, ymax, conf_score]]}

