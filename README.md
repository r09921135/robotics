# CSIE5047 Robotics: Speech Control Care Robot
We propose a robotic system called Speech Control Care Robot, where the robot arm can be controlled by human speech commands to help with some daily activities, more specifically, grabbing things and feeding food that human describe.

## Execution
To execute the entire sustem, run the following command:
    
    python send_script.py

## BERT for Object Description Extraction
### Training
To train the BERT model for extracting object description, first modify the training data at `bert/train.csv` for your own purpose. Then run the following command:
    
    python bert/train.py

The trained BERT model will be saved at `bert/saved_model/`.

### Inferencing
To inference the trained model, run the following command:
    
    python bert/inference.py

## Action Classification
### Training
The backbone BERT model used in this step is the same as the previous one. To train the action classifier, first modify the training data at `bert/classifier_train.csv` for your own purpose. Then run the following command:
    
    python bert/classifier_train.py

The trained classifier will be saved as `bert/Action_Classifier.ckpt`.

### Inferencing
To inference the trained classifier, run the following command:
    
    python bert/classfier_inference.py

## Refering Image Segmentation (RIS)
The code in this part is modify from [refvos](https://github.com/miriambellver/refvos).
### Fine-tuning
To fine-tuning the RIS model, first download the [RefVOS Pre-trained weights](https://github.com/miriambellver/refvos). Next, the fine-tuning data should be arranged as the following:
```
refvos
└── images
    └── finetune
        ├── image
        ├── mask
        └── text
```
To start fine-tuning, run the following bash code:

    bash refvos/finetune.sh

The fine-tuned RIS model will be saved as `refvos/checkpoints/model_best_my_model.pth`.

### Inferencing
To inference the trained RIS model, run the following basg code:
    
    bash refvos/inference.sh
    
The inferencing results will be saved at `refvos/results`.

## Main
To inference the above mentioned models at once, run the following command:
    
    python main.py

    
    
