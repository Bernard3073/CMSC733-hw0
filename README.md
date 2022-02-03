# Phase 1  
Navigate to the "Code" folder in the "Phase 1" folder  
And run
```
python3 Wrapper.py
```
# Phase 2
Place the "CIFAR10" folder is in the "Phase 2" folder  
Check if there are two folders named "Test_graph" and "Train_graph" and make sure each of them has four subfolders with the name of the neural network.  
Inside the "Code" folder in the "Phase 2" folder   
## Train the network  
```
python3 Train.py --NetworkType ResNet
```
`--BasePath`: BasePath path to load images from the "CIFAR10" folder  
`--CheckPointPath`: path to save Checkpoints  
`--NumEpochs`: number of Epochs to train for, default = 30  
`--DivTrain`: factor to reduce train data by per epoch, default = 1  
`--MiniBatchSize`: size of the MiniBatch to use, default = 16  
`--LoadCheckPoint`: load model from latest Checkpoint from CheckPointsPath, enter "1" to trigger, default = 0  
`--LogsPath`: path to save Logs for Tensorboard  
`--NetworkType`: my_NN, ResNet, ResNeXt, DenseNet, default = my_NN  
## Test the network  
```
python3 Test.py --NetworkType ResNet 
```
`--NetworkType`: my_NN, ResNet, ResNeXt, DenseNet, default = my_NN  
`--ModelPath`: path to load latest model from the "Checkpoints" folder  
`--BasePath`: path to load images from  
`--LabelsPath`: path to load the label file, default=./TxtFiles/LabelsTest.txt  
`--NumEpochs`: number of epochs to train for, default=30   