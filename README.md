This is a simple realization of Pose-based CNN Features for Action Recognition (**P-CNN**) algorithm. You could get detailed information from their project [website](https://www.di.ens.fr/willow/research/p-cnn/). 

Different from the original algorithm, I use the **Pose** estimated by [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation). The **Optical Flow** features are computed using [FlowNet](https://github.com/NVIDIA/flownet2-pytorch). And I tested it on **HMDB51** and **UCF101** dataset obtaining **27.19%** and **51.23%** accuracy respectively.

### Usage
1. Download the pre-trained CNN weights from the **P-CNN** project [website](https://www.di.ens.fr/willow/research/p-cnn/). 
2. Run the PCNN.py to extract the feature vectors for classification.
3. Run the linearSVM.py to classify actions by a linear SVM.

Note, you should do some minor changes to accomadate this demo to your own coding environment. I pre-extract the **Flow** and **Pose** features for each frame so I could use a Dataloader to input the data to the networks efficiently. You should change the data folder as well as some data input.

Feel free to contact me if you have any question.
