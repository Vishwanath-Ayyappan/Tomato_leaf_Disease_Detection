# Tomato Leaf Disease Detection
## Different Deep Learning models are trained for better results : Using Pytorch

Tomato is the most widely used crop in India. So early detection of the diseases present will help the farmers and agriculturalists in significant ways. The ancient method of finding the diseases is visual observation, but this method is used very rarely, and we are in need of a more accurate and fast method This project briefs the detection of diseases present in a tomato leaf using Convolutional Neural Networks (CNNs) which is a class under deep neural networks. As an initial step, the dataset is segregated prior to the detection of tomato leaf. The concept of transfer learning is used where a pre-trained model (ResNet-50) is imported and adjusted according to our classification problem. To increase the quality of the ResNet model and to enhance the result as close to the actual prevailing disease, data augmentation has been implemented. Taking all these into consideration, a tomato leaf disease detection model has been developed using PyTorch that uses deep - CNNs. Finally, the testing dataset is processed for validation based on the learned parameters from the ResNet 50 model. Six most prevailing diseases in tomato crops have been taken for classification. Data augmentation has been introduced to increase the data set to 4 times the actual data and the model has shown an accuracy of 94.4%. We also applied the VGG 19 model and our own model to the tomato dataset and found that ResNet 50 outperforms the other two models. Finally a feature was added to the ResNet 50 model i.e. if the model predicts a leaf to be diseased then it will direct the user to the website where they can get to know about the description of the disease and cure for it.


![agriengineering-03-00020-g004](https://user-images.githubusercontent.com/85700873/168486556-9e00432d-a7e6-4abc-b435-9659315ec291.jpeg)


## Installation

Install the dependencies 

```sh
pip install torchvision
```
> Note: `torch_version==1.10.0`, `pillow_version =8.3.0`, is used in this project
> 
Different version may give results but with some warnings.

## Dataset
Datasets are required for all processes in the project. A data set with a strength of 9,801 images are collected from the repository of Plant village. Tomato diseases which are used as datasets are Bacterial spot, Early blight, Tomato yellow leaf curl virus, Septoria leaf spot, Tomato mosaic virus. The healthy leaf is also included as one of the classes in the five classes of disease. Datasets are divided into 80% for training and 20% for testing.

Dataset is available in the following drive link

| Dataset | Link |
| ------ | ------ |
| Google Drive | [https://drive.google.com/drive/folders/1ee_VfiKzxOof5oz9eoDPr6VvbWl33qSL?usp=sharing]|

> Note: Store the data in the correct directory or load directly from your drive using the following command in Google colab 
```sh
from google.colab import drive
drive.mount('/content/drive')
```

## GPU Requirements
The project needs GPU for faster training of these models. We used Colab's free GPU
Check whether you have appropriate resources before running the model.
```sh
print(torch.cuda.is_available())
```
## Training

The project has two models, ResNet-50 and VGG-19.
To run the Resnet50 model
```sh
model = models.resnet50(pretrained=True)
```
To run the VGG-19 model

```sh
model = models.vgg19(pretrained=True)
```

#### Hyperparameters

For our project, the following specifications were used. This can be tuned further for different requirements

```sh
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(),lr = 0.001)
```

## Saving the Model

The model parameters and weights can be viewed using the follwoing command

```sh
for param_tensor in model.state_dict():
   print(param_tensor, "\t", model.state_dict()[param_tensor].size())
```

This will display all the layers and weights associated.

For saving the best model, a checkpoint can be created and model weights can be saved for future inference.

```sh
PATH = "90percentResnet"
torch.save(model, PATH)
```

> Note: The result can be different when you train the model.

The saved model can be easily loaded and evaluated

```sh
model_res = torch.load("90percentResnet")
model_res.eval()
```

## Results
the models are tested on the test set and the accuracies are recorded. Also, the training loss and validation loss plot, Confusion matrix, and ROC curve for each of the three models are calculated to depict their performance.

At the first level, the model will classify between a healthy and unhealthy leaf. If first- level classification results in an unhealthy leaf, the second level of classification will predict the type of disease among five diseases. After this, the model will also display details about the disease such as causes and symptoms and its treatments and control.


<img width="478" alt="Screenshot 2022-05-15 at 12 44 12 PM" src="https://user-images.githubusercontent.com/85700873/168488418-4e7e9975-c6e7-4a5f-853f-49e495b70967.png">
<img width="478" alt="Screenshot 2022-05-15 at 12 44 21 PM" src="https://user-images.githubusercontent.com/85700873/168488430-1908a3dd-417b-4a62-a640-19c08fcd473d.png">
<img width="416" alt="Screenshot 2022-05-15 at 12 44 32 PM" src="https://user-images.githubusercontent.com/85700873/168488434-e7d84fa7-baca-4524-8390-9ed668e9ed75.png">

<img width="416" alt="Screenshot 2022-05-15 at 12 49 19 PM" src="https://user-images.githubusercontent.com/85700873/168488443-023ac279-bdba-4c35-b488-ef0e994f90da.png">
<img width="416" alt="Screenshot 2022-05-15 at 12 49 29 PM" src="https://user-images.githubusercontent.com/85700873/168488444-eff389ce-2144-4a9c-97de-179a3f61c957.png">
<img width="416" alt="Screenshot 2022-05-15 at 12 49 37 PM" src="https://user-images.githubusercontent.com/85700873/168488446-07e7cd79-3d14-4a65-8faa-48bda48e4b95.png">


Our Custom Model Output: a diseased leaf (Mosaic virus) is given as an input to the model. From the figure below, we can observe that even though we have given a leaf containing mosaic virus, the model predicted it as Septoria leaf spot. This is an incorrect prediction.
<img width="436" alt="Screenshot 2022-05-15 at 12 56 24 PM" src="https://user-images.githubusercontent.com/85700873/168488452-10249582-b2fa-4022-a1e8-be359c9abdcc.png">


VGG-19 Model Output: A healthy leaf is given as input to the model. From the figure below, we can observe that the model has correctly classified the image as a healthy one.
<img width="320" alt="Screenshot 2022-05-15 at 12 56 14 PM" src="https://user-images.githubusercontent.com/85700873/168488456-fcd02588-cad7-435b-8d69-437605c16b8a.png">


RESNET-50 Model Output: A diseased leaf (Bacterial spot) is given as an input. We can see that the model has classified it as a diseased leaf and also predicted it to which class it belongs to.
<img width="261" alt="Screenshot 2022-05-15 at 12 55 59 PM" src="https://user-images.githubusercontent.com/85700873/168488457-80c9a397-e809-43db-bcb7-206c16adfc5f.png">

#### Observations: 
- it is clearly evident that the RESNET-50 Model Outperforms the other 2 models with ease.
- VGG-19 performance is close to RESNET-50 model and it is way better than the model proposed by us
- So the main reason our model performs like a random model is because of data imbalance, so because of this reason our model is not able to perform that well.
- The Training time taken by RESNET-50 Model is 6 hours, the training time taken by VGG-19 Model is 5.5 hours and the training time taken by our model is 2 hours.
- The state of the art algorithms take more time to train because the architecture is complex compared to our algorithm.


## Acknowledgment
This project is done as part of the curriculum for Introduction to HPML course taught at NYU Tandon, Spring 2022 by Prof. Parijat Dube.
