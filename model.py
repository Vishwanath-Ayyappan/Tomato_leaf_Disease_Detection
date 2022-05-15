# To use the VGG19 model, comment out the ResNet50 model and use the VGG19 model. 
model = models.resnet50(pretrained=True)
# model = models.vgg19(pretrained=True)
model

for params in model.parameters():
    params.requires_grad = False

classifier  = nn.Sequential(nn.Linear(2048,1024),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(1024,512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512,256),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(128,6),
                           nn.LogSoftmax(dim = 1))
model.fc = classifier
