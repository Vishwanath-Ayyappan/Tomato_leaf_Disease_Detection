criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(),lr = 0.001)
model.to('cpu')

epoch = 100
train_lossesAD, test_lossesAD = [], []
for e in range(epoch):
    running_loss = 0
    for images, labels in trainloader:
#         print("here")
#         images = images.view(images.shape[0], -1)
        start = time.time()
        images, labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
#         print("here2")
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
#                 images = images.view(images.shape[0], -1)
                images, labels = images.to(device),labels.to(device)
                pred = model.forward(images)
                test_loss += criterion(pred,labels)
        
        
                ps = torch.exp(pred)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()       
        train_lossesAD.append(running_loss/len(trainloader))
        test_lossesAD.append(test_loss/len(testloader))        
        print("Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              f"Time per batch: {(time.time() - start)/3:.3f} seconds"
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
