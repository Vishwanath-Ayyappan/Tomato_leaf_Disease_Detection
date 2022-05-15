model_resnet.eval()
classes= ['bacterial_spots', 'early_blight', 'healthy', 'septoria_leaf_spot', 'tomato_Yellow_Leaf_Curl_Virus', 'tomato_mosaic_virus']
test_image_path = "septoria_virus.jpg"
im = Image.open(test_image_path)
tens = test_transforms(im)
tens = tens.unsqueeze(0)
inp = Variable(tens)
inp = inp.to("cpu")
pred = model_res(inp)
index = pred.data.cpu().numpy().argmax()
img=mpimg.imread(test_image_path)
imgplot = plt.imshow(img)
plt.show()
if index == 2:
    print("The leaf is Healthy")
else:
    print("The leaf is diseased")
    print("Disease is {}".format(classes[index]))
    dis = classes[index]
    dis = re.sub('tomato_',  '',    dis)
    dis = re.sub('_',  '-',    dis)
    webbrowser.open("https://gardenerspath.com/how-to/disease-and-pests/common-tomato-diseases/"+dis)
# print("Predicted image is {}".format(classes[index]))
