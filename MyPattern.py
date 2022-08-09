from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear, Binarize
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np

## prepare the train data and the test data
# train_size = 70; test_size = 30       # total size: train: 70x5 = 350; test: 30x5 = 150
# f_train = open('TrainData.txt',mode='w')
# f_test = open('TestData.txt',mode='w')
# for letter in ['Q','S','J','T','U']:
#     if letter == 'Q':
#         label = '1'
#     else:
#         label = '0'
#     for i in range(1,train_size+1):
#         f_train.write(letter+str(i)+'.bmp '+label+'\n')
#     for ii in range(1,test_size+1):
#         f_test.write(letter+str(train_size+ii)+'.bmp '+label+'\n')
# f_train.close()
# f_test.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


#路径是自己电脑里所对应的路径
datapath = r'F:\AIR\BNNCIS\SpiceSim\MyTrainData'
trainpath = r'TrainData.txt'
testpath = r'TestData.txt'

class MyDataset(Dataset):
    def __init__(self,txtpath):
        #创建一个list用来储存图片和标签信息
        imgs = []
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs
    #返回数据集大小
    def __len__(self):
        return len(self.imgs)
    #打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'\\'+pic)
        pic = transforms.ToTensor()(pic)
        return pic,label

#实例化对象
train_data = MyDataset(trainpath)
test_data = MyDataset(testpath)
#将数据集导入DataLoader，进行shuffle以及选取batch_size
train_loader = DataLoader(train_data,batch_size=70,shuffle=True,num_workers=0)
test_loader = DataLoader(test_data,batch_size=70,shuffle=True,num_workers=0)
row = 10; column = 10                   # size of the input image
num_channel = 8                         # channel number: 8
outsize = 2                             # size of the final output: 2 classes, trigger or not
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BinarizeLinear(row*column, num_channel,bias=False)
        self.htanh1 = nn.Hardtanh()
        # self.bn1 = nn.BatchNorm1d(num_channel)
        self.fc2 = BinarizeLinear(num_channel, outsize,bias=False,act=True)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, row*column)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        # return self.logsoftmax(x)
        return x

model = Net()
lr = 0.01 ; epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
acc_list = []       # 用于画 Accuracy-Epoch图

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.as_tensor(tuple(map(int,target)))    # 将target转为tensor类型
        data, target = Variable(torch.as_tensor(data)), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
        #                100. * (batch_idx+1) / len(train_loader), loss.item(),))
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                    100. * (batch_idx+1) / len(train_loader), loss.item(),))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = torch.as_tensor(tuple(map(int,target)))
            data, target = Variable(torch.as_tensor(data)), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc_list.append(100. * correct / len(test_loader.dataset))

for epoch in range(1, epochs + 1):
    train(epoch)
    # test()
    if epoch%40==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
torch.save(model.state_dict(),'Pattern.pt')
test()
# plt.plot(range(1,epochs+1),acc_list)
# plt.title('Validation Accuracy')
# plt.show()

################ 手动推理
# mymodel = torch.load('Pattern.pt')
# f1_weight = Binarize(mymodel['fc1.weight'])
# f2_weight = Binarize(mymodel['fc  2.weight'])
# for data, target in test_loader:
#     target = torch.as_tensor(tuple(map(int,target)))
#     out1 = torch

