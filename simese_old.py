
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import cv2
import torch.nn.functional as F
import numpy as np
import os
import random
import sys
from torchvision import models

#workingdir = 'D:\\thesis_working\\poredata_cropped_scaled_224x244'
workingdir = 'D:\\thesis_working\\Mat_Cropped_Imgs\\scaled'

weightname="weight\\weight.pt"
tempna='./'
name=weightname
test_only=0
#else:
#    weightname=sys.argv[2]
#    tempna='./'
#    name=tempna+weightname
#    test_only=0
N=10
class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 img_transform1,
                 img_transform2,
                 study
                 ):
        #load 100 first images
        self.size = 20
        self.study = study
        poreimgs = os.listdir(img_path+'\\pore')
        nonporeimgs = os.listdir(img_path+'\\non-pore')
        
        random.shuffle(poreimgs)
        random.shuffle(nonporeimgs)
        if study=='train':
            poreimgs = poreimgs[0:self.size]
            nonporeimgs = nonporeimgs[0:self.size]

        if study=='test':
            train_size = self.size
            self.size = 500
            poreimgs = poreimgs[train_size:self.size+train_size]
            nonporeimgs = nonporeimgs[train_size:self.size+train_size] 
                 
        self.poreimgs_list = [
                os.path.join(img_path+'\\pore\\', i) for i in poreimgs
            ]
        self.nonporeimgs_list = [
                os.path.join(img_path+'\\non-pore\\', i) for i in nonporeimgs
            ]
        shuffle1 = np.arange(self.size);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size);np.random.shuffle(shuffle2)
        self.labels1 = np.concatenate((np.ones(int(self.size/2)),np.zeros(int(self.size/2))),axis=0)[shuffle1]
        self.labels2 = np.concatenate((np.ones(int(self.size/2)),np.zeros(int(self.size/2))),axis=0)[shuffle2]
        
        self.img1_list = np.concatenate((self.poreimgs_list[0:int(self.size/2)],self.nonporeimgs_list[0:int(self.size/2)]),axis=0)[shuffle1]
        self.img2_list = np.concatenate((self.poreimgs_list[int(self.size/2):],self.nonporeimgs_list[int(self.size/2):]),axis=0)[shuffle2]
        #compute logical XNOR
        self.label_list = np.logical_and(np.logical_or(self.labels1,np.logical_not(self.labels2)),np.logical_or(self.labels2,np.logical_not(self.labels1)))
        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
        #self.imgs_class1 = img_class1
        #self.imgs_class2 = img_class2

    def __getitem__(self, index):
        if self.study=='train':
            self.shuffle()

        img1_path = self.img1_list[index]
        img2_path = self.img2_list[index]
        label = self.label_list[index]
        label=int(label)
        rand1 = False
        rand2 = False
        # add noise during training
        if (random.random()>0.5 and self.study=='train'):
            img1 = np.random.rand(224,224,3)*255
            rand1 =True
            label =0
        else:
            img1 = cv2.imread(img1_path)
        if (random.random()>0.99 and self.study=='train'):
            img2 = np.random.rand(224,224,3)*255
            rand2 = True
            label =0
        else:
            img2 = cv2.imread(img2_path)

        if rand1 and rand2:
            label = 1
            print(rand1,rand2)
        img1 = img1.astype(np.float)/255
        img2 = img2.astype(np.float)/255
        #img1 = cv2.resize(img1,(224,224), interpolation = cv2.INTER_AREA)
        #img2 = cv2.resize(img2,(224,224), interpolation = cv2.INTER_AREA)
        img1 = self.img_transform1(img1)
        img2 = self.img_transform2(img2)
            
        #else:
        #    img2 = np.random.rand(224,224,3).astype(np.float)
        #    label = int(0)

        return img1,img2,label
    def __len__(self):
        return len(self.label_list)
    def shuffle(self):
        shuffle1 = np.arange(self.size);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size);np.random.shuffle(shuffle2)
        self.labels1 = np.concatenate((np.ones(int(self.size/2)),np.zeros(int(self.size/2))),axis=0)[shuffle1]
        self.labels2 = np.concatenate((np.ones(int(self.size/2)),np.zeros(int(self.size/2))),axis=0)[shuffle2]
        self.img1_list = np.concatenate((self.poreimgs_list[0:int(self.size/2)],self.nonporeimgs_list[0:int(self.size/2)]),axis=0)[shuffle1]
        self.img2_list = np.concatenate((self.poreimgs_list[int(self.size/2):],self.nonporeimgs_list[int(self.size/2):]),axis=0)[shuffle2]
        self.label_list = np.logical_and(np.logical_or(self.labels1,np.logical_not(self.labels2)),np.logical_or(self.labels2,np.logical_not(self.labels1)))
    


def load_images_to_memory(img_path):
        imgs_class1 = []
        imgs_class2 = []
        poreimgs = os.listdir(img_path+'\\pore')
        nonporeimgs = os.listdir(img_path+'\\non-pore')
        poreimgs_list = [
            os.path.join(img_path+'\\pore\\', i) for i in poreimgs
            ]
        nonporeimgs_list = [
            os.path.join(img_path+'\\non-pore\\', i) for i in nonporeimgs
            ]
        for i in poreimgs_list:
            imgs_class1.append(cv2.imread(i))
        for i in nonporeimgs_list:
            imgs_class2.append(cv2.imread(i))
        return imgs_class1,imgs_class2

class Rescale(object):
    def __call__(self, img):
        if random.random()<0.0:
            f = round(0.1*random.randint(7, 13),2)
            if f>1:
                img = cv2.resize(img,None,fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
                a = int(round((f*224-224)/2))
                img = img[a:a+224,a:a+224]
            else:
                img = cv2.resize(img,None,fx=f, fy=f, interpolation = cv2.INTER_AREA)
                a= int(round((224-f*224)/2))
                temp=np.zeros([224,224,3],dtype=np.uint8)
                temp.fill(0) 
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        temp[i+a,j+a]=img[i,j]
                img=temp
        return img

class Flip(object):
    def __call__(self,img):
        if random.random()<0.5:
            return cv2.flip(img,1)
        return img
        
class Rotate(object):
    def __call__(self,img):
        if random.random()<0.5:
            angle=random.random()*60-30
            rows,cols,cn = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            return img
        return img

class Translate(object):
    def __call__(self,img):
        if random.random()<0.5:
            x=random.random()*20-10
            y=random.random()*20-10
            rows,cols,cn = img.shape
            M= np.float32([[1,0,x],[0,1,y]])
            img = cv2.warpAffine(img,M,(cols,rows))
        return img
            

resnet18 = models.resnet18(pretrained=True)
my_model = nn.Sequential(*list(resnet18.children())[:-2])
my_model = my_model.cuda()

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 0),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 7, 1, 0),
            nn.ReLU(),
            #nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 7, 1, 0),
            nn.ReLU(),
            #nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            )
        self.conv4 =nn.Sequential(
            nn.Conv2d(256, 512, 7, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )
        self.fc = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
        )
  
    def forward(self, x):
        #print(x.shape)
        x = x.view(-1,3, 224,224)
        #print(x.shape)
        x = my_model(x)
        #print(x.shape)
        #print(x.shape)
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        x = self.fc2(x)
        #print(x.shape)
        return x


if __name__ == '__main__': 
    transform1 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    transform2 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    

    #set1,set2 = load_images_to_memory(workingdir)            
    train_set = custom_dset(workingdir, transform1,transform2,'train')
    train_loader = DataLoader(train_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)
    test_set = custom_dset(workingdir,transform1,transform2,'test')
    test_loader = DataLoader(test_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)  
    lr = 1e-5
    num_epoches = 1000

    net=Cnn()         
    if torch.cuda.is_available() :
        net = net.cuda()  
        
    optimizer = torch.optim.Adam(net.parameters(), lr)
    feature_encoder_scheduler = StepLR(optimizer,step_size=10000,gamma=0.5)
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
    
        def forward(self, output1, output2, label):
            euclidean_distance = F.pairwise_distance(output1, output2)
            loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
    
            return loss_contrastive
        
    
    loss_func = ContrastiveLoss() 
    l_his=[]
    acc_hist = []
    if test_only==0:
        acc = 0
        for epoch in range(num_epoches):
            print('Epoch:', epoch + 1, 'Training...')
            running_loss = 0.0 
            for i,data in enumerate(train_loader, 0):
                image1s,image2s,labels=data
                if torch.cuda.is_available():
                    image1s = image1s.cuda()
                    image2s = image2s.cuda()
                    labels = labels.cuda()
                image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
                optimizer.zero_grad()
                f1=net(image1s.float())
                f2=net(image2s.float())
                loss = loss_func(f1,f2,labels)
                loss.backward()
                optimizer.step()
                #print(i)
                if i % 6 == 5:
                    l_his.append(loss.cpu().detach().numpy())
                # print statistics
                running_loss += loss
                if i % 6==5:    
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 6))
                    running_loss = 0.0
            
            
            correct = 0
            total = 0

            for datat in test_loader:
                image1st,image2st,labelst = datat
                if torch.cuda.is_available():
                    image1st = image1st.cuda()
                    image2st = image2st.cuda()
                    labelst = labelst.cuda()

                f1=net(image1st.float())
                f2=net(image2st.float())
                dist = F.pairwise_distance(f1, f2)
                dist = dist.cpu()
                for j in range(dist.size()[0]):
                    if ((dist.data.numpy()[j]<0.7)):
                        if labelst.cpu().data.numpy()[j]==1:
                            correct +=1
                    else:
                        if labelst.cpu().data.numpy()[j]==0:
                            correct+=1
                    total+=1     
                #print(correct)
            #print(correct,total)
            #print(dist)
            #print(labels.cpu())
            curr_acc = 100.0 * correct / total           
            print('Accuracy of the network on the test images: %0.2f %%' % (
                curr_acc))
            if curr_acc > acc:
                torch.save(net.state_dict(), name)
                acc = curr_acc
            acc_hist.append(curr_acc)
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(acc_hist)    
            plt.xlabel('Steps')  
            plt.ylabel('Acc')  
            fig.savefig('plott_acc.png') 
            plt.close()

        print('Finished Training')
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(l_his)    
        plt.xlabel('Steps')  
        plt.ylabel('Loss')  
        fig.savefig('plott2.png')  
        torch.save(net.state_dict(), 'weight\\weight_final.pt')
    else:   
        net.load_state_dict(torch.load(name))
        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        test_set = custom_dset(workingdir,transform,transform,'train')
        test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)   
        correct = 0
        total = 0
        for data in test_loader:
            image1s,image2s,labels = data
            if torch.cuda.is_available():
                image1s = image1s.cuda()
                image2s = image2s.cuda()
                labels = labels.cuda()
            image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())   
            f1=net(image1s.float())
            f2=net(image2s.float())
            dist = F.pairwise_distance(f1, f2)
            dist = dist.cpu()
            for j in range(dist.size()[0]):
                if ((dist.data.numpy()[j]<0.5)):
                    if labels.cpu().data.numpy()[j]==1:
                        correct +=1
                        total+=1
                    else:
                        total+=1
                else:
                    if labels.cpu().data.numpy()[j]==0:
                        correct+=1
                        total+=1
                    else:
                        total+=1                
        print('Accuracy of the network on the train images: %d %%' % (
            100 * correct / total))
        
        test_set = custom_dset(workingdir,transform,transform,'test')
        test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)  
        correct = 0
        total = 0
        for data in test_loader:
            image1s,image2s,labels = data
            if torch.cuda.is_available():
                image1s = image1s.cuda()
                image2s = image2s.cuda()
                labels = labels.cuda()
            image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())   
            f1=net(image1s.float())
            f2=net(image2s.float())
            dist = F.pairwise_distance(f1, f2)
            dist = dist.cpu()
            for j in range(dist.size()[0]):
                if ((dist.data.numpy()[j]<0.8)):
                    if labels.cpu().data.numpy()[j]==1:
                        correct +=1
                        total+=1
                    else:
                        total+=1
                else:
                    if labels.cpu().data.numpy()[j]==0:
                        correct+=1
                        total+=1
                    else:
                        total+=1                
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total)) 