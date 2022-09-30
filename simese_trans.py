
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
import sys, argparse
from torchvision import models

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-d","--directory",type = str, default = 'D:\\thesis_working\\Mat_Cropped_Imgs\\scaled_few_shot')
parser.add_argument("-c","--class_name",type = str, default = 'new')
parser.add_argument("-n","--run_name",type = str, default = 'new_10')
parser.add_argument("-l","--load_weight_name",type = str, default = "weight_a.pt")
parser.add_argument("-t","--test_only",type = int, default = 0)
parser.add_argument("-m","--model_only",type = int, default = 0)
parser.add_argument("-ts","--train_size",type = int, default= 0)
parser.add_argument("-sh","--shot_size", type = int, default = 10)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-e","--epoch",type=int, default=1000)

args = parser.parse_args()

#workingdir = 'D:\\thesis_working\\poredata_cropped_scaled_224x244'
workingdir = args.directory
clas = args.class_name
pore_folder = 'pore_'+clas
nonpore_folder = 'non-pore_'+clas
clas = args.run_name
name="weight\\weight_"+clas+".pt"
tempna='./'

load_name = "weight\\"+args.load_weight_name 
test_only=args.test_only
model_only = args.model_only
train_size = args.train_size
shot=args.shot_size
patiance = 250
#else:
#    weightname=sys.argv[2]
#    tempna='./'
#    name=tempna+weightname
#    test_only=0
N=min(10,args.shot_size)
class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 poreimgs,
                 nonporeimgs,
                 poreimgs_shot, 
                 nonporeimgs_shot,
                 img_transform1,
                 img_transform2,
                 study
                 ):
        #load 100 first images
        
        self.study = study
        #print(poreimgs)
        if study == 'train':
            poreimgs = poreimgs+poreimgs_shot
            nonporeimgs = nonporeimgs+nonporeimgs_shot
        self.size = len(poreimgs)
        #print(poreimgs_shot)   
        self.poreimgs_list = [
                os.path.join(img_path+'\\'+pore_folder+'\\', i) for i in poreimgs
            ]
        self.nonporeimgs_list = [
                os.path.join(img_path+'\\'+nonpore_folder+'\\', i) for i in nonporeimgs
            ]
        self.poreimgs_list_shot = [
                os.path.join(img_path+'\\'+pore_folder+'\\', i) for i in poreimgs_shot
            ]
        self.nonporeimgs_list_shot = [
                os.path.join(img_path+'\\'+nonpore_folder+'\\', i) for i in nonporeimgs_shot
            ]
        shuffle1 = np.arange(self.size*2);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size*2);np.random.shuffle(shuffle2)
        self.labels1 = np.concatenate((np.ones(int(self.size)),np.zeros(int(self.size))),axis=0)[shuffle1]
        self.labels2 = np.concatenate((np.ones(int(self.size)),np.zeros(int(self.size))),axis=0)[shuffle2]
        
        print(len(self.poreimgs_list),len(self.poreimgs_list_shot),self.size)
        self.img1_list = np.concatenate((self.poreimgs_list[0:int(self.size)],self.nonporeimgs_list[0:int(self.size)]),axis=0)[shuffle1]

        print(self.size/len(self.poreimgs_list_shot),len(self.nonporeimgs_list_shot*int(self.size/len(self.nonporeimgs_list_shot))),len(self.img1_list))

        self.img2_list = np.concatenate((self.poreimgs_list_shot*int(np.floor(self.size/len(self.poreimgs_list_shot))),self.poreimgs_list_shot[0:self.size%len(self.poreimgs_list_shot)],self.nonporeimgs_list_shot*int(np.floor(self.size/len(self.nonporeimgs_list_shot))),self.nonporeimgs_list_shot[0:self.size%len(self.nonporeimgs_list_shot)]),axis=0)[shuffle2]
        #compute logical XNOR
        self.label_list = np.logical_and(np.logical_or(self.labels1,np.logical_not(self.labels2)),np.logical_or(self.labels2,np.logical_not(self.labels1)))
        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
        #self.imgs_class1 = img_class1
        #self.imgs_class2 = img_class2

    def __getitem__(self, index):
        if (np.random.random()>0.5 and index==0):# and self.study == 'train'):
            self.shuffle()
            if self.study == 'train':
                print("Training set shuffled")
            else:
                print("Testing set shuffled")

        img1_path = self.img1_list[index]
        img2_path = self.img2_list[index]
        label = self.label_list[index]
        label=int(label)
        rand1 = False
        rand2 = False
        # add noise during training
        if (random.random()>0.995 and self.study=='train'):
            img1 = np.random.rand(224,224,3)*255
            rand1 =True
            label =0
        else:
            img1 = cv2.imread(img1_path)
        #if (random.random()>0.995 and self.study=='train'):
        #    img2 = np.random.rand(224,224,3)*255
        #    rand2 = True
        #    label =0
        #else:
        #    img2 = cv2.imread(img2_path)
        img2 = cv2.imread(img2_path)
        if rand1 and rand2:
            label = 1
            #print(rand1,rand2)
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
        shuffle1 = np.arange(self.size*2);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size*2);np.random.shuffle(shuffle2)
        self.labels1 = np.concatenate((np.ones(int(self.size)),np.zeros(int(self.size))),axis=0)[shuffle1]
        self.labels2 = np.concatenate((np.ones(int(self.size)),np.zeros(int(self.size))),axis=0)[shuffle2]
        self.img1_list = np.concatenate((self.poreimgs_list[0:int(self.size)],self.nonporeimgs_list[0:int(self.size)]),axis=0)[shuffle1]
        self.img2_list = np.concatenate((self.poreimgs_list_shot*int(np.floor(self.size/len(self.poreimgs_list_shot))),self.poreimgs_list_shot[0:self.size%len(self.poreimgs_list_shot)],self.nonporeimgs_list_shot*int(np.floor(self.size/len(self.nonporeimgs_list_shot))),self.nonporeimgs_list_shot[0:self.size%len(self.nonporeimgs_list_shot)]),axis=0)[shuffle2]
        self.label_list = np.logical_and(np.logical_or(self.labels1,np.logical_not(self.labels2)),np.logical_or(self.labels2,np.logical_not(self.labels1)))
    



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
        if random.random()<0.25:
            return cv2.flip(img,1)
        return img
        
class Rotate(object):
    def __call__(self,img):
        if random.random()<0.25:
            angle=random.random()*60-30
            rows,cols,cn = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            return img
        return img

class Translate(object):
    def __call__(self,img):
        if random.random()<0.00:
            x=random.random()*20-10
            y=random.random()*20-10
            rows,cols,cn = img.shape
            M= np.float32([[1,0,x],[0,1,y]])
            img = cv2.warpAffine(img,M,(cols,rows))
        return img
            
# load pretrained model
resnet18 = models.resnet18(pretrained=True)
my_model = nn.Sequential(*list(resnet18.children())[:-2])
my_model = my_model.cuda()

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        #unused with resnet18
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
        # used fully connected layers with resnet18
        self.fc = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            #nn.BatchNorm1d(512),
        )
        # fc2 outputs encodes an image to a 1024 vector space
        # loss function classifies based on distances within this space

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

def save_model(name,model,poreimgs,nonporeimgs,shot_size,train_size,val_size,test_size):
    save_dict = {'model':model.state_dict(),
                 'poreimgs':poreimgs,
                 'nonporeimgs':nonporeimgs,
                 'shot':shot_size,
                 'train':train_size,
                 'val':val_size,
                 'test':test_size}
    torch.save(save_dict, name)

def load_model(name):
    save_dict = torch.load(name)
    net_dic = save_dict['model']
    poreimgs = save_dict['poreimgs']
    nonporeimgs = save_dict['nonporeimgs']
    shot = save_dict['shot']
    train_size = save_dict['train']
    test_size = save_dict['val']
    final_test_size = save_dict['test']
    return net_dic,poreimgs[0:shot],nonporeimgs[0:shot],poreimgs[shot:shot+train_size],nonporeimgs[shot:shot+train_size],poreimgs[shot+train_size:shot+train_size+test_size],nonporeimgs[shot+train_size:shot+train_size+test_size],poreimgs[shot+train_size+test_size:final_test_size],nonporeimgs[shot+train_size+test_size:final_test_size]

if __name__ == '__main__': 
    transform1 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    transform2 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    net=Cnn()

    poreimgs_list = os.listdir(workingdir+'\\'+pore_folder)
    nonporeimgs_list = os.listdir(workingdir+'\\'+nonpore_folder)
    print(len(poreimgs_list),len(nonporeimgs_list),train_size)
    test_size = min(int(np.floor((len(poreimgs_list)-shot-train_size)*0.5/10)*10),int(np.floor((len(nonporeimgs_list)-shot-train_size)*0.5/10)*10))
    
    final_test_size = min(int(np.floor(len(poreimgs_list)/10)*10),int(np.floor(len(nonporeimgs_list)/10)*10))
    print(train_size,test_size,final_test_size)
    
    # generate train, validation, test sets
    random.shuffle(poreimgs_list)
    random.shuffle(nonporeimgs_list)
    poreimgs_shot = poreimgs_list[0:shot]
    nonporeimgs_shot = nonporeimgs_list[0:shot]
    poreimgs_train = poreimgs_list[shot:train_size+shot]
    nonporeimgs_train = nonporeimgs_list[shot:train_size+shot]
    poreimgs_val = poreimgs_list[train_size+shot:test_size+train_size+shot]
    nonporeimgs_val = nonporeimgs_list[train_size+shot:test_size+train_size+shot]
    poreimgs_test = poreimgs_list[test_size+train_size+shot:final_test_size]
    nonporeimgs_test = nonporeimgs_list[test_size+train_size+shot:final_test_size]
    #print(poreimgs_list,nonporeimgs_list)
    if test_only:
        print("loading")
        # load saved sets from model
        net_dic,poreimgs_shot,nonporeimgs_shot,poreimgs_train,nonporeimgs_train,poreimgs_val,nonporeimgs_val,poreimgs_test,nonporeimgs_test = load_model(load_name)

        net.load_state_dict(net_dic)
    elif model_only:
        print("loading")
        net_dic,_,_,_,_,_,_,_,_ = load_model(load_name)
        net.load_state_dict(net_dic)

    print("Validation size is: "+str(len(poreimgs_val)))
    print("Test size is: "+str(len(poreimgs_test)))

    if torch.cuda.is_available() :
        net = net.cuda()  
    
    train_set = custom_dset(workingdir,poreimgs_train, nonporeimgs_train, poreimgs_shot, nonporeimgs_shot, transform1,transform2,'train')
    train_loader = DataLoader(train_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)
    val_set = custom_dset(workingdir,poreimgs_val, nonporeimgs_val,poreimgs_shot, nonporeimgs_shot,transform1,transform2,'test')
    val_loader = DataLoader(val_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)  
    lr = args.learning_rate
    num_epoches = args.epoch


    optimizer = torch.optim.Adam(net.parameters(), lr)
    feature_encoder_scheduler = StepLR(optimizer,step_size=10,gamma=0.01)
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
    
        def forward(self, output1, output2, label):
            euclidean_distance = F.pairwise_distance(output1, output2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)+
                (label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
    
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
                    
                # print statistics
                running_loss += loss

            running_loss = running_loss / (i+1)
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            l_his.append(running_loss.cpu().detach().numpy())

            
            correct = 0
            total = 0

            for datat in val_loader:
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

            curr_acc = 100.0 * correct / total           
            print('Accuracy of the network on the validation images: %0.2f %%' % (
                curr_acc))
            if curr_acc > acc:
                save_model(name,net,poreimgs_list,nonporeimgs_list,shot,train_size,test_size,final_test_size)
                acc = curr_acc
            acc_hist.append(curr_acc)
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(acc_hist)    
            plt.xlabel('Epoch')  
            plt.ylabel('Acc') 
            try:
                fig.savefig('plots\\plott_acc'+clas+'.png') 
            except:
                print('save failed for some reason')
            plt.close()
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(l_his)    
            plt.xlabel('Epoch')  
            plt.ylabel('Loss')  
            try:
                fig.savefig('plots\\plot_loss'+clas+'.png') 
            except:
                print('save failed for some reason')
            plt.close()
            # if accuracy does not increase during patiance then overfitting likely occured
            if (np.array(acc_hist[-patiance:])<max(acc_hist)).all():
                break

        print('Finished Training')
        
        save_model('weight\\weight_final'+clas+'.pt',net,poreimgs_list,nonporeimgs_list,shot,train_size,test_size,final_test_size)
        #torch.save(net.state_dict(), 'weight\\weight_final_B4C.pt')
    else:   
        test_set = custom_dset(workingdir,poreimgs_test, nonporeimgs_test,poreimgs_shot, nonporeimgs_shot,transform1,transform2,'test')
        test_loader = DataLoader(test_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)
        #transform = transforms.Compose([transforms.ToTensor(),
        #                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        #val_set = custom_dset(workingdir,transform,transform,'train')
        #val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=2)   
        correct = 0
        total = 0
        for data in val_loader:
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
                if ((dist.data.numpy()[j]<0.7)):
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
        print('Accuracy of the network on the validation images: %d %%' % (
            100 * correct / total))
        
        #val_set = custom_dset(workingdir,transform,transform,'test')
        #val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=2)  
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
                if ((dist.data.numpy()[j]<0.7)):
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
