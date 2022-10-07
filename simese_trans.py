
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
import os,copy
import random
import sys, argparse
from torchvision import models

# see run.bat and test.bat for examples
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-d","--directory",type = str, default = 'D:\\thesis_working\\Mat_Cropped_Imgs\\scaled_4class\\')
parser.add_argument("-c","--class_name",type = str, default = 'new')
parser.add_argument("-n","--run_name",type = str, default = 'new_10')
parser.add_argument("-l","--load_weight_name",type = str, default = "weight_a.pt")
parser.add_argument("-t","--test_only",type = int, default = 0)
parser.add_argument("-m","--model_only",type = int, default = 0)
parser.add_argument("-ts","--train_size",type = int, default= 150)
parser.add_argument("-sh","--shot_size", type = int, default = 10)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-e","--epoch",type=int, default=5000)

args = parser.parse_args()

#workingdir = 'D:\\thesis_working\\poredata_cropped_scaled_224x244'
clas = args.class_name
workingdir = args.directory+clas

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
N=min(15,args.shot_size)

# run once for test or validation then use
# results for knn classification
class classification_dset(Dataset):
    def __init__(self,
                 file_dict):
        self.shot_set = []
        self.shot_class = []
        self.shot_folder =[]
        self.dir = file_dict['workingdir']
        for i,k in enumerate(file_dict['shot'].keys()):
            self.shot_set = self.shot_set+file_dict['shot'][k]
            index_id = file_dict['class_index'][k]
            self.shot_class = self.shot_class+[index_id]*len(file_dict['shot'][k])
            self.shot_folder = self.shot_folder+[k]*len(file_dict['shot'][k])
            #print(self.shot_class,file_dict['shot'][k])

    def __getitem__(self, index):
        #print('sizes',len(self.shot_set),len(self.shot_class))
        sclass = self.shot_class[index]
        folder = self.shot_folder[index]
        #print(self.shot_set[index],folder,sclass)
        img_path = folder+"\\"+self.shot_set[index]
        img = cv2.imread(self.dir+"\\"+img_path)
        return img,np.array(sclass)

    def __len__(self):
        return len(self.shot_set)


class custom_dset(Dataset):
    def __init__(self,
                 file_dict,
                 img_transform1,
                 img_transform2,
                 study
                 ):
        #load 100 first images
        
        self.study = study
        self.dir = file_dict['workingdir']
        # Generate training set with indexes
        self.img_set=[]
        self.img_index =[]
        self.num_images = 0
        for k in file_dict[self.study].keys():
            self.num_images+=len(file_dict[self.study][k])

        self.hot = torch.zeros([self.num_images, 1])
        self.hot_shot = torch.zeros([self.num_images, len(file_dict[self.study].keys())])
        flag = 0
        for i,k in enumerate(file_dict[self.study].keys()):
            self.img_set=self.img_set+file_dict[self.study][k]
            index_id = file_dict['class_index'][k]
            self.img_index=self.img_index+[k]*len(file_dict[self.study][k])
            self.hot[flag:flag+len(file_dict[self.study][k]),0]=index_id
            flag += len(file_dict[self.study][k])

        
        #print(self.img_index)
        # Generate support set with indexes
        num_classes = len(file_dict['shot'].keys())
        img_per_class = int(np.floor(len(self.img_index)/num_classes/shot))
        remainder = len(self.img_index)%(num_classes*shot)
        self.shot_set = []
        self.shot_index = []
        for k in file_dict['shot'].keys():
            self.shot_set = self.shot_set+file_dict['shot'][k]*img_per_class
            self.shot_index = self.shot_index+[k]*img_per_class*len(file_dict['shot'][k])
            print("int check",len(file_dict['shot'][k]*img_per_class),len([k]*img_per_class*len(file_dict['shot'][k])))
        self.shot_set = self.shot_set+file_dict['shot'][k]*remainder
        self.shot_index = self.shot_index+[k]*remainder
        #print(self.shot_set)
        #print(self.shot_index)
        print(len(self.shot_index),len(self.img_index))

        self.size = len(self.shot_index)

 

        shuffle1 = np.arange(self.size);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size);np.random.shuffle(shuffle2)
        print(len(shuffle1),len(self.img_set))
        self.img_set=np.array(self.img_set)[shuffle1]
        self.img_index = np.array(self.img_index)[shuffle1]
        self.hot=self.hot[shuffle1,:]
        #print(self.hot)
        self.shot_set= np.array(self.shot_set)[shuffle2]
        self.shot_index = np.array(self.shot_index)[shuffle2]
        self.hot_shot = self.hot_shot[shuffle2,:]
        #compute logical XNOR
        self.label_list = [int(x==self.img_index[i]) for i,x in enumerate(self.shot_index)]
        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
        #self.imgs_class1 = img_class1
        #self.imgs_class2 = img_class2

    def __getitem__(self, index):
        if (np.random.random()>0.5 and index==0):# and args.test_only == 0):
            if self.study == 'train':
                print("Training set shuffled")
                self.shuffle()

        folder1 = self.img_index[index]
        folder2 = self.shot_index[index]
        img1_path = folder1+"\\"+self.img_set[index]
        img2_path = folder2+"\\"+self.shot_set[index]
        hot = self.hot[index]
        hot_shot = self.hot_shot[index]
        label = self.label_list[index]
        label=int(label)
        rand1 = False
        rand2 = False
        # add noise during training
        if (random.random()>10.995 and self.study=='train'):
            img1 = np.random.rand(224,224,3)*255
            rand1 =True
            label =0
        else:
            img1 = cv2.imread(self.dir+"\\"+img1_path)
            #print(self.dir+"\\"+img1_path)
        #if (random.random()>0.995 and self.study=='train'):
        #    img2 = np.random.rand(224,224,3)*255
        #    rand2 = True
        #    label =0
        #else:
        #    img2 = cv2.imread(img2_path)
        img2 = cv2.imread(self.dir+"\\"+img2_path)
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

        return img1,img2,label,hot,hot_shot
    def __len__(self):
        return len(self.label_list)
    def shuffle(self):
        shuffle1 = np.arange(self.size);np.random.shuffle(shuffle1)
        shuffle2 = np.arange(self.size);np.random.shuffle(shuffle2)
        self.img_set=self.img_set[shuffle1]
        self.img_index = self.img_index[shuffle1]
        self.hot=self.hot[shuffle1,:]
        self.shot_set=self.shot_set[shuffle2]
        self.shot_index =self.shot_index[shuffle2]
        self.hot_shot = self.hot_shot[shuffle2,:]
        self.label_list = [int(x==self.img_index[i]) for i,x in enumerate(self.shot_index)]



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

        # used fully connected layers with resnet18
        self.fc = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(),
            #nn.BatchNorm1d(2048),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            #nn.BatchNorm1d(1024),
        )

            
        
        # fc2 outputs encodes an image to a 1024 vector space
        # loss function classifies based on distances within this space

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1,3, 224,224)
        #print(x.shape)
        x = my_model(x)
        #print(x.shape)

        x = x.view(x.size(0), -1)
        #print(x.shape)
        x1 = self.fc(x)
        x_sim = self.fc2(x1)

        #print(x_cls.shape)
        return x_sim

def save_model(name,model,file_dict):
    save_dict = {'model':model.state_dict(),
                 'data_dict':file_dict}
    torch.save(save_dict, name)

def load_model(name):
    save_dict = torch.load(name)
    net_dic = save_dict['model']
    file_dict = save_dict['data_dict']
    return net_dic,file_dict

def generate_sets(workingdir):
    # Creates classes from folders
    folders = os.listdir(workingdir)
    file_dict = {}
    file_dict['workingdir']=workingdir
    file_dict['shot' ]={}
    file_dict['val'  ]={}
    file_dict['test' ]={}
    file_dict['train']={}
    file_dict['train_weight']=[]
    file_dict['class_index']={}
    total = 0
    index = 0
    for classes in folders:
        subdir = workingdir+"\\"+classes
        print(subdir)
        if os.path.isdir(subdir):
            files = os.listdir(subdir)
            random.shuffle(files)
            file_dict['shot' ][classes]=files[0:shot]
            file_dict['val'  ][classes]=files[shot:train_size+shot]
            file_dict['test' ][classes]=files[shot+train_size:shot+2*(train_size)]
            file_dict['train'][classes]=files[shot+2*(train_size):]+files[0:shot]
            file_dict['class_index'][classes]=index
            index+=1
            total += len(file_dict['train'][classes])
            print(len(file_dict['shot' ][classes]),len(file_dict['val'  ][classes]),len(file_dict['test' ][classes]),len(file_dict['train'][classes]))
    for classes in file_dict['train'].keys():
        file_dict['train_weight'].append(total/len(file_dict['train'][classes]))
    file_dict['train_weight'] = file_dict['train_weight']/np.sum(file_dict['train_weight'])
    #print(file_dict)
    return file_dict
            
            
class batch_knn():
    def __init__(self,neighbors,batch,num_classes):
        self.num_neighbors = neighbors
        self.num_classes = num_classes
        self.batch = batch

    def load_neighbors(self,neighbor_features, neighbor_classes):
        # We need to concat the tensor so we can vectorize the comparison
        # this means the tensor is repeated for each output in the batch
        #self.neighbors = torch.cat([neighbor_features]*self.batch,dim=0)
        self.neighbors = neighbor_features
        self.classes = neighbor_classes
        #print(self.neighbors.shape,neighbor_features.shape)
    def classify(self,batch_tensor):
        # Now we need to compute the eucledian distance from the neighbors
        # to each element in the batch, then sort them and return the class
        # corresponding with the closest number of neighbors
        # To do this we need to concant the tensor so that each entry 
        # is repeated for the neighbors it is compared to
        batch = batch_tensor.shape[0]
        feature_len = batch_tensor.shape[1]
        thesh = (feature_len*0.5)**(1/feature_len)

        predict_class = torch.zeros(self.batch).cuda()
        for i in range(batch):
            #prediction = torch.cat([batch_tensor[i,:]]*self.num_classes*batch,dim=0).view(self.num_classes*batch,-1)
            prediction = batch_tensor[i,:]

            dist = F.pairwise_distance(prediction,self.neighbors)
            #print(dist)
            close = torch.lt(dist,thesh)
            #print(dist)
            #print(close)
            dist_class = close.nonzero()
            #print(dist_class)
            try:
                class_select,_ = torch.mode(self.classes[dist_class],0)
            except:
                class_select = 5


            #index = torch.lt(torch.argsort(dist,dim=0,descending=True),self.num_neighbors).nonzero()
            #class_select,_ = torch.mode(self.classes[index],0)
            #print(class_select)
            predict_class[i]=class_select

        return predict_class


if __name__ == '__main__': 
    transform1 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    transform2 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    net=Cnn()
    file_dict=generate_sets(workingdir)

    if test_only:
        print("loading")
        # load saved sets from model
        net_dic,file_dict = load_model(load_name)

        net.load_state_dict(net_dic)
    elif model_only:
        print("loading")
        net_dic,_ = load_model(load_name)
        net.load_state_dict(net_dic)

    #print("Validation size is: "+str(len(poreimgs_val)))
    #print("Test size is: "+str(len(poreimgs_test)))

    if torch.cuda.is_available() :
        net = net.cuda()  
    
    train_set = custom_dset(file_dict, transform1,transform2,'train')
    train_loader = DataLoader(train_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)
    val_set = custom_dset(file_dict,transform1,transform2,'val')
    val_loader = DataLoader(val_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)  
    shot_set = classification_dset(file_dict)
    shot_loader = DataLoader(shot_set, batch_size=N, shuffle=False, num_workers=5,pin_memory=True,persistent_workers=True)  
    lr = args.learning_rate
    num_epoches = args.epoch


    optimizer = torch.optim.Adam(net.parameters(), lr)
    feature_encoder_scheduler = StepLR(optimizer,step_size=1000,gamma=0.1)
    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
            self.loss_cre = nn.CosineEmbeddingLoss()
    
        def forward(self, output1, output2, labels):
            #euclidean_distance = F.pairwise_distance(output1, output2)
            #loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)+
            #    (label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            #print(output1.shape,output2.shape)
            loss_contrastive = 0
            for i,label in enumerate(labels):
                #print(label[0])
                extend = torch.cat([output1[i,:]]*output2.shape[0],dim=0).view(output2.shape[0],-1)
                #print(extend.shape,output2.shape)
                loss_contrastive += self.loss_cre(extend,output2,label[0]*2-1)
            #loss_cross_entropy = self.loss_cre(hot,hot_pre)
            #return (loss_contrastive*10/(epoch+10)+loss_cross_entropy*(1-10/(epoch+10)))
            return loss_contrastive#*0.99+loss_cross_entropy*0.01)
        
    
    loss_func = ContrastiveLoss() 
    knn_class = batch_knn(5,10,4)
    l_his=[]
    acc_hist = []
    if test_only==0:
        acc = 0
        for epoch in range(num_epoches):
            print('Epoch:', epoch + 1, 'Training...')
            running_loss = 0.0 
            

            for i,data in enumerate(train_loader, 0):
                shot_features = []
                shot_classes = []
                for shot_data in shot_loader:
                    shot_image, shot_class = shot_data
                    if torch.cuda.is_available():
                        shot_image = shot_image.cuda()
                        shot_class = shot_class.cuda()
    
                    sf = net(shot_image.float())
                    shot_features.append(sf)
                    shot_classes.append(shot_class)
                shot_classes=torch.cat(tuple(shot_classes),dim=0)
                shot_features=torch.cat(tuple(shot_features),dim=0)
                knn_class.load_neighbors(shot_features,shot_classes)
                image1s,image2s,labels,hot,hot_shot=data
                if torch.cuda.is_available():
                    image1s = image1s.cuda()
                    image2s = image2s.cuda()
                    labels = labels.cuda()
                    hot = hot.cuda()
                    hot_shot = hot_shot.cuda()

                image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
                optimizer.zero_grad()
                f1=net(image1s.float())
                #print(shot_classes)
                #print(hot)
                labels = [[x==shot_classes] for i,x in enumerate(hot)]
                #print(labels)
                loss = loss_func(f1,shot_features,labels)
                loss.backward()
                optimizer.step()

                    
                # print statistics
                running_loss += loss

            running_loss = running_loss / (i+1)
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            l_his.append(running_loss.cpu().detach().numpy())

            
            correct = 0
            total = 0
            with torch.no_grad():
                #store features and classes for knn comparison
                shot_features = []
                shot_classes = []
                for shot_data in shot_loader:
                    shot_image, shot_class = shot_data
                    if torch.cuda.is_available():
                        shot_image = shot_image.cuda()
                        shot_class = shot_class.cuda()
    
                    sf = net(shot_image.float())
                    shot_features.append(sf)
                    shot_classes.append(shot_class)
                shot_classes=torch.cat(tuple(shot_classes),dim=0)
                shot_features=torch.cat(tuple(shot_features),dim=0)
                knn_class.load_neighbors(shot_features,shot_classes)
                #print(shot_classes)
                for datat in val_loader:
                    image1st,_,_,labelst,_ = datat
                    if torch.cuda.is_available():
                        image1st = image1st.cuda()
                        labelst = labelst.cuda()
                    
                    # We compute the output feature and use knn to predict the label
                    f1 = net(image1st.float())
                    predict = knn_class.classify(f1)
                    correct += torch.sum(predict.view(-1,1)==labelst)/labelst.shape[0]
                    #print("pred",predict.view(-1,1))
                    #print("label",labelst)
                    #print(correct)
                    total+=1


                        

            curr_acc = 100.0 * correct / total           
            print('Accuracy of the network on the validation images: %0.2f %%' % (
                curr_acc))
            if curr_acc > acc:
                save_model(name,net,file_dict)
                acc = curr_acc
            acc_hist.append(curr_acc.cpu().numpy())
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
        
        save_model('weight\\weight_final'+clas+'.pt',net,file_dict)
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
