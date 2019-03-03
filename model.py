from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception
from keras.preprocessing import image
import os, pandas as pd
import numpy as np
from sklearn.cluster import KMeans
class Model:
    def __init__(self,model_name):
        self.model=model_name
    def build_model(self):
        if self.model=='Xception' or self.model=='xception':
            return Xception(include_top=False,pooling='avg',weights='imagenet'),'Xception'
        elif self.model=='DenseNet' or self.model=='densenet':
            return DenseNet121(include_top=False,pooling='avg',weights='imagenet'),'DenseNet'
        elif self.model=='ResNet' or self.model=='resnet':
            return ResNet50(include_top=False,pooling='avg',weights='imagenet'),'ResNet'
        elif self.model=='VGG16' or self.model=='vgg16':
            return VGG16(include_top=False,pooling='avg',weights='imagenet'),'VGG16'
        return VGG19(include_top=False,pooling='avg',weights='imagenet'),'VGG19'

def get_features(img_folder,Model):
    fp1=pd.DataFrame(columns=['filename','feature'])
    fp2 = pd.DataFrame(columns=['filename', 'feature'])
    for i in os.listdir(img_folder):#stage1 stage2
        next_folder=os.path.join(img_folder,i)
        for j in os.listdir(next_folder):#Belt st1...
            next_folder_plus=os.path.join(next_folder,j)
            for h in os.listdir(next_folder_plus):
                path=os.path.join(next_folder_plus,h)
                print(path)
                img=image.load_img(path,target_size=(224,224))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                features=Model[0].predict(x)[0]
                fp1=fp1.append({'filename':path,'feature':features},ignore_index=True)

                img = image.load_img(path)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                features = Model[0].predict(x)[0]
                fp2 = fp2.append({'filename': path, 'feature': features}, ignore_index=True)

                print(path+' finished!!')
    fp1.to_csv('data/{}_features_fixed_size.csv'.format(Model[1]),index=False)
    fp2.to_csv('data/{}_features_default_size.csv'.format(Model[1]),index=False)
    print('ALL DONE!')

class ImageCluster(object):
    def __init__(self,csv_file_path,base_img_folder,resorted_img_folder,
                 cluster_algo='kmeans',base_model='vgg16',k=None,maxK=None):
        self.csv_file=pd.read_csv(csv_file_path)
        self.cluster_algo=cluster_algo
        self.k=k
        self.maxK=maxK
        self.base_model,self.base_model_name=Model(model_name=base_model).build_model()
        self.base_img_folder=base_img_folder
        self.resorted_img_folder=resorted_img_folder
    def kmeans(self):
        x=[]
        for i in self.csv_file['feature']:
            x.append([float(t) for t in i.strip('[').strip(']').split(' ')])
        x=np.array(x)
        if os.path.exists('output'):
            pass
        else:
            os.mkdir('output')

        if os.path.exists('matplot'):
            pass
        else:
            os.mkdir('matplot')

        def func(k):
            model = KMeans(n_clusters=k, init='k-means++')
            model.fit(x)
            print('cluster_center', model.cluster_centers_)
            f = pd.DataFrame(columns=['filename', 'label'])
            f['filename'] = self.csv_file['filename']
            f['label'] = model.labels_
            f.to_csv('output/cluster_kmeans_{}.csv'.format(str(k)))
            return model.inertia_

        if self.k==None:
            sse=[]
            for k in range(2,self.maxK+1):
                sse.append(func(k))
            import matplotlib.pyplot as plt
            plt.plot(range(2,self.maxK+1),sse,marker='o')
            plt.xlabel('number of K(cluster)')
            plt.ylabel('SSE Value for each K')
            plt.title('KMeans for ImageCluster')
            plt.savefig('matplot/KMeans_maxK_{}.png'.format(str(self.maxK)))
            plt.show()
        else:
            func(self.k)
    def imagecluster(self):
        if self.cluster_algo.lower()=='kmeans':
            self.kmeans()
        else:
            print('不存在的模型，请重新输入模型名称！')
            return

    def resorted_img(self,selected_k_num):
        import shutil
        if os.path.exists(self.resorted_img_folder):
            pass
        else:
            os.mkdir(self.resorted_img_folder)

        resorted_csv=pd.read_csv('output/cluster_kmeans_{}.csv'.format(str(selected_k_num)))
        for i in resorted_csv.index:
            filename=resorted_csv.loc[i,'filename']
            label=resorted_csv.loc[i,'label']
            if os.path.exists(os.path.join(self.resorted_img_folder,str(label))):
                pass
            else:
                os.mkdir(os.path.join(self.resorted_img_folder,str(label)))
            shutil.copy(filename,os.path.join(self.resorted_img_folder,str(label)))
            print(os.path.join(self.resorted_img_folder,str(label))+' 复制成功！')

if __name__=='__main__':
    c=ImageCluster(
        csv_file_path='data/VGG16_features_fixed_size.csv',
        cluster_algo='kmeans',
        maxK=30,
        base_img_folder='data',
        resorted_img_folder='resorted_data',
    )
    c.resorted_img(selected_k_num=21)