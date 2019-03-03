from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception
from keras.preprocessing import image
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
    import os,pandas as pd
    import numpy as np
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
if __name__=='__main__':
    model=Model(model_name='vgg16').build_model()
    get_features('data',model)





