from tensorflow.keras.layers import Dropout,Dense,Conv2D,GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16,VGG19,ResNet50,InceptionResNetV2,DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import SGD,Adam
import os
class classifier(object):
    def __init__(self,img_dir,split_radio,batch_size,val_batch_size,base_model='vgg16',img_shape=(None,None,3),cls_num=21):
        self.img_shape=img_shape
        self.img_dir=img_dir
        self.base_model=base_model
        self.split_radio=split_radio
        self.batch_size=batch_size if self.img_shape!=(None,None,3) else 1
        self.val_batch_size=val_batch_size if self.img_shape!=(None,None,3) else 1
        self.cls_num=cls_num
    def model(self):
        if self.base_model.lower()=='vgg16':
            base_model=VGG16(weights='imagenet',include_top=False,input_shape=self.img_shape)
            input_layer=base_model.input
            layer=base_model.get_layer('block5_conv3').output
        elif self.base_model.lower()=='vgg19':
            base_model=VGG19(weights='imagenet',include_top=False,input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        elif self.base_model.lower()=='resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        elif 'inception' in self.base_model.lower():
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        else:
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.img_shape)
            input_layer = base_model.input
            layer = base_model.output
        # layer=GlobalAveragePooling2D()(layer)
        layer=Conv2D(1024,(1,1),padding='same',activation='relu',kernel_initializer='he_normal')(layer)
        # layer=Dense(512,activation='relu',kernel_initializer='he_normal')(layer)
        # layer=Dropout(0.5)(layer)
        # layer = Dense(128, activation='relu', kernel_initializer='he_normal')(layer)
        # layer = Dropout(0.5)(layer)
        # layer = Dense(64, activation='relu', kernel_initializer='he_normal')(layer)
        # layer = Dropout(0.5)(layer)
        # layer = Dense(32, activation='relu', kernel_initializer='he_normal')(layer)
        # layer = Dropout(0.5)(layer)
        layer = GlobalAveragePooling2D()(layer)
        prediction =Dense(self.cls_num,activation='softmax',kernel_initializer='he_normal')(layer)

        model=Model(input_layer,prediction)
        for layer in base_model.layers:
            layer.trainable=False

        model_finetune=Model(input_layer,prediction)
        for layer in base_model.layers:
            layer.trainable=True

        return model,model_finetune
    def train(self,initial_epoch=0,epochs=20,opt='sgd'):
        gen=ImageDataGenerator(
            rotation_range=360,horizontal_flip=True,vertical_flip=True,
            rescale=1/255,validation_split=self.split_radio
        )
        train_data = gen.flow_from_directory(
            directory=self.img_dir,
            batch_size=self.batch_size,
            target_size=(self.img_shape[0],self.img_shape[1]),
            shuffle=True,
            class_mode='categorical',
            subset='training'
        )
        val_data = gen.flow_from_directory(
            directory=self.img_dir,
            batch_size=self.val_batch_size,
            target_size=(self.img_shape[0],self.img_shape[1]),
            shuffle=True,
            class_mode='categorical',
            subset='validation'
        )
        model,model_finetune=self.model()
        optimizer=SGD(lr=0.001,momentum=0.9,decay=1e-6,nesterov=True) if opt=='sgd' else Adam(lr=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        model.summary()
        tb=TensorBoard(log_dir='logs',batch_size=self.batch_size)
        es=EarlyStopping(monitor='val_loss',patience=10,verbose=1,min_delta=0.0001)
        cp=ModelCheckpoint(filepath=os.path.join('model','cls--epoch_{epoch:02d}--val_loss_{val_loss:.5f}--val_acc_{val_acc:.5f}--train_loss_{loss:.5f}--train_acc_{acc:.5f}.hdf5'),
                           monitor='val_loss', save_best_only=False,save_weights_only=False,
                           verbose=1, mode='min', period=1)
        lr=ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1)
        his=model.fit_generator(
            generator=train_data,
            steps_per_epoch=train_data.samples//self.batch_size,
            validation_data=val_data,
            validation_steps=val_data.samples//self.val_batch_size,
            initial_epoch=initial_epoch,
            epochs=epochs,
            callbacks=[tb,es,cp,lr]
        )
        print(his)

if __name__=='__main__':
    model=classifier(
        img_dir='resorted_data',
        split_radio=0.2,
        batch_size=19,
        val_batch_size=19,
        base_model='vgg16',
        img_shape=(256,256,3),
        cls_num=21
    )
    model.train(initial_epoch=0,epochs=50,opt='adam')














