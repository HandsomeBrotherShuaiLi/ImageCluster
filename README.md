# ImageCluster
The `ImageCluster` project is aimed to cluster unlabeled images based on the SOTA models.  
It designed for engineers and students to fast feature maps and cluster the image according to your cluster-algo hyperparameter.  
This flow contains two main steps:  
1. Use SOTA pre-trained models(egs:VGG16,VGG19,ResNet50 ) on `imagenet` to extract feature maps.  
   You can choose the bone model and  resize the images(default size is (224,224)) as you like.  
2. Choose a clustering algorithm (eg: kmeans ) to label these feature maps. Therefore the images can be labeled to K classes.  
So,the data flow is shown as below  
Image->array->feature map->labeling(cluster algorithm)  
