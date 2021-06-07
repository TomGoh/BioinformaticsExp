from astunparse.unparser import main
import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt


def build_single_block(filter_arr,kernel_size_arr,stride_arr,inputs,times):

    x=inputs;
    for i in range(0,times):
        x=tf.keras.layers.Conv2D(filters=filter_arr[0],kernel_size=(kernel_size_arr[0],kernel_size_arr[0]),strides=(stride_arr[0],stride_arr[0]),kernel_initializer='he_normal',padding='same')(x)
        x=tf.keras.layers.BatchNormalization(axis=3)(x)
        x=tf.keras.layers.Activation('elu')(x)

        x=tf.keras.layers.Conv2D(filters=filter_arr[1],kernel_size=(kernel_size_arr[1],kernel_size_arr[1]),strides=(stride_arr[1],stride_arr[1]),kernel_initializer='he_normal',padding='same')(x)
        x=tf.keras.layers.BatchNormalization(axis=3)(x)
        x=tf.keras.layers.Activation('elu')(x)
    

    return x

def constructer():
    inputs=tf.keras.layers.Input((512,512,3))
    c0=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='elu',kernel_initializer='he_normal',padding='same')(inputs)
    p0=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(c0)
    Deconv2=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(4,4),strides=(2,2),kernel_initializer='he_normal')(p0)

    block1=build_single_block(filter_arr=[128,128],kernel_size_arr=[3,3],stride_arr=[2,1],inputs=p0,times=1)
    block1=build_single_block(filter_arr=[128,128],kernel_size_arr=[3,3],stride_arr=[1,1],inputs=block1,times=2)
    Deconv3=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(8,8),strides=(4,4),kernel_initializer='he_normal',padding='same')(block1)


    block2=build_single_block(filter_arr=[256,256],kernel_size_arr=[3,3],stride_arr=[2,1],inputs=block1,times=1)
    block2=build_single_block(filter_arr=[256,256],kernel_size_arr=[3,3],stride_arr=[1,1],inputs=block2,times=1)

    block3=build_single_block(filter_arr=[512,512],kernel_size_arr=[3,3],stride_arr=[1,1],inputs=block2,times=6)

    block4=build_single_block(filter_arr=[512,1024],kernel_size_arr=[3,3],stride_arr=[1,1],inputs=block3,times=3)

    block5=build_single_block(filter_arr=[512,1024],kernel_size_arr=[1,3],stride_arr=[1,1],inputs=block4,times=1)
    block6=build_single_block(filter_arr=[2048,1024],kernel_size_arr=[1,1],stride_arr=[1,1],inputs=block5,times=1)
    block6=build_single_block(filter_arr=[2048,4096],kernel_size_arr=[3,1],stride_arr=[1,1],inputs=block6,times=1)

    Deconv1=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(16,16),strides=(8,8),padding='same',kernel_initializer='he_normal')(block6)
    
    fusion=tf.add(Deconv1,Deconv2)
    fusion=tf.add(fusion,Deconv3)

    outConv1=tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),kernel_initializer='he_normal',padding='same')(fusion)
    d1=tf.keras.layers.Dropout(rate=0.5)(outConv1)
    outConv2=tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),kernel_initializer='he_normal',padding='same')(d1)
    outputs=tf.keras.layers.Softmax()(outConv2)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
if __name__=='__main__':
    constructer()


# 模型搭建完了，接下来就是对于数据进行预处理后训练模型
#TODO: 增强数据的预处理，增加比例拉伸等内容训练数据
#TODO: 训练模型

