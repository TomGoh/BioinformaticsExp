<h2>
    目录
</h2>

[toc]

<div STYLE="page-break-after: always;"></div>

## 实验目的

- 掌握深度学习图像分割的相关方法，如：CNN，FCN，残差网络，扩展网络 等。
- 完成图像分割模型的创建。 
- 了解与医疗生物数据相关的数据库,如：ISBI 挑战赛中的数据库等。
- 完成细胞相关数据库的筛选和整理。

<div STYLE="page-break-after: always;"></div>

## 实验原理

图像分割（Image Segmentation）一直是计算机视觉的热点问题，作为一项极其有价值的技术被包括生物信息学在内的其他交叉学科所重视。图像分割技术旨在将所给的图片根据需求分割成若干个部分，其中较简单的例子就是将图像中的前景主题对象与背景进行分离。而在生物信息学领域[1]，图像分割的重心在对于医学图像（Medical Image，MI​）的分割上，进行分割的图像载体包括CT图像，MRI图像，EM图像和X光图像等。以本次实验为例，本次实验进行的细胞壁分割所使用的数据集即为ISBI 挑战赛（ISBI Challenge 2012）所提供的果蝇胚胎腹神经索的电镜图片。

在图像分割领域，目前具体流派分为三种，分别为实例分割（Instance Segmentation），语义分割（Semantic Segmentation）和全景分割（Panoramic Segmentation）。在医学图像领域的分割主要为语义分割。在神经网络诞生之前，图像分割的主流方法为基于阈值、基于区域和基于边界检测的图像分割的算法。一些比较优秀的算法，如分水岭算法[2]（Watershed Algorithm）在细胞壁分割这一主题上可以起到很好的效果，我们也将看到这些在神经网络出现之前使用的方法在神经网络的预处理和后处理过程中也将发挥不小的作用。在AlexNet诞生后的神经网络年代，在图像的语义分割领域出现了许多基于卷积网络的优秀神经网络模型，包括在CVPR2015众人瞩目的FCN[3]（Fully Convolutional Network）、同年发表在MICAI上的UNet[4]，以及同样在2015年被提出的ResNet[4]，都在图像的语义分割领域展现了良好的表现。本次实验将重点研究在医学图像领域拥有极好效果的UNet、部分基于UNet的改进网络[6]和FCN。

### 全卷积网络

2015年，Long等人[3]提出全卷积网络（Fully Convolutional Networks，FCN）用于图像的语义分割。FCN是图像语义分割的基本框架，之后提出的语义分割网络多数是基于其进行的改进。

与传统的CNN分类网络（如VGG16，ResNet等）不同的是，FCN将全连接层替换为卷积层，以保留特征图的空间信息，实现像素级别的精确分类。FCN通过使用局部连接和权重共享的卷积层取代连接密集的全连接层的策略，大大减少了需要训练的参数量。

总体而言，FCN采取编码器-解码器（encoder-decoder）结构，以接受任意大小的输入并且产生相同大小的输出。对于给定的输入图像，编码器将输入转换为高层特征的表示，解码器使用反卷积和特征融合来恢复图像，并且通过Softmax提供每个像素的分割结果。编码器逐步降低空间维度提取特征信息的过程，称为下采样，解码器根据特征信息逐步恢复目标细节和空间维度的过程，称为上采样。



<img src="https://gitee.com/TomGoh/img/raw/master/20210607092220.png" style="zoom: 80%;" />

<center style="color:#C0C0C0;">Fig1. FCN的基本结构示意图</center>



FCN基本结构如图所示，在结构示意图中， 仅展示了池化层和预测层，省略了中间卷积层（包括转换连接层）。该结构可以描述为：输入图像经过多个卷积层和一个池化层得到特征pool1，尺度为初始的1/2；pool1经过多个卷积层和一个池化层得到特征pool2，尺度为初始的1/4；重复这个过程，直至获得特征pool5，尺度为初始的1/32。FCN-32s直接对pool5进行32倍上采样获得特征，并对32倍上采样特征每个点做softmax预测。FCN-16s需要对pool5进行2倍上采样获得特征，再把pool4和2倍上采样获得的特征相加，然后对相加的特征进行16倍上采样，并做softmax预测。进一步的，FCN-8s在进行pool4和2倍上采样获得的特征相加之后，继续将pool3与上采样获得的特征相加，进行了多次的特征融合。其中，上采样操作使用反卷积实现，如使用3×3的反卷积核，步长（stride）为2，将特征图尺度恢复为2倍，而反卷积层可以用反池化层和上采样层代替。

根据[3]中的分析和其实验得出的结论，使用多次的特征融合能有效提高图像语义分割的准确度，因此我们在本实验中实现时选择了FCN-8s的策略。



### U-Net

Ronneberger等人[4]基于FCN提出了U-Net，该模型目前已经广泛应用于医学图像分析和处理。

U-Net基于FCN使用反卷积恢复图像大小和特征的思想，建立了类似的编码器-解码器结构，如图所示。U-Net的上采样过程中没有使用FCN采样时直接相加特征的融合操作，而是在使用拼接（concatenate）操作来形成特征。在拼接之后，对特征进行反卷积操作。与传统的卷积、池化等操作不同，这种直接利用浅层特征的策略称为残差连接(skip connection)。U-Net将残差连接应用于所有尺度的浅层特征信息，充分地利用了编码器下采样部分的特征来提供给上采样过程。

<img src="https://gitee.com/TomGoh/img/raw/master/20210607092224.png" style="zoom:80%;" />

<center style="color:#C0C0C0;">Fig2. UNet的基本结构示意图</center>

### 基于UNet的改进

同样的，对于UNet进行改进的举措在论文中也很常见。而在对UNet的改进的一种网络FusionNet[7]中，Tran Minh Quan等人在原有的UNet的基础上增加了残差网络中的残差模块，用以防止梯度消失，同时保存了UNet中的残差链接，以利用UNet的优秀特性。

除了添加残差模块这种对于网络层级进行修改的方法，研究者还对UNet的整体结构进行了部分调整。如在M-Net[6]中，Huazhu Fu等人在原有UNet的单张图片输入+残差链接的基础上，在网络的输入层部分也增加了多个放缩后的输入，并在输出层面综合解码器部分的多个输出叠加在一起后总和输出，增强了网络特征提取的的能力。

<div STYLE="page-break-after: always;"></div>

## 实验内容

### 医疗影像语义分割任务数据集整理

在数据集筛选和整理阶段，我们发现在细胞壁分割任务领域的标注数据集较少，相关文献数量有限。因此，基于对于医学图像分割算法（主要研究FCN、UNet及其改进算法）对比研究的需要，我们筛选了一些诸如视网膜血管分割任务DRIVE等数据集在此列出。

####  [ISBI 2012 Challenge](http://brainiac2.mit.edu/isbi_challenge/home)

IEEE 国际生物医学成像研讨会 (IEEE International Symposium on Biomedical Imaging，ISBI) 是一个致力于在所有观察尺度上讨论生物医学成像的数学、算法和计算方面的问题的会议。 ISBI 2012 挑战赛提供的训练数据为透射电镜拍摄的30幅果蝇幼虫腹神经索的连续切片图像(512×512像素)。每张图像都附有相应的完整细胞（白色）、细胞膜（黑色）的GT分割图。

####  [DRIVE](https://drive.grand-challenge.org/)

DRIVE[8]（Digital Retinal Images for Vessel Extraction）数据库用于对视网膜图像中的血管分割进行比较研究。数据集中有40张照片，其中7张显示出轻度早期糖尿病视网膜病变的迹象。 

#### [LiTS](https://competitions.codalab.org/competitions/17094)

肝脏是原发性或继发性肿瘤发展的常见部位。由于它们的异质和扩散形状，肿瘤病变的自动分割非常具有挑战性。到目前为止，只有使用交互式方法才能获得可接受的肝脏病变分割结果。

该数据集[9]和分割结果由世界各地的各种临床站点提供。训练数据集包含130个CT扫描数据，测试数据集包含70个CT扫描数据。该挑战是与ISBI 2017和MICCAI 2017一起组织的。

####  [CHAOS](https://chaos.grand-challenge.org/)

该挑战旨在从CT和MRI数据中分割腹部器官（肝、肾和脾）。挑战中提供了两个数据库：腹部CT和MRI。这两个数据库中的每个数据集都对应于属于单个患者的一系列DICOM图像，从CT和MR数据库获得的数据集之间没有联系。

#### [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

该挑战和数据集旨在通过标准化分析和验证过程，在几个高度不同的任务上开源大型医学图像数据集。

|   任务   |            标注目标             |        模态         | 数量 |
| :------: | :-----------------------------: | :-----------------: | :--: |
| 肝脏肿瘤 |           肝脏和肿瘤            |     门静脉期CT      | 201  |
|  脑肿瘤  | 胶质瘤分割坏死/活动性肿瘤和水肿 | 多模态多部位MRI数据 | 750  |
|  海马体  |             海马体              |      单模态MRI      | 394  |
|  肺肿瘤  |            肺和肿瘤             |         CT          |  96  |
|  前列腺  |    前列腺中央腺体和周边区域     |      多模态MR       |  48  |
|   心脏   |             左心房              |      单模态MRI      |  30  |
|  结肠癌  |             结肠癌              |         CT          | 190  |
|  肝血管  |          肝血管和肿瘤           |         CT          | 443  |
|    脾    |               脾                |         CT          |  61  |

#### [LoLa11](https://lola11.grand-challenge.org/)

该数据集提供了许多胸部CT扫描数据。扫描数据来源不一，选取了各种临床常用的扫描仪和协议的扫描结果。在数据集大约一半的扫描数据中，肺和/或肺叶分割任务是“容易的（easy）”，而在另一半数据中是“困难的（hard）”。最大切片间距为1.5毫米，其中大多数扫描是（近）各向同性的。为确保一致的评估，尚未有数据集的参考分割。

#### [KiTS19](https://kits19.grand-challenge.org/)

该挑战旨在促进可靠的肾脏和肾脏肿瘤语义分割方法的发展。数据集提供了300名在指定机构接受部分或根治性肾切除术的肾癌患者的动脉期腹部 CT 扫描生成了真实语义分割图像。公开发布了其中210个用于模型训练和验证，其余90个将用于客观模型评估。

#### [Data Science Bowl 18](https://www.kaggle.com/c/data-science-bowl-2018)

该挑战[10]提供了细胞核分割数据集，包括841张/37333个细胞核标注的图片。

#### [Isic Archive - Melanoma](https://www.isic-archive.com/)

该档案包含分类的皮肤病变的图像。它包含恶性和良性的示例。每个示例都包含病变的图像、有关病变的元数据（包括分类和分割）和有关患者的元数据。

#### [SCR数据库](http://www.isi.uu.nl/Research/Databases/SCR/)

胸片中解剖结构的自动分割对于在这些图像中的计算机辅助诊断工作而言非常重要。 该数据库旨在促进对标准前后胸片中肺野、心脏和锁骨分割的比较研究。

<div STYLE="page-break-after: always;"></div>

### 搭建、训练、测试神经网络

在本次实验中，我们具体实现了UNet，FCN-8s（基于VGG）以及UNet的一个改进版本MNet。同时，我们对UNet和FCN使用ISBI 2012 挑战赛的数据集（ISBI 2012 Challenge）以及糖尿病人眼底血管的DRIVE（Digital Retinal Images for Vessel Extraction）数据集进行训练，在不同数据集以及不同训练次数的条件下评估其表现。

神经网络模型均基于Tensorflow 2.5.0中的Keras进行搭建，实验环境如下：

> OS: Windows10 21H1
>
> CPU: AMD R7-3750H @2.3GHz 4C8T
>
> Memory: 16GB
>
> Graphics Card: Nvidia GeForce 1660Ti with MaxQ Design
>
> Graphics Memory: 6GB
>
> CUDA Version: CUDA 11.1
>
> cuDNN Version: 8.2.0
>
> Tensorflow Version: 2.4.0

在模型的训练过程中，由于算力有限，使用Google Colaboratory对部分模型进行训练，Google Colaboratory提供的GPU为Nvidia Tesla T4，CUDA版本为11.2。

<div STYLE="page-break-after: always;"></div>

### 数据增强与预处理

在本次实验选择进行训练的数据集中，无论是ISBI 2012 Challenge的果蝇EM图像还是DRIVE眼底血管图像都存在数据量严重不足的问题：ISBI 2012挑战赛数据集仅有30张图像，而DRIVE仅有40张图像，其中训练集20张，测试集20张且测试集中不包括眼球血管分割结果。

这样体量的数据当然无法满足神经网络的训练，即使使用这些有限的数据对UNet这样参数量较小的模型进行训练，其最终结果也将是严重过拟合的。而对于参数量较大的FCN则可能存在参数训练不到位的问题，因此需要对数据进行增强。

对于每张数据集中的图像，采取三种方法进行数据增强：

- 水平对称

  将图像以图象的水平中线为对称轴进行对称。

- 垂直对称

  将图像以图像的竖直中线为对称轴进行对称。

- 剪切

  将图像均等分割为四个部分，每个部分重新设置大小为源图像的大小。

在经过上述过程分别对训练集中的原图像和标签图像进行数据增强后，图像数据集的规模将增至原先的7倍：ISBI 2012 Challenge数据集的训练集增加至210张，DRIVE训练集增加至140张。

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210606235502.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.3 裁切原图片
    </div> 
</center>

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210606235646.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.4 镜像原图片
    </div> 
</center>




在数据增强后获得的数据集的基础上针对不同的神经网络进行不同的预处理：

- UNet

  根据[4]，UNet的输入图像的尺寸大小设置为$256\times 256\times 3$，输出的图像为$256\times 256\times 1$因此需要将输入的图像预处理至改尺寸。同时，针对训练集中的训练图像采取RGB通道读取，针对训练集中的标签图像采取灰度读取，以适应神经网络输入和输出的通道数。

- FCN

  根据[3]，FCN的输入层的图像尺寸为$224 \times 224 \times 3$，输出的图像为$224\times 224 \times n$，其中$n$代表的是语义分割中类别，在ISBI 2012 Challenge数据集中存在两个类别：细胞质和细胞壁，在DRIVE中也存在两个数据类别：眼球血管和眼底其他物质。因此在该FCN中，语义分割的实际类别将被设置为$2$，因此输出的结果为$224 \times 224 \times 2$的矩阵，因此作为标签图像的输入的每个像素点需要对应一个数值，该数值表示该像素所对应的类别，这里分别为0和1，因此在读入标签图像时采取灰度模式，在图像唯一的通道中仅包含0和1两个数值即可。

<div STYLE="page-break-after: always;"></div>

### 神经网络搭建

#### UNet

##### 输入层

在数据增强和预处理完成的基础之上，我们着手构建神经网络的架构。UNet的架构参考2015年在CVPR上的论文进行构建。

输入层为一个$256\times 256\times 3$的模块，使用Tensorflow.Keras包中的`layers.Input`进行搭建，这件免去将图片的Numpy转换成Tensorflow支持的张量的过程。

##### 编码器

在输入层之后就是UNet网络的编码部分。这里根据原论文采用了四次基于MaxPooling的下采样进行编码。在每一次下采样之间有两个卷积层。为了方便残差连接时的处理方便，即避免在UNet编码器和解码器之间的残差连接由于卷积后数据尾部不一致而必须做裁切操作，在实际实现UNet时将每次卷积的padding设置为same，这样在编码器和解码器之间的残差连接时就无需考虑卷积层数据维度的问题，直接连接即可。当然，这样的处理对于UNet模型的性能不会有可见的影响。

编码器中一层的代码如下：

```python
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
```

除了CVPR2015原文中的卷积层和MaxPooling层之外，为了较好地避免过拟合，特地在每个编码器层的两个卷积层之间添加一次Dropout。

##### 解码器

在4个编码器之后的是U型网络的右半部分，也就是由四次上采样组成的解码器层。在每个解码器之中需要完成对于上一次层输入的反卷积、两次卷积和对于U型网络前半部分编码器的残差连接。同样的，为了避免过拟合，在这里在每一个解码器层都设置一个Dropout层。

解码器中一层的代码如下：

```python
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
```

##### 输出层

输出层部分较为简单，仅仅采用一个针对解码器的卷积层就可以得到最终处理之后的图像了。需要注意的是，在最后的输出层与前文的卷积层存在一个不一致的设定，那就是这里的输出层采用sigmoid激活函数，对于结果进行归一化处理，以得到最终的Mask输出。

输出层代码如下：

```python
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
```

后续结果与评估参见报告的“实验结果”模块。

#### FCN

##### 输入层

与UNet类似，输入都是使用相同的模块完成。只不过在编写FCN网络的过程采用的面向对象的方式，通过继承Tensorflow.Keras中的Model类来实现模块的定义。类中通过继承`__init__`函数和`call`函数来分别实现神经网络的定义和构建调用。此时仅需要在`call`的参数中声明输入即可，而无需类似UNet实现中使用Sequential显式指定输入的方式。

##### 卷积模块

本实验实现的FCN是采用VGG16作为主干网络的。因此在实现卷积层时与VGG16基本相似。前两个卷积模块均是使用 2个卷积层+1个MaxPooling的组合实现；后三个卷积模块均是使用 3个卷积层+1个MaxPooling的组合。

卷积模块其中一个的代码如下：

```python
self.conv4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
self.conv4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
self.conv4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') 
```

##### 全连接层

在5个卷积层之后时全连接层。全连接层主要由一次卷积组成。同时与UNet类似，为了减弱过拟合的影响，这里进行Dropout归一化处理。

全连接层中的一层如下：

```python
self.fc7 = tf.keras.layers.Conv2D(4096, 1, activation='relu', padding='valid')
self.drop7 = tf.keras.layers.Dropout(0.5)
```

##### 上采样层

在全连接层之后是上采样层。每个上采样结合全连接层前的卷积模块与全连接层，在这两者的基础上进行反卷积，试图还原原图尺寸大小的特征提取信息。

上采样层的代码如下：

```python
self.socre_fr = tf.keras.layers.Conv2D(n_class, 1)
self.score_pool3 = tf.keras.layers.Conv2D(n_class, 1)
self.score_pool4 = tf.keras.layers.Conv2D(n_class, 1)

self.upscore2 = tf.keras.layers.Conv2DTranspose(
      n_class, 4, strides=2, padding='valid', use_bias=False)
self.upscore8 = tf.keras.layers.Conv2DTranspose(
      n_class, 16, strides=8, padding='valid', use_bias=False)
self.upscore_pool4 = tf.keras.layers.Conv2DTranspose(
      n_class, 4, strides=2, padding='valid', use_bias=False)
```

上采样层结合卷积模块和全连接层的部分代码：

```python
h = self.score_pool4(pool4 * 0.01)  
h = h[:, 5:5 + upscore2.shape[1], 5:5 + upscore2.shape[2], :] 
score_pool4c = h 

h = upscore2 + score_pool4c  
h = self.upscore_pool4(h)
upscore_pool4 = h
```

##### 输出层

在上采样完成之后，使用Softmax将结果映射为不同像素对于不同语义类别的概率并输出。

<div STYLE="page-break-after: always;"></div>

### 神经网络训练

在神经网络的训练部分，UNet与FCN的步骤与策略大致一致。

针对同一种网络模型，我们分别采取上文提及的ISBI 2012 Challenge数据集和DRIVE数据集进行训练与评估。同时，为了比较模型本身的表征能力和效用，针对同一个模型、同一组数据集，采用不同的周期进行训练。

以UNet为例，在搭建模型完毕和数据预处理完毕的基础上，进行训练。

调用Tensorflow.Keras 中Model的fit方法进行训练。需要的参数包括损失函数、优化器、迭代次数、验证集占比、训练回调等。

在UNet中，我们采用的损失函数二值交叉熵损失函数，使用的优化器为Adam。为了比较UNet在统一数据上不同训练周期的效果，我们分别针对ISBI 2012 Challenge数据集进行了迭代次数为20，1000，针对DRIVE数据集设定迭代次数为150和1000。每次训练从训练集中分割出比例为0.1的数据构建验证集对每一次训练迭代的结果进行评估。在训练时设置回调，每迭代一次保存一次模型权重至本地文件中。

训练部分的参数设置代码如下：

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=150)
model.save("UNetAlter.h5")
```

其中，由于FCN模型较为复杂，仅在FCN的训练过程中使用回调函数保存模型参数：

```python
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = tf.keras.callbacks.ModelCheckpoint("FCN8sAlter.h5", verbose=2, save_weights_only=True)
results = model.fit(X_train, test, epochs=1000, validation_split=0.1, callbacks=[callback], batch_size=2)
```

训练完成之后，获取本次训练的全部数据，将在实验结果部分对训练部分获得的数据进行分析。

<div STYLE="page-break-after: always;"></div>

## 实验结果

### 训练结果

正如上文所言，我们使用ISBI 2012 Challenge 和 DRIVE两个数数据集分别对UNet和FCN进行150、1000个循环的训练，得到了训练过程中以及训练完成后的神经网络的表现，现呈现如下。

#### UNet

在针对ISBI 2012 Challenge数据集的训练上，UNet的迭代150次的训练表现如下图：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000222.jpg"
         width="75%"
         height="75%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.5 UNet在ISBI 2012 Challenge, Epoch=150 训练数据
    </div> 
</center>

具体的分割表现如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000820.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.6 UNet在ISBI 2012 Challenge, Epoch=150 分割结果，左侧为原图像
    </div> 
</center>

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000826.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.7 UNet在ISBI 2012 Challenge, Epoch=150 分割结果，右侧为Ground Truth
    </div> 
</center>



在针对ISBI 2012 Challenge数据集的训练上，UNet的迭代1000次的训练表现如下图：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000316.jpg"
         width="75%"
         height="75%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.8 UNet在ISBI 2012 Challenge, Epoch=1000 训练数据
    </div> 
</center>
具体的分割表现如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000917.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.9 UNet在ISBI 2012 Challenge, Epoch=1000 分割结果，左侧为原图像
    </div> 
</center>

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607001001.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.10 UNet在ISBI 2012 Challenge, Epoch=1000 分割结果，右侧为Ground Truth
    </div> 
</center>






#### FCN

在针对ISBI 2012 Challenge数据集的训练上，FCN的迭代150次的训练表现如下图：
<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000355.jpg"
         width="75%"
         height="75%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.11 FCN在ISBI 2012 Challenge, Epoch=150 训练数据
    </div> 
</center>
具体的分割表现如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607001125.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.12 FCN在ISBI 2012 Challenge, Epoch=150 分割结果，右侧为Ground Truth
    </div> 
</center>



在针对ISBI 2012 Challenge数据集的训练上，FCN的迭代1000次的训练表现如下图：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607000417.jpg"
         width="75%"
         height="75%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.13 FCN在ISBI 2012 Challenge, Epoch=1000 训练数据
    </div> 
</center>

具体的分割表现如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 2px 2px 0 rgba(34,36,38,.12),0 2px 2px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607001155.png"
         width="50%"
         height="50%">    
    <br>    
        <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">
        Fig.14 FCN在ISBI 2012 Challenge, Epoch=150 分割结果，右侧为Ground Truth
    </div> 
</center>

<div STYLE="page-break-after: always;"></div>

### 结果分析

对于UNet和FCN在不同数据集上的表现分别进行横向对比和纵向对比，并对对比结果进行分析。

#### UNet的内部对比

在ISBI 2012 Challenge 数据集上训练1000 Epoch的UNet与训练150 Epoch的UNet表现对比如下图：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607083606.jpg"
         width="75%"
         height="75%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607083643.jpg"
         width="75%"
         height="75%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 15. UNet在ISBI 2012 Challenge不同Epoch的训练表现
    </div> 
    <br>
</center>

整体上可以看出随着训练迭代次数的增加，UNet在训练集上的表现，无论是Accuracy还是Loss都在向好的方向发展，Accuracy在不断增加而Loss不断降低。但在观察验证集上的表现时不难看出，无论是训练150 Epoch还是1000 Epoch，网络均在不同程度上出现了过拟合的现象。而过拟合现象在训练1000 Epoch的网络上更加严重。不难看出不同的迭代次数对于模型的表现具有较大的影响。因此我们使用DRIVE数据集进行进一步训练与分析。

在DRIVE数据上，UNet在训练集上的表现对比如下图：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607084216.jpg"
         width="75%"
         height="75%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607084350.jpg"
         width="75%"
         height="75%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 16. UNet在DRIVE不同Epoch的训练表现
    </div> 
    <br>
</center>



可以观察到，训练1000 Epoch的UNet在训练集上的Loss明显低于150 Epoch的UNet，正确率亦是如此。但是在验证集上的表现则没有这么好。同时，网络在测试集上的测试结果也可以证明这一点。



可以看出随着迭代次数的增加，UNet在1000 Epoch训练后期的过拟合导致UNet在验证集上的Loss逐渐增大，并在在验证集上的Accuracy在降低，这使得网络的性能受到不小影响。

具体的分割结果对比如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607085318.png"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607085311.png"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 17. UNet在DRIVE不同Epoch的测试结果对比一，上方为训练150 Epoch，下方为训练1000 Epoch
    </div> 
    <br>
</center>

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607085333.png"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607085338.png"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 18. UNet在DRIVE不同Epoch的测试结果对比二，上方为训练150 Epoch，下方为训练1000 Epoch
    </div> 
    <br>
</center>

不难看出，在部分细节部分的分割上，训练1000 Epoch的UNet做的并不比训练150 Epoch的UNet好很多，并且在眼球的边缘部分，训练1000 Epoch的UNet出现了较大的错误分割。这些都是在训练集上过拟合导致的网络在验证集上表现不佳。



#### FCN与UNet的对比

这里我们将分别对比训练不同循环周期的UNet与FCN在ISBI 2012 Challenge上的表现。

训练150 Epoch的UNet与FCN在训练集与验证集的Accuracy表现对比如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607090205.jpg"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607090230.jpg"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 19. UNet与FCN在ISBI 2012 Challenge训练150 Epoch Accuracy数据对比
    </div> 
    <br>
</center>



而在训练集与验证集上的Loss的对比如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607090704.jpg"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607090713.jpg"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 20. UNet与FCN在ISBI 2012 Challenge训练150 Epoch Loss数据对比
    </div> 
    <br>
</center>



在训练150 Epoch的UNet与FCN在训练集上的表现相差不大，但在验证集上的表现相差较大。UNet在验证集上表现出除了过拟合的情况，而FCN一直到训练结束都没有表现出过拟合的情况，其训练集与验证集上的Loss始终在不断降低。这种情况与网络本身的特性相关，UNet自身参数量较少，在训练到后期容易出现过拟合的情况，FCN参数较多，训练到收敛所需要的时间会更多，因此大量的训练对于FCN来说更加必要。

在训练1000 Epoch的UNet与FCN上，这一点将会更加显著。训练1000 Epoch的UNet与FCN的数据对比如下：

其中，在训练集与验证上的Loss数据如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607091509.jpg"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607091539.jpg"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 21. UNet与FCN在ISBI 2012 Challenge训练1000 Epoch Accuracy数据对比
    </div> 
    <br>
</center>



在训练集与验证机上Loss数据对比如下：

<center>    
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607103959.jpg"
         width="50%"
         height="50%">   
    <img style="border-radius: 0.3125em;    
                box-shadow: 0 10px 10px 0 rgba(34,36,38,.12),0 10px 10px 0 rgba(34,36,38,.08);"     
         src="https://gitee.com/TomGoh/img/raw/master/20210607091556.jpg"
         width="50%"
         height="50%">  
    <br>    
    <div style="color:orange;     display: inline-block;    color: #999;    padding: 2px;">
        Fig 22. UNet与FCN在ISBI 2012 Challenge训练1000 Epoch Loss数据对比
    </div> 
    <br>
</center>



可以看到，即使UNet在训练的中期就已出现过拟合，但无论是在训练集还是验证集上的Accuracy表现一直好于FCN。而在验证集上，FCN随着训练迭代次数的增加，也出现了过拟合的情况，并且在训练集上的Loss相较于UNet的大致稳定的趋势呈现出不断上升的态势，Accuracy也略有下降这足以说明FCN对于过拟合更加敏感，在过拟合后表现出的性能会更差。

<div STYLE="page-break-after: always;"></div>

## 实验总结

我们在这次实验中，主要完成了医学图像分割任务数据集的整理、医学图像分割模型的建立和对比研究两个任务。

我们整理了一些医学图像分割标注的数据集，列举出10个数据集及其图像类型、标注目标、数据量等信息。出于图像分割模型对比研究的需要，选择了ISBI 2012 Challenge数据集和DRIVE数据集供本次实验使用。

在医学图像分割模型的建立和对比研究的工作中，研究对象主要包括FCN、UNet及其改进模型。对于UNet，这一目前已经在图像语义分割领域广泛使用的模型，以及作为UNet基本框架来源的FCN模型，做了横向、纵向两个维度的比较分析。由于数据集的数据量少，我们采取了数据增强策略，获得了可供训练的数据量。

进行实验结果分析时，我们仅对FCN和UNet在150 Epoch、1000 Epoch训练的训练集、验证集的Accuracy、Loss两个指标进行可视化和分析。对于UNet而言，随着训练迭代次数的增加，在训练集上，UNet的Accuracy呈现不断提升的趋势，Loss呈现下降趋势；在验证集上，训练150 Epoch时，已经出现了过拟合的现象，而这一现象在训练了1000 Epoch的网络中更加显著。将UNet与FCN进行对比，我们发现FCN由于参数较多，导致收敛较慢；观察1000 Epoch的训练结果，我们发现UNet在训练集和验证集上的表现均优于FCN，两者在训练后期均出现了过拟合的情况，但是FCN在过拟合之后性能下降更为明显。

<div STYLE="page-break-after: always;"></div>

## 实验分工

吴昊泽负责

- 算法调研
- UNet代码实现
- FCN代码实现
- 实验报告撰写

徐梵负责

- 算法调研
- FCN代码实现
- 相关数据集收集
- 实验报告撰写

<div STYLE="page-break-after: always;"></div>

## 参考文献

[1]Liu, Xiangbin & Song, Liping & Liu, Shuai & Zhang, Yudong. (2021). A Review of Deep-Learning-Based Medical Image Segmentation Methods. Sustainability. 13. 1224. 10.3390/su13031224. 

[2]Meyer, F.. (1992). Color image segmentation. 303 - 306.

[3]Long J, Shelhamer E & Darrell T. Fully Convolutional Networks for Semantic Segmentation. Proceedings of the  IEEE conference on computer vision and pattern recognition. 2015:  3431-.

[4]Ronneberger O & Fischer P & Brox T.  U-net: Convolutional networks for biomedical image  segmentation. International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.

[5]Xiao, Chi & Liu, Jing & Chen, Xi & Han, Hua & Shu, Chang & Xie, Qiwei. (2018). Deep contextual residual network for electron microscopy image segmentation in connectomics. 378-381. 10.1109/ISBI.2018.8363597. 

[6]Fu, Huazhu & Cheng, Jun & Xu, Yanwu & Wong, Damon & Liu, Jiang & Cao, Xiaochun. (2018). Joint Optic Disc and Cup Segmentation Based on Multi-Label Deep Network and Polar Transformation. IEEE Transactions on Medical Imaging. PP. 10.1109/TMI.2018.2791488. 

[7]Quan, Tran Minh & Hildebrand, David & Jeong, Won-Ki. (2016). FusionNet: A Deep Fully Residual Convolutional Neural Network for Image Segmentation in Connectomics. Frontiers in Computer Science. 3. 10.3389/fcomp.2021.613981. 

[8]Staal, J., Abràmoff, M. D., Niemeijer, M., Viergever, M. A., & Van Ginneken, B. (2004). Ridge-based vessel segmentation in color images of the retina. IEEE transactions on medical imaging, 23(4), 501-509.

[9]Bilic, P., Christ, P. F., Vorontsov, E., Chlebus, G., Chen, H., Dou, Q., ... & Menze, B. H. (2019). The liver tumor segmentation benchmark (lits). arXiv preprint arXiv:1901.04056.

[10]Caicedo, J. C., Goodman, A., Karhohs, K. W., Cimini, B. A., Ackerman, J., Haghighi, M., Heng, C., Becker, T., Doan, M., McQuin, C., Rohban, M., Singh, S., & Carpenter, A. E. (2019). Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nature methods, 16(12), 1247–1253. 

<div STYLE="page-break-after: always;"></div>

