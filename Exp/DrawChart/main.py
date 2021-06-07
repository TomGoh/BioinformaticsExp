from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

U1000_acc_list = []
U1000_loss_list = []
U1000_val_acc_list = []
U1000_val_loss_list = []
U150_acc_list = []
U150_loss_list = []
U150_val_acc_list = []
U150_val_loss_list = []
file = open(file='UNet_1000_DRIVE.txt', mode='r')
contents = file.readlines()
for content in contents:
    if len(content) > 128:
        U1000_loss_list.append(float(content[62:69]))
        U1000_acc_list.append(float(content[82:87]))
        U1000_val_loss_list.append(float(content[101:106]))
        U1000_val_acc_list.append(float(content[123:-1]))

file = open(file='UNet_150_DRIVE.txt', mode='r')
contents = file.readlines()
for content in contents:
    if len(content) > 128:
        U150_loss_list.append(float(content[62:69]))
        U150_acc_list.append(float(content[82:87]))
        U150_val_loss_list.append(float(content[101:106]))
        U150_val_acc_list.append(float(content[123:-1]))
index = list(range(0, 1000))
index1 = list(range(0, 150))
# for a, b, c, d in zip(loss_list, acc_list, val_loss_list, val_acc_list):
#     print('Loss,Acc,Val_Loss,Val_Acc', a, b, c, d)
# print(len(loss_list))

plt.figure(figsize=(16, 9), dpi=100)
plt.plot(index, U1000_val_acc_list, label='1000 Epoch Accuracy')
plt.plot(index1, U150_val_acc_list, label='150 Epoch Accuracy')
plt.plot(index, U1000_val_loss_list, label='1000 Epoch Loss')
plt.plot(index1, U150_val_loss_list, label='150 Epoch Loss')
plt.title("Validation Data on DRIVE by UNet", fontsize=30)
x_major_locator = MultipleLocator(50)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(0.1)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数
plt.xlim(-5, 520)
# 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0, 1)
plt.legend()
plt.savefig('UNet_Compare_Epoch_DRIVE_VALI.jpg')
plt.show()
