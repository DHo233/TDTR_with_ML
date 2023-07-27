import numpy as np
# from __future__ import absolute_import, division, print_function
import torch
import tensorflow
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

sample_distance1 = 5
sample_distance2 = 5
coef1_range = np.linspace(10e-4, 50e-4,
                          sample_distance1)
coef2_range = np.linspace(1.35e6, 1.95e6, sample_distance2)
coef_list = []
for i in coef1_range:
    for j in coef2_range:
        single_coef = np.array([i, j])
        coef_list.append(single_coef)



def Bn(gn, gn_1, un, Ln, Bn1_plus, Bn1_minus):
    pos11 = torch.exp(-un * Ln) * (gn + gn_1)
    pos12 = torch.exp(-un * Ln) * (gn - gn_1)
    pos21 = torch.exp(un * Ln) * (gn - gn_1)
    pos22 = torch.exp(un * Ln) * (gn + gn_1)
    Bn_plus = (1 / 2 / gn) * (Bn1_plus * pos11 + Bn1_minus * pos12)
    Bn_minus = (1 / 2 / gn) * (Bn1_plus * pos21 + Bn1_minus * pos22)
    return Bn_plus, Bn_minus



def dt_func(K, w0, w1, pace, A, g):
    return 2 * torch.pi * g * A * torch.exp(-torch.pi * torch.pi * K * K * (w0 ** 2 + w1 ** 2) / 2) * K * pace



def multy_dt_func(K_total, w0, w1, pace, A, G_total, colidxs):
    ret = torch.zeros(K_total.shape[0], dtype=torch.complex64)
    for c_idx in colidxs:
        temp_array = dt_func(K_total[:, int(c_idx)], w0, w1, pace, A, G_total[:, int(c_idx)])
        if (c_idx == colidxs[0]):
            ret = temp_array
        else:
            ret = torch.row_stack((ret, temp_array))
    return ret



def sum2(input_array):
    colnum = input_array.shape[1]
    rownum = input_array.shape[0]
    ret_array = torch.zeros(colnum, dtype=torch.complex64)

    for colidx in range(colnum):
        for rowidx in range(rownum):
            ret_array[colidx] += input_array[rowidx][colidx]
    return ret_array



def u_func(K, Q):
    return torch.sqrt(4 * torch.pi * torch.pi * K * K + Q * Q)



def two_materials_simple_2layer(w, coef0, coef1):
    global current_list
    coef = [coef0, coef1, coef_list[0][0], coef_list[0][1]]             #改过
    w0 = coef[1]
    w1 = w0
    # ddd=w0.shape
    # w0=w0.squeeze(0)
    # w1=w0
    c=coef[0]            #改了 将维度改了 0->1，这样后续才可以用cat将coef[0]加入tensor，避免整体使用torch.tensor
    c=c.expand(1)

    # thermal_conductivity = torch.tensor([230, coef[2], coef[0]])
    # volumetric_heat_capacity = torch.tensor([19300 * 130, 1, coef[3]])
    # thickness = torch.tensor([70e-9, 1e-10, 500e-6])
    # thermal_conductivity = [torch.tensor(230), torch.tensor(coef[2]), coef[0]]
    # volumetric_heat_capacity = [torch.tensor(19300 * 130), torch.tensor(1), torch.tensor(coef[3])]
    # thickness = torch.tensor([70e-9, 1e-10, 500e-6])
    thermal_conductivity = torch.tensor([230,coef[2]])                       #换方法创建tensor
    # thermal_conductivity = torch.cat((thermal_conductivity,coef[0]))
    thermal_conductivity = torch.cat((thermal_conductivity,c))
    volumetric_heat_capacity = torch.tensor([19300 * 130, 1, coef[3]])
    thickness = torch.tensor([70e-9, 1e-10, 500e-6])


    D = thermal_conductivity / volumetric_heat_capacity
    # D = [torch.tensor(230/19300/130),torch.tensor(coef[2]),coef[0]/torch.tensor(coef[3])]
    A = 0.002
    time_steps = 1000

    simpson_steps = 501
    up = 2 / torch.sqrt(w0 ** 2 + w1 ** 2)
    pace = up / simpson_steps
    m = torch.linspace(0, float(up), simpson_steps + 1)

    even_indices = torch.linspace(2, 500, 250)
    odd_indices = torch.linspace(3, 499, 249)

    k = m
    number_of_layers = len(thermal_conductivity)
    g = []
    B_plus = []
    B_minus = []
    K = []
    Q = []
    u_waste = []
    for n in range(0, number_of_layers):
        # K_term, Q_term = np.meshgrid(k, np.sqrt(w * (0 + 1j) / D[n]))
        # K_term = torch.tensor(K_term)
        # Q_term = torch.tensor(Q_term)\
        ttt=k
        qqq=torch.sqrt(w * (0 + 1j) / D[n])


        ttt = ttt.to(torch.complex64)
        qqq = qqq.to(torch.complex64)
        # print(ttt.dtype)
        # print(qqq.dtype)

        Q_term, K_term = torch.meshgrid(qqq,ttt)            #用torch.meshgrid替换了numpy 但这两个函数有区别，输入位置是相反的 此外还要求两个参数数据类型相同
        # K_term = torch.tensor(K_term)
        # Q_term = torch.tensor(Q_term)

        K.append(K_term)
        Q.append(Q_term)
        u = u_func(K_term, Q_term)
        u_waste.append(u)
        g.append(u * thermal_conductivity[n])
        size = (np.size(g[0], 0), np.size(g[0], 1))
        B_plus.append(torch.zeros(size))
        B_minus.append(torch.ones(size))

    for n in range(2, number_of_layers + 1):
        B_plus[number_of_layers - n], B_minus[number_of_layers - n] = Bn(g[number_of_layers - n],
                                                                         g[number_of_layers - n + 1],
                                                                         u_waste[number_of_layers - n],
                                                                         thickness[number_of_layers - n],
                                                                         B_plus[number_of_layers - n + 1],
                                                                         B_minus[number_of_layers - n + 1])
    G1 = (B_plus[0] + B_minus[0]) / (B_minus[0] - B_plus[0]) / g[0];

    delT_2Dsimp = (1 / 3) * (dt_func(K[0][:, 0], w0, w1, pace, A, G1[:, 0]) + 2 * sum2(
        multy_dt_func(K[0][:, :], w0, w1, pace, A, G1[:, :], odd_indices - 1)) + 4 * sum2(
        multy_dt_func(K[0][:, :], w0, w1, pace, A, G1[:, :], even_indices - 1)) + dt_func(K[0][:, simpson_steps - 1],
                                                                                          w0, w1, pace, A,
                                                                                          G1[:, simpson_steps - 1]));
    phase = (torch.angle(delT_2Dsimp) * 180 / torch.pi)
    # phase = phase * 5
    # amp = abs(delT_2Dsimp);
    # norm_amp = amp / max(amp);
    # phase_amp = [1e0 * phase.T, 1e0 * norm_amp.T];
    # Mamp = amp;
    # phase = np.squeeze(phase)
    # phase = phase.detach().numpy()

    return phase.squeeze(-1)

# a = torch.tensor(6.8112e+04)
# b = torch.tensor(2.3613e+01)
# c = torch.tensor(4.7600e-07)
# # b = torch.tensor(100)
# # c = torch.tensor(2.8e-6)
# print(two_materials_simple_2layer(a * 2 * torch.pi, b, c))
# print("hh")

# a = torch.tensor(11e6)
# b = torch.tensor(100)
# c = torch.tensor(2.8e-6)
# print(two_materials_simple_2layer(a * 2 * torch.pi, b, c))

# freq = torch.logspace(np.log10(60e3), np.log10(11e6), 100)
# phase = two_materials_simple_2layer(freq * 2 * np.pi, b, c)
# print(freq)
# print(phase)
# plt.plot(freq, phase)
# print("h")


# current_list = torch.tensor([1.00e-3, 1.8e+06])
# # test
# x=1.0
# x=torch.tensor(x)
# print(two_materials_simple_2layer(x,x,x))


# MLP = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn. Linear(32, 2))
class HeHe(nn.Module):
    def __init__(self):
        super(HeHe, self).__init__()
        # self.MLP = nn.Sequential(nn.Linear(2, 64, bias=True),
        #                             nn.ReLU(),
        #                             nn.Linear(64, 2))
        # self.MLP = nn.Sequential(nn.Linear(2, 200, bias=True),
        #                             nn.ReLU(),
        #                             nn.Linear(200, 80),
        #                             nn.ReLU(),
        #                             nn.Linear(80, 10),
        #                             nn.ReLU(),
        #                             nn.Linear(10, 2))
        # self.MLP = nn.Sequential(
        #     nn.Linear(2, 150),
        #     nn.Tanh(),
        #     nn.Linear(150, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 2),
        # )
        # self.MLP = nn.Sequential(nn.Linear(2, 200, bias=True),
        #                             nn.Sigmoid(),
        #                             nn.Linear(200, 80),
        #                             nn.Sigmoid(),
        #                             nn.Linear(80, 10),
        #                             nn.Sigmoid(),
        #                             nn.Linear(10, 2))
        self.MLP = nn.Sequential(nn.Linear(2, 200, bias=True),
                                  nn.Sigmoid(),
                                  nn.Linear(200, 200, bias=True),
                                  nn.Sigmoid(),
                                  nn.Linear(200, 200, bias=True),
                                  nn.Sigmoid(),
                                  nn.Linear(200, 200, bias=True),
                                  nn.Sigmoid(),
                                  nn.Linear(200, 4))
        # self.MLP = nn.Sequential(nn.Linear(2, 10, bias=True),
        #                           nn.ReLU(),
        #                           nn.Linear(10, 10, bias=True),
        #                           nn.ReLU(),
        #                           nn.Linear(10, 10, bias=True),
        #                           nn.ReLU(),
        #                           nn.Linear(10, 10, bias=True),
        #                           nn.ReLU(),
        #                           nn.Linear(10, 2))
    def forward(self, inputs):
        x = self.MLP(inputs)
        return x

class norm:
    def __init__(self):
        print("normalization")

    def main(self,dataset):
        leng,wid=dataset.shape
        for j in range(wid):
            row=dataset[:,j]
            max=torch.max(row)
            min_n=torch.min(row)
            denominator_n=max-min_n
            #print(min_n)
            if j==0:
                self.min=torch.tensor([min_n])
                self.denominator=torch.tensor([denominator_n ])

            else:
                self.min=torch.cat((self.min,torch.tensor([min_n])))
                self.denominator=torch.cat((self.denominator,torch.tensor([denominator_n])))

            for i in range(leng):
                numerator=dataset[i,j]-min_n
                dataset[i,j]=numerator/denominator_n
            #print(self.min,dataset)
        return dataset

    def reverse(self,dataset):
        leng, wid = dataset.shape
        for j in range(wid):
            min_n = self.min[j]
            denominator_n = self.denominator[j]
            for i in range(leng):
                dataset[i, j] = dataset[i, j]* denominator_n + min_n
        return dataset
    
    
#test
# print(two_materials_simple_2layer(torch.tensor(191026),coef1,coef2))
# zxzzxx=8
#修改
def synthetic_data(a, b, num_examples):
    freq = torch.logspace(np.log10(60e3), np.log10(11e6), num_examples)
    
    
    phase = two_materials_simple_2layer(freq * 2 * np.pi, a, b)
    freq = freq.unsqueeze(1)
    phase = phase.unsqueeze(1)
    
    # phase_noise = torch.randn(1000, 1)*0.005 + 0.01
    # phase = phase * (1 + phase_noise)                                           #添加1%左右的噪声
    
    matrix = torch.cat((freq, phase), -1)
    matrix = norm.main(matrix)
    freq = matrix[:,0]
    phase = matrix[:,1]
    return freq, phase


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# print(freq, phase)


# plt.scatter(freq, phase)


def combine(a, b):
    a = a.numpy()
    b = b.numpy()
    c = torch.tensor([[a[0], b[0]]])
    for i in range(1, len(a)):
        d = torch.tensor([[a[i], b[i]]])
        torch.t(d)
        c = torch.cat((c, d), 0)
    return c

hehe = HeHe()
norm = norm()


#这里做修改，因为开头修改后这里要是tensor类型才可以跑
coef1 = torch.tensor(100)                                                       #确定两个要预测的参数的正确值并生成数据集
coef2 = torch.tensor(2.8e-6)
# coef1 = torch.tensor([100])
# coef2 = torch.tensor([2.8e-6])
coef1=coef1.to(torch.float64)
coef2 = coef2.to(torch.float64)

freq, phase = synthetic_data(coef1, coef2, 1000)                                 #这里是生成数据集

scale_coef1 = 100
scale_coef2 = 10e-6      
noise_coef1 = 10e-2
noise_coef2 = 10e-2



batch_size = 10 
data_iter = load_array((freq, phase), batch_size)
next(iter(data_iter))


loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(hehe.parameters(), lr=0.00015)
cnt = 0


hehe.train()
num_epochs = 10
for epoch in range(num_epochs):
    for X, Y in data_iter:
        x = X.unsqueeze(1)
        y = Y.unsqueeze(1)                                                      #x, y存储原始的数据;X, Y存储normalization之后的数据
        origin = torch.cat((x, y), -1)
        origin = norm.reverse(origin)  
        # print(origin)
        x = origin[:,0]
        y = origin[:,1]
        c = combine(X, Y)                                                       # 将数据集中的freq, phase分为两两一组，作为神经网络的输入
        # print(origin)
        # print(c)
        coefx = hehe(c)                                                         # 神经网络返回coefx（也就是要预测的10组2个参数）
        # coefx_scale = coefx.clone()
        # print(coefx)
        coefx[:,0] = torch.abs( coefx[:,0] * scale_coef1 )                      #用scale确保数量级不会出错
        coefx[:,2] = torch.abs( coefx[:,2] * scale_coef2 )
        
        coefx[:,1] = coefx[:,1] * noise_coef1                      #2.28:用noise_coef矫正噪声的数量级 
        coefx[:,3] = coefx[:,3] * noise_coef2
        print(coefx)
        loss = 0
        for i in range(batch_size):
            mark1 = y[i].to(torch.float32)
            # mark3 = mark1.clone().detach().requires_grad_(True)
            # print(x[i])
            loss1 = loss_fn(mark1, 
                            two_materials_simple_2layer(x[i] * 2 * torch.pi, coefx[0][0] * (1 - coefx[0][1]), coefx[0][2] * (1 - coefx[0][3]) ) )
            loss = loss + loss1
            # print(loss)
        loss = loss / batch_size
        print("loss为",loss)
        # if(loss <= 0.1): sys.exit()
        # loss = torch.tensor([loss], dtype=torch.double, requires_grad=True)  #修改
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        print("第",cnt,"次训练")
        print(origin)
    #     print(c)
    #     coefx = hehe(c)
    #     print(coefx)
    #     coefx[:,0] = coefx[:,0] * scale_coef1
    #     coefx[:,1] = coefx[:,1] * scale_coef2
    #     print(coefx)
    #     loss = 0
    #     for i in range(batch_size):
    #         mark1=y[i].to(torch.float32)
    #         mark3=mark1.clone().detach().requires_grad_(True)
    #         # print( two_materials_simple_2layer(x[i], coefx[0][0], coefx[0][1]) )
    #         # print(mark3)
    #         loss1 = loss_fn(mark3, two_materials_simple_2layer(x[i], coefx[0][0], coefx[0][1]))
    #         loss = loss + loss1
    #         # print(loss)
    #     loss = loss / batch_size
    #     print(loss)
    # print(f'epoch{epoch + 1}, loss{loss: f}')
    # print(coefx)


