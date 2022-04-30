import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math_util
from perceptual_loss import vgg
from perceptual_loss import util
from math_util import  witness_g,witness_mix_g,get_squared_dist,jacobian_squared_frobenius_norm

'''
            1.tf.multiply（）两个矩阵中对应元素各自相乘

格式: tf.multiply(x, y, name=None) 
参数: 
x: 一个类型为:half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128的张量。 
y: 一个类型跟张量x相同的张量。  
返回值： x * y element-wise.  
注意： 
（1）multiply这个函数实现的是元素级别的相乘，也就是两个相乘的数元素各自相乘，而不是矩阵乘法，注意和tf.matmul区别。 
（2）两个相乘的数必须有相同的数据类型，不然就会报错。
2.tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
            '''


'''
EBGAN idea, add noise to G and D
'''

def pullaway_loss(embeddings):
    norm = embeddings.norm(2, 3).norm(2, 2).norm(2, 1).repeat(1, embeddings.size()[1],
                                                              embeddings.size()[2], embeddings.size()[3])
    normalized_embeddings = embeddings / norm

from pytorch_msssim import SSIM,MSSSIM

def log_odds(p):
    p = torch.clamp(p.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(p / (1 - p))


class MaxOut(nn.Module):
    def __init__(self, k=2):
        """ MaxOut nonlinearity.

        Args:
          k: Number of linear pieces in the MaxOut opeartion. Default: 2
        """
        super().__init__()

        self.k = k

    def forward(self, input):
        output_dim = input.size(1) // self.k
        input = input.view(input.size(0), output_dim, self.k, input.size(2), input.size(3))
        output, _ = input.max(dim=2)
        return output

#
# class DeterministicConditional2(nn.Module):
#     def __init__(self, mapping, shift=None):
#         """ A deterministic conditional mapping. Used as an encoder or a generator.
#         Args:
#           mapping: An nn.Sequential module that maps the input to the output deterministically.
#           shift: A pixel-wise shift added to the output of mapping. Default: None
#         """
#         super().__init__()
#
#         self.mapping = mapping
#         self.shift = shift
#
#     def set_shift(self, value):
#         if self.shift is None:
#             return
#         assert list(self.shift.data.size()) == list(value.size())
#         self.shift.data = value
#
#     def forward(self, input):
#         # print("input:")
#         # print(input.size())
#         # print(self.mapping)
#         output = self.mapping(input)
#         if self.shift is not None:
#             output = output + self.shift
#         return output

class DeterministicConditional(nn.Module):
    def __init__(self, mapping, shift=None):
        """ A deterministic conditional mapping. Used as an encoder or a generator.
        Args:
          mapping: An nn.Sequential module that maps the input to the output deterministically.
          shift: A pixel-wise shift added to the output of mapping. Default: None
        """
        super().__init__()

        self.mapping = mapping
        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input):
        # print("input:")
        # print(input.size())
        # print(self.mapping)
        output = self.mapping(input)
        if self.shift is not None:
            output = output + self.shift
        return output


class GaussianConditional(nn.Module):
    def __init__(self, mapping, shift=None):
        """ A Gaussian conditional mapping. Used as an encoder or a generator.
        Args:
          mapping: An nn.Sequential module that maps the input to the parameters of the Gaussian.
          shift: A pixel-wise shift added to the output of mapping. Default: None
        """
        super().__init__()

        self.mapping = mapping
        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, input):
        params = self.mapping(input)
        nlatent = params.size(1) // 2
        mu, log_sigma = params[:, :nlatent], params[:, nlatent:]
        sigma = log_sigma.exp()
        eps = torch.randn(mu.size()).to(input.device)
        output = mu + sigma * eps
        if self.shift is not None:
            output = output + self.shift
        return output

# , z_mapping
class JointCritic(nn.Module):
    def __init__(self, x_mapping, joint_mapping):
        """ A joint Wasserstein critic function.
        Args:
          x_mapping: An nn.Sequential module that processes x.
          z_mapping: An nn.Sequential module that processes z.
          joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
        """
        super().__init__()

        self.x_net = x_mapping
        # self.z_net = z_mapping
        self.joint_net = joint_mapping
        # self.drop1 = torch.nn.Dropout()
        # self.mmd_x = mmd_x
        # self.mmd_z = mmd_z

    def get_x_net_parameters(self):
        return self.x_net.parameters()

    def get_joint_net_parameters(self):
        return self.joint_net.parameters()

    # def get_mmdx_parameters(self):
    #     return self.mm

    def forward(self, x):
        x_out = self.x_net(x)
        # torch.cat((x_out, z_out), dim=1)
        joint_input = x_out
        output = self.joint_net(joint_input)

        return output

class WALI(nn.Module):
    # E,
    # ,MMDX,MMDZ
    def __init__(self,  G, C,rep_weights=[0.0,-1.0],mmd_g_scale=0.1,loss_type='msssim',sigma=[1,np.sqrt(2),2,np.sqrt(8),4],window_size=11,size_average=True,val_range=2,channel=3,l1=True,l2=False,pads=False,loss_type2=None):
        """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.
        Args:
          E: Encoder p(z|x).
          G: Generator p(x|z).
          C: Wasserstein critic function f(x, z).
        """
        super().__init__()

        # self.E = E
        self.G = G
        self.C = C

        self.rep_weights = rep_weights
        self.penalty_weight = mmd_g_scale
        self.loss_type = loss_type
        self.sigma = sigma
        self.window_size=window_size
        self.size_average=size_average
        self.val_range=val_range
        self.channel = channel
        self.l1=l1
        self.pads = pads
        self.l2 = l2
        self.sm = None
        self.percept = None
        if loss_type.strip() == "per".strip():
            self.percept = vgg.Vgg16()
            for per_param in self.percept.parameters():
                per_param.requires_grad = False

        print(loss_type.strip() == 'msssim'.strip())
        if loss_type.strip() == 'ssim'.strip():
            ssim = SSIM(window_size=window_size, size_average=size_average)
            self.sm = ssim
        if loss_type.strip() == 'msssim'.strip():
            msssim = MSSSIM(window_size=window_size, size_average=size_average, channel=self.channel)
            self.sm = msssim
        # size_average = True, val_range = 2, channel = 3, l1 = True, pads = False


    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def generate(self, z,var_beta=-1,clip_beta=-1):
        return self.G(z)

    # def reconstruct(self, x):
    #     return self.generate(self.encode(x))

    # def reconder(self,z):
    #     return self.encode(self.generate(z))

    def get_G(self):
        return self.G

    def get_C(self):
        return self.C


    def criticize(self, x, x_tilde):
        # print(x.size())
        input_x = torch.cat((x, x_tilde), dim=0)
        # input_z = torch.cat((z_hat, z), dim=0)
        # ,_,_
        output = self.C(input_x)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def percept_criticize_x(self,x,x_recon):
        #     out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        if self.percept is not None:
            out_x = self.percept(x)
            out_x_recon = self.percept(x_recon)
            feature_loss = F.mse_loss(input=out_x_recon[2],target=out_x[2],reduction="mean")

            style_x = [util.gram(xi) for xi in out_x]
            style_recon = [util.gram(xi) for xi in out_x_recon]

            style_loss = F.mse_loss(input=style_recon[0],target=style_x[0],reduction="mean")

            for i in range(1,4):
                style_loss += F.mse_loss(input=style_recon[i],target=style_x[i],reduction="mean")

            losses = feature_loss + style_loss
            return losses,feature_loss,style_loss
        else:
            return None,None,None


    def calculate_grad_penalty(self, x, x_tilde):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        # intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        # intp_z.requires_grad = True
        # ,_,_
        tmp_C = self.C(intp_x)
        C_intp_loss = tmp_C.sum()
        # , intp_z
        # , grads_z
        # , grads[1].view(bsize, -1)
        grads = autograd.grad(C_intp_loss, (intp_x), retain_graph=True, create_graph=True)
        # print(grads)
        grads_x = grads[0].view(bsize, -1)
        # grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads_x.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x, z, lamb=10,beta1=0.01,beta2=0.01,beta3=0.03,gan=0,loss_type='raw',var_beta=-1,clip_beta=-1,methods=0,l1=True,l2=False,val_range=2,normalize="relu",pads=False,ssm_alpha=0.84):
        batch_sizes = x.size()[0]

        if methods > 0:
            x_tilde =  self.generate(z,var_beta=var_beta,clip_beta=clip_beta)
        else:
            x_tilde = self.generate(z)

        # z_hat,
        # , z
        if gan == 0:
            self.loss_type = loss_type
            data_preds, sample_preds = self.criticize(x, x_tilde)
            EG_loss = torch.mean(data_preds - sample_preds)
            #  z_hat.data,
            # , z.data
            C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, x_tilde.data)
            # C_loss = C_loss.mean()
            # print(RECON_X_loss*beta1/RECON_X_loss)
            if self.loss_type == 'raw':
                # C_loss = torch.mean(C_loss)
                return C_loss,EG_loss

            else:
                return C_loss,EG_loss
            # elif self.loss_type == 'mmd0':
        else:
            self.loss_type = loss_type
            data_preds, sample_preds = self.criticize(x,x_tilde)
            EG_loss = torch.mean(data_preds - sample_preds)
            C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, x_tilde.data)

            # C_loss = C_loss.mean()
            # print(RECON_X_loss*beta1/RECON_X_loss)
            if self.loss_type == 'raw':
                return C_loss, EG_loss
            else:
                return C_loss, EG_loss
