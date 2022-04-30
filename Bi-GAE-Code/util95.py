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


class JointCritic(nn.Module):
    def __init__(self, x_mapping, z_mapping, joint_mapping):
        """ A joint Wasserstein critic function.
        Args:
          x_mapping: An nn.Sequential module that processes x.
          z_mapping: An nn.Sequential module that processes z.
          joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
        """
        super().__init__()

        self.x_net = x_mapping
        self.z_net = z_mapping
        self.joint_net = joint_mapping
        # self.drop1 = torch.nn.Dropout()
        # self.mmd_x = mmd_x
        # self.mmd_z = mmd_z

    def get_x_net_parameters(self):
        return self.x_net.parameters()

    def get_z_net_parameters(self):
        return self.z_net.parameters()

    def get_joint_net_parameters(self):
        return self.joint_net.parameters()

    # def get_mmdx_parameters(self):
    #     return self.mm

    def forward(self, x, z):
        if z is not None and x is not None:
            assert x.size(0) == z.size(0)
            x_out = self.x_net(x)
            z_out = self.z_net(z)
            joint_input = torch.cat((x_out, z_out), dim=1)
            output = self.joint_net(joint_input)

            return output, x_out, z_out
        elif z is None:
            x_out = self.x_net(x)
            return x_out
        elif x is None:
            z_out = self.z_net(z)
            return z_out

class MMD_NET(nn.Module):
    def __init__(self,mmds):
        super(MMD_NET, self).__init__()

        self.mmds = mmds

    def forward(self,x_out):
        s_x = self.mmds(x_out)
        # s_z = self.mmd_z(z_out)

        return s_x


class WALI(nn.Module):
    def __init__(self, E, G, C,MMDX,MMDZ,rep_weights=[0.0,-1.0],mmd_g_scale=0.1,loss_type='msssim',sigma=[1,np.sqrt(2),2,np.sqrt(8),4],window_size=11,size_average=True,val_range=2,channel=3,l1=True,l2=False,pads=False,loss_type2=None):

        super().__init__()

        self.E = E
        self.G = G
        self.C = C
        self.MMDX = MMDX
        self.MMDZ = MMDZ
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
        if "per".strip() in loss_type.strip():
            print("load vgg16 to compute perceptual loss")
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



    def get_encoder_parameters(self):
        return self.E.parameters()

    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def get_mmdx_parameters(self):
        return self.MMDX.parameters()

    def get_mmdz_parameters(self):
        return self.MMDZ.parameters()

    def encode(self, x):
        return self.E(x)

    def generate(self, z,var_beta=-1,clip_beta=-1):
        return self.G(z)

    def reconstruct(self, x):
        return self.generate(self.encode(x))

    def reconder(self,z):
        return self.encode(self.generate(z))

    def get_E(self):
        return self.E

    def get_G(self):
        return self.G

    def get_C(self):
        return self.C

    def get_MMDX(self):
        return self.MMDX

    def get_MMDZ(self):
        return self.MMDZ

    def criticize(self, x, z_hat, x_tilde, z):
        # print(x.size())
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        output,_,_ = self.C(input_x, input_z)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def mmd_criticize_x(self,x,x_recon):
        input_x =  torch.cat((x,x_recon),dim=0)
        output = self.C(input_x,None)
        output = self.MMDX(output)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]

        return data_preds,sample_preds

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

    def mmd_criticize_z(self,z_hat,z_recon):
        input_z = torch.cat((z_hat, z_recon), dim=0)
        output = self.C(None,input_z)
        output = self.MMDZ(output)
        data_preds, sample_preds = output[:z_hat.size(0)], output[z_hat.size(0):]
        return data_preds, sample_preds


    def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        intp_z.requires_grad = True
        tmp_C,_,_ = self.C(intp_x, intp_z)
        C_intp_loss = tmp_C.sum()
        grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
        grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def mmd_gradient_penalty(self,x, x_recon, mode='fixed_g_gp',aim=0):
        """ This function calculates the gradient penalty used in mmd-gan
        This code is inspired by the code for the following paper:
        Binkowski M., Sutherland D.J., Arbel M., and Gretton A.
        Demystifying MMD GANs. ICLR 2018
        :param x: real images
        :param x_gen: generated images
        :param s_x: scores of real images
        :param s_gen: scores of generated images
        :param batch_size:
        :param mode:
        :return:
        """
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_recon

        if aim == 0:
            # x_out,x_recon_out = self.C(x,None)
            # s_x = self.MMDX(x)
            # s_recon = self.MMDX(x_recon)
            s_x,s_recon = self.mmd_criticize_x(x,x_recon)
            intp_out = self.C(intp_x,None)
            s_intp_x = self.MMDX(intp_out)
        else:
            s_x, s_recon = self.mmd_criticize_z(x, x_recon)
            intp_out = self.C(None,intp_x)
            s_intp_x = self.MMDZ(intp_out)
            # print(s_intp_x.size())
            # s_x = self.MMDZ(x)
            # s_recon = self.MMDZ * (x_recon)
            # s_intp_x = self.MMDZ(intp_x)

        # s_intp_x.requires_grad = True
        s_x = torch.squeeze(s_x)
        s_recon = torch.squeeze(s_recon)
        s_intp_x = torch.squeeze(s_intp_x)
        dist_zx = get_squared_dist(s_intp_x, s_x, mode='xy', name='dist_zx')
        dist_zy = get_squared_dist(s_intp_x, s_recon, mode='xy', name='dist_zy')
        if mode == 'fixed_g_gp':
            witness = witness_mix_g(
                dist_zx, dist_zy, sigma=[1.0, np.sqrt(2.0), 2.0, np.sqrt(8.0), 4.0],
                name='witness')
        # elif mode == 'fixed_t_gp':
        #     witness = witness_mix_t(
        #         dist_zx, dist_zy, alpha=[0.25, 0.5, 0.9, 2.0, 25.0], beta=2.0,
        #         name='witness', do_summary=self.do_summary)
        elif mode in {'rep_gp', 'rmb_gp'}:
            witness = witness_g(
                dist_zx, dist_zy, sigma=1.0, name='witness')
        else:
            raise NotImplementedError('gradient penalty: {} not implemented'.format(mode))
        # print(witness.size())

        grads = autograd.grad(witness.sum(),s_intp_x,retain_graph=True, create_graph=True)
        gradss = grads[0]
        # print(gradss.size())
        g_x_hat = torch.reshape(gradss,[bsize,-1])
        # g_x_hat = .reshape(
        #     tf.gradients(witness, s_intp_x, name='gradient_x_hat')[0],
        #     [bsize, -1])
        loss_grad_norm = ((g_x_hat.norm(2,dim=1)-1)**2).mean()
            # tf.reduce_mean(
            # tf.square(tf.norm(g_x_hat, ord=2, axis=1) - 1))
    #     grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return loss_grad_norm

    def mmd_gradient_scale(self, x, s_x):
        """ This function calculates the gradient penalty used in scaled mmd-gan.
        This code is inspired by the following paper:
        Arbel M., Sutherland, D.J., Binkowski M., and Gretton A.
        On gradient regularizers for MMD GANs Michael. NIPS, 2018.
        :param x:
        :param s_x:
        :return:
        """
        jaco_sfn = jacobian_squared_frobenius_norm(s_x, x)
        dis_loss_scale = 1.0 / (self.penalty_weight * torch.mean(jaco_sfn) + 1.0)

        return dis_loss_scale

    def forward(self, x, z, lamb=10,beta1=0.01,beta2=0.01,beta3=0.03,gan=0,loss_type='raw',var_beta=-1,clip_beta=-1,methods=0,l1=True,l2=False,val_range=2,normalize="relu",pads=False,ssm_alpha=0.84):
        batch_sizes = x.size()[0]
        # print(x.size())
        # print(self.encode(x).size())
        # print("\tIn Model: input size", x.size(),
        #       "latent size", z.size())


        if methods > 0:
            z_hat, x_tilde = self.encode(x), self.generate(z,var_beta=var_beta,clip_beta=clip_beta)
        else:
            z_hat, x_tilde = self.encode(x), self.generate(z)


        if gan == 0:
            self.loss_type = loss_type
            data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
            EG_loss = torch.mean(data_preds - sample_preds)
            C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
            # C_loss = C_loss.mean()
            # print(RECON_X_loss*beta1/RECON_X_loss)
            if self.loss_type == 'raw':
                # C_loss = torch.mean(C_loss)
                return C_loss,EG_loss
            if self.loss_type == 'mse0':
                x_tilde_copy = x_tilde.clone().detach()
                z_recon = self.encode(x_tilde_copy)
                z_hat_copy = z_hat.clone().detach()
                x_recon = self.generate(z_hat_copy)
                RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='sum') / batch_sizes
                RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes

                EG_loss2 = EG_loss + beta1 * RECON_X_loss + beta2 * RECON_Z_loss
                return C_loss, EG_loss, RECON_X_loss, RECON_Z_loss, EG_loss2
            elif self.loss_type == 'mse1':
                z_recon = self.encode(x_tilde)
                x_recon = self.generate(z_hat)
                RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='sum') / batch_sizes
                # RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes
                EG_loss2 = EG_loss + (beta1 + beta2) * RECON_X_loss
                return C_loss, EG_loss, RECON_X_loss,EG_loss2
            elif self.loss_type == 'mse2':
                z_recon = self.encode(x_tilde)
                # x_recon = self.generate(z_hat)
                # RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='sum') / batch_sizes
                RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes
                EG_loss2 = EG_loss + (beta1 + beta2) * RECON_Z_loss
                return C_loss, EG_loss,RECON_Z_loss, EG_loss2

            elif self.loss_type == 'ssim':

                x_tilde_copy = x_tilde.clone().detach()
                z_hat_copy = z_hat.clone().detach()
                x_recon = self.generate(z_hat_copy)
                if l1:
                    if not self.size_average:
                        # ret,torch.sum(ret),cs,l1_conv,torch.sum(l1_conv)
                        sm_x_loss,ssm_x_sum,cs,l1_conv,l1_conv_sum = self.sm.forward(x, x_recon, val_range, l1, pads)
                        sm_x_loss = torch.sum(1-sm_x_loss)
                        l1_conv = torch.sum(l1_conv)
                        ssm_loss = sm_x_loss*ssm_alpha+(1-ssm_alpha)*l1_conv
                    else:
                        # ret, cs, l1_conv
                        sm_x_loss, cs,l1_conv = self.sm.forward(x, x_recon, val_range, l1, pads)
                        sm_x_loss = 1-sm_x_loss
                        ssm_loss = sm_x_loss*ssm_alpha+(1-ssm_alpha)*l1_conv
                elif l2:
                    mss_x_loss, l1_conv = self.sm.forward(x, x_recon, l1=l1, l2=l2, pads=pads, val_range=val_range,
                                                          normalize=normalize)
                    l1_conv = (x_recon - x) ** 2
                    if not self.size_average:
                        l1_conv = torch.sum(l1_conv)
                        l1_conv = l1_conv / batch_sizes
                    else:
                        l1_conv = torch.mean(l1_conv)
                    mss_x_loss = 1 - mss_x_loss

                    mss_x_loss = torch.mean(mss_x_loss)
                    mss_loss = ssm_alpha * mss_x_loss + (1 - ssm_alpha) * l1_conv
                    # mss_loss = mss_loss.mean()
                else:
                    if self.size_average:
                        sm_x_loss, cs= self.sm.forward(x, x_recon, val_range, l1, pads)
                        ssm_loss = 1 - sm_x_loss
                        # ssm_loss = ssm_loss.mean()
                        # ssm_loss = sm_x_loss * ssm_alpha + (1 - ssm_alpha) * l1_conv
                    else:
                        sm_x_loss,cs,ssm_x_sum = self.sm.forward(x, x_recon, val_range, l1, pads)
                        ssm_loss = torch.sum(1-sm_x_loss)
                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                return C_loss,EG_loss,ssm_loss,RECON_Z_loss

            elif self.loss_type == 'msssim':
                z_hat_copy = z_hat.clone().detach()
                x_tilde_copy = x_tilde.clone().detach()
                x_recon = self.generate(z_hat_copy)

                if l1:
                    mss_x_loss,l1_conv = self.sm.forward(x,x_recon,l1=l1,pads=pads,val_range=val_range,normalize=normalize)
                    if not self.size_average:
                        l1_conv = l1_conv/batch_sizes
                    mss_x_loss = 1 - mss_x_loss

                    mss_x_loss = torch.mean(mss_x_loss)
                    mss_loss = ssm_alpha*mss_x_loss+(1-ssm_alpha)*l1_conv
                    # mss_loss = mss_loss.mean()
                elif l2:
                    mss_x_loss, l1_conv = self.sm.forward(x, x_recon, l1=l1, l2=l2, pads=pads, val_range=val_range,
                                                          normalize=normalize)
                    l1_conv = (x_recon - x) ** 2
                    # l1_conv = torch.sum(l1_conv)
                    if not self.size_average:
                        l1_conv = torch.sum(l1_conv)
                        l1_conv = l1_conv / batch_sizes
                    else:
                        l1_conv = torch.mean(l1_conv)
                    mss_x_loss = 1 - mss_x_loss
                    # if len(mss_x_loss.size()) == 0 or mss_x_loss.size()[0] <= 1:
                    #     mss_x_loss = mss_x_loss
                    # else:
                    #     mss_x_loss = torch.mean(mss_x_loss)
                    mss_x_loss = torch.mean(mss_x_loss)
                    mss_loss = ssm_alpha * mss_x_loss + (1 - ssm_alpha) * l1_conv
                    # mss_loss = mss_loss.mean()
                else:
                    mss_x_loss = self.sm.forward(x,x_recon,l1=l1,pads=pads,val_range=val_range,normalize=normalize)
                    mss_loss = 1 - mss_x_loss

                    mss_loss = torch.mean(mss_loss)
                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                # mss_loss*beta3+
                EG_loss2 = beta1*(RECON_Z_loss*beta2)+EG_loss
                if l1 or l2:
                    return C_loss,EG_loss,mss_loss,l1_conv,mss_x_loss,EG_loss2,RECON_Z_loss
                else:
                    return C_loss,EG_loss,mss_loss,mss_x_loss,EG_loss2
                # F.l1_loss()
            elif self.loss_type == "per":
                z_hat_copy = z_hat.clone().detach()
                x_tilde_copy = x_tilde.clone().detach()
                x_recon = self.generate(z_hat_copy)
                # percept_criticize_x(self,x,x_recon)
                per_loss,feature_loss,style_loss = self.percept_criticize_x(x,x_recon)

                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                EG_loss2 = beta1 * (per_loss * beta3 + RECON_Z_loss * beta2) + EG_loss
                return C_loss,EG_loss,per_loss,feature_loss,style_loss,EG_loss2,RECON_Z_loss

            else:
                return C_loss,EG_loss
            # elif self.loss_type == 'mmd0':
        elif gan == 1:
            if self.loss_type == 'mmd':
                x_tilde_copy = x_tilde.clone().detach()
                # z_hat_copy = z
                if methods > 0:
                    z_recon = self.encode(x_tilde_copy)

                else:
                    # x_recon = self.generate(z_hat)
                    z_recon = self.encode(x_tilde_copy)
                s_z,s_recon = self.mmd_criticize_z(z,z_recon)
                # s_x,s_recon = self.mmd_criticize_x(x,x_recon)
                s_z = torch.squeeze(s_z)
                # s_x = torch.squeeze(s_x)
                s_recon = torch.squeeze(s_recon)
                mmd_penalty = self.mmd_gradient_penalty(z, z_recon,aim=1)
                mmd_penalty = lamb*mmd_penalty
                C_loss,EG_loss = math_util.mmd_g_loss(s_recon,s_z,sigma=self.sigma,dis_penalty=mmd_penalty)

                C_loss = beta1*C_loss
                EG_loss = beta1*EG_loss

                return C_loss,EG_loss,mmd_penalty

            elif self.loss_type == 'mmd_b':
                x_tilde_copy = x_tilde.clone().detach()
                if methods > 0:
                    # x_recon = self.generate(z_hat, var_beta=var_beta, clip_beta=clip_beta)
                    z_recon = self.encode(x_tilde_copy)
                else:
                    # x_recon = self.generate(z_hat)
                    z_recon = self.encode(x_tilde_copy)
                # s_x, s_recon = self.mmd_criticize_x(x, x_recon)
                s_x, s_recon = self.mmd_criticize_z(z, z_recon)
                s_x = torch.squeeze(s_x)
                s_recon = torch.squeeze(s_recon)
                # mmd_penalty = self.mmd_gradient_penalty(x, x_recon)
                mmd_penalty = self.mmd_gradient_penalty(z, z_recon,aim=1)
                mmd_penalty = lamb * mmd_penalty
                C_loss,EG_loss = math_util.mmd_g_bound_loss(s_recon,s_x,mmd_penalty)

                C_loss = beta1*C_loss
                EG_loss = beta1*EG_loss

                return C_loss,EG_loss,mmd_penalty
            elif self.loss_type == 'rep_gp':
                x_tilde_copy = x_tilde.clone().detach()
                if methods > 0:
                    # x_recon = self.generate(z_hat, var_beta=var_beta, clip_beta=clip_beta)
                    z_recon = self.encode(x_tilde_copy)
                else:
                    # x_recon = self.generate(z_hat)
                    z_recon = self.encode(x_tilde_copy)
                # s_x,s_recon = self.mmd_criticize_x(x,x_recon)
                s_x, s_recon = self.mmd_criticize_z(z, z_recon)
                s_x = torch.squeeze(s_x)
                s_recon = torch.squeeze(s_recon)
                # mmd_penalty = self.mmd_gradient_penalty(x, x_recon,mode='rep_gp')
                mmd_penalty = self.mmd_gradient_penalty(z, z_recon, mode='rep_gp',aim=1)
                mmd_penalty = lamb * mmd_penalty
                C_loss,EG_loss = math_util.mmd_repulsive_g_loss(s_recon,s_x,repulsive_weights=self.rep_weights,
                                                                dis_penalty=mmd_penalty)

                C_loss = beta1*C_loss
                EG_loss = beta1*EG_loss

                return C_loss,EG_loss,mmd_penalty
            elif self.loss_type == 'rmb_gp':
                x_tilde_copy = x_tilde.clone().detach()
                if methods > 0:
                    z_recon = self.encode(x_tilde_copy)
                    # x_recon = self.generate(z_hat, var_beta=var_beta, clip_beta=clip_beta)
                else:
                    z_recon = self.encode(x_tilde_copy)
                    # x_recon = self.generate(z_hat)
                # s_x,s_recon = self.mmd_criticize_x(x,x_recon)
                s_x, s_recon = self.mmd_criticize_z(z, z_recon)
                s_x = torch.squeeze(s_x)
                s_recon = torch.squeeze(s_recon)
                # mmd_penalty = self.mmd_gradient_penalty(x,x_recon,mode='rmb_gp')
                mmd_penalty = self.mmd_gradient_penalty(z, z_recon, mode='rmb_gp',aim=1)
                mmd_penalty = lamb*mmd_penalty
                C_loss,EG_loss = math_util.mmd_repulsive_g_bounded_loss(s_recon,s_x,self.rep_weights,mmd_penalty)
                C_loss = beta1*C_loss
                EG_loss = beta1*EG_loss

                return C_loss,EG_loss,mmd_penalty
            else:
                x_tilde_copy = x_tilde.clone().detach()
                if methods > 0:
                    z_recon = self.encode(x_tilde_copy)
                    # x_recon = self.generate(z_hat, var_beta=var_beta, clip_beta=clip_beta)
                else:
                    z_recon = self.encode(x_tilde_copy)
                    # x_recon = self.generate(z_hat)
                # s_x, s_recon = self.mmd_criticize_x(x, x_recon)
                s_x, s_recon = self.mmd_criticize_z(z, z_recon)
                s_x = torch.squeeze(s_x)
                s_recon = torch.squeeze(s_recon)
                # mmd_penalty = self.mmd_gradient_penalty(z, z_recon)
                mmd_penalty = self.mmd_gradient_penalty(z, z_recon,aim=1)
                mmd_penalty = lamb * mmd_penalty
                C_loss, EG_loss = math_util.mmd_g_loss(s_recon, s_x, sigma=self.sigma, dis_penalty=mmd_penalty)
                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='mean')
                C_loss = beta1 * C_loss
                EG_loss = beta1 * EG_loss

                return C_loss, EG_loss, mmd_penalty
        else:
            self.loss_type = loss_type
            data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
            EG_loss = torch.mean(data_preds - sample_preds)
            C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)


            if self.loss_type == 'raw':
                return C_loss, EG_loss
            if self.loss_type == 'mse0':
                x_tilde_copy = x_tilde.clone().detach()
                z_recon = self.encode(x_tilde_copy)
                z_hat_copy = z_hat
                x_recon = self.generate(z_hat_copy)
                RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='sum') / batch_sizes
                RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes

                return RECON_X_loss, RECON_Z_loss
            elif self.loss_type == 'mse1':
                z_recon = self.encode(x_tilde)
                x_recon = self.generate(z_hat)
                RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='sum') / batch_sizes
                # RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes
                return RECON_X_loss
            elif self.loss_type == 'mse2':
                z_recon = self.encode(x_tilde)

                RECON_Z_loss = F.mse_loss(input=z_recon, target=z, reduction='sum') / batch_sizes
                return RECON_Z_loss
            elif self.loss_type == 'ssim':

                x_tilde_copy = x_tilde.clone().detach()
                z_hat_copy = z_hat
                x_recon = self.generate(z_hat_copy)
                if l1:
                    if not self.size_average:
                        # ret,torch.sum(ret),cs,l1_conv,torch.sum(l1_conv)
                        sm_x_loss, ssm_x_sum, cs, l1_conv, l1_conv_sum = self.sm.forward(x, x_recon, val_range, l1,
                                                                                         pads)
                        sm_x_loss = torch.sum(1 - sm_x_loss)
                        l1_conv = torch.sum(l1_conv)
                        ssm_loss = sm_x_loss * ssm_alpha + (1 - ssm_alpha) * l1_conv
                    else:
                        # ret, cs, l1_conv
                        sm_x_loss, cs, l1_conv = self.sm.forward(x, x_recon, val_range, l1, pads)
                        sm_x_loss = 1 - sm_x_loss
                        ssm_loss = sm_x_loss * ssm_alpha + (1 - ssm_alpha) * l1_conv
                else:
                    if self.size_average:
                        sm_x_loss, cs = self.sm.forward(x, x_recon, val_range, l1, pads)
                        ssm_loss = 1 - sm_x_loss

                        # ssm_loss = ssm_loss.mean()
                        # ssm_loss = sm_x_loss * ssm_alpha + (1 - ssm_alpha) * l1_conv
                    else:
                        sm_x_loss, cs, ssm_x_sum = self.sm.forward(x, x_recon, val_range, l1, pads)
                        ssm_loss = torch.sum(1 - sm_x_loss)
                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                return C_loss, EG_loss, ssm_loss, RECON_Z_loss
            elif self.loss_type == "per":
                z_hat_copy = z_hat.clone().detach()
                x_tilde_copy = x_tilde.clone().detach()
                x_recon = self.generate(z_hat_copy)
                # percept_criticize_x(self,x,x_recon)
                per_loss,feature_loss,style_loss = self.percept_criticize_x(x,x_recon)

                z_recon2 = self.encode(x_tilde_copy)

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                EG_loss2 = beta1 * (per_loss * beta3 + RECON_Z_loss * beta2) + EG_loss
                RECON_X_loss = F.mse_loss(input=x_recon, target=x, reduction='mean')
                return C_loss,EG_loss,per_loss,feature_loss,style_loss,EG_loss2,RECON_Z_loss,RECON_X_loss
            elif self.loss_type == 'msssim':
                z_hat_copy = z_hat
                x_tilde_copy = x_tilde.clone().detach()
                x_recon = self.generate(z_hat_copy)

                if l1:
                    mss_x_loss, l1_conv = self.sm.forward(x, x_recon, l1=l1, pads=pads, val_range=val_range,
                                                          normalize=normalize)
                    if not self.size_average:
                        l1_conv = l1_conv / batch_sizes
                    mss_x_loss = 1 - mss_x_loss

                    mss_x_loss = torch.mean(mss_x_loss)
                    mss_loss = ssm_alpha * mss_x_loss + (1 - ssm_alpha) * l1_conv

                elif l2:
                    mss_x_loss, l1_conv = self.sm.forward(x, x_recon, l1=l1, l2=l2, pads=pads, val_range=val_range,
                                                          normalize=normalize)
                    l1_conv = (x_recon - x) ** 2
                    # l1_conv = torch.sum(l1_conv)
                    if not self.size_average:
                        l1_conv = torch.sum(l1_conv)
                        l1_conv = l1_conv / batch_sizes
                    else:
                        l1_conv = torch.mean(l1_conv)
                    mss_x_loss = 1 - mss_x_loss

                    mss_x_loss = torch.mean(mss_x_loss)
                    mss_loss = ssm_alpha * mss_x_loss + (1 - ssm_alpha) * l1_conv
                    # mss_loss = mss_loss.mean()
                else:
                    mss_x_loss = self.sm.forward(x, x_recon, l1=l1, pads=pads, val_range=val_range, normalize=normalize)
                    mss_loss = 1 - mss_x_loss

                    mss_loss = torch.mean(mss_loss)
                z_recon2 = self.encode(x_tilde_copy)
                RECON_X_loss = F.mse_loss(input=x_recon,target=x,reduction='mean')

                RECON_Z_loss = F.mse_loss(input=z_recon2, target=z, reduction='mean')
                EG_loss2 = beta1 * (RECON_Z_loss * beta2) + EG_loss
                # mss_loss * beta3 +
                if l1 or l2:
                    return C_loss, EG_loss, mss_loss, l1_conv, mss_x_loss, EG_loss2, RECON_Z_loss,RECON_X_loss
                else:
                    return C_loss, EG_loss, mss_loss, mss_x_loss, EG_loss2,RECON_X_loss
            else:
                return C_loss, EG_loss







