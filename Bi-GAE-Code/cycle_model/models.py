import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from .utils import weights_init_normal


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc,n_feature=64,down=2,encode_feature=[],nlat=256,image_size=256, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model1 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, n_feature, 7),
                    nn.InstanceNorm2d(n_feature),
                    nn.ReLU(inplace=True)]

        # Downsampling
        in_features = n_feature
        out_features = in_features*2
        now_size = image_size
        # model1 = []
        for _ in range(down):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
            now_size = now_size // 2

        tmp_using_feature = [in_features]
        if encode_feature:
            for k in range(len(encode_feature)):
                if now_size * now_size * tmp_using_feature[-1] <= nlat or now_size <= 1:
                    break
                else:
                    model1 += [ nn.Conv2d(tmp_using_feature[-1], encode_feature[k], 3, stride=2, padding=1),
                        nn.InstanceNorm2d(encode_feature[k]),
                        nn.ReLU(inplace=True) ]

                    tmp_using_feature.append(encode_feature[k])

        # Residual blocks
        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(tmp_using_feature[-1])]

        # Upsampling
        model3 = []
        if len(tmp_using_feature)>1:
            for k in range(len(tmp_using_feature),0,-1):
                model3 += [nn.ConvTranspose2d(tmp_using_feature[k], tmp_using_feature[k-1], 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]

            in_features = tmp_using_feature[0]
            out_features = in_features // 2
            for _ in range(down):
                model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
                in_features = out_features
                out_features = in_features // 2
        else:
            in_features = tmp_using_feature[-1]
            out_features = in_features // 2
            for _ in range(down):
                model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
                in_features = out_features
                out_features = in_features // 2

        # Output layer
        model3 += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(n_feature, output_nc, 7),
                    nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(out1)
        out3 = self.model3(out2)

        return out3,out2

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class CycleGANFramework(nn.Module):
    def __init__(self,netG_A2B,netG_B2A,netD_A,netD_B,loss_type='raw',max_size=100,initial=True):
        super().__init__()
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_A = netD_A
        self.netD_B = netD_B
        self.loss_type = loss_type
        self.target_real = Variable(torch.tensor([1.0 for i in range(max_size)]), requires_grad=False)
        self.target_fake = Variable(torch.tensor([0.0 for i in range(max_size)]), requires_grad=False)

        if initial:
            self.netG_A2B.apply(weights_init_normal)
            self.netG_B2A.apply(weights_init_normal)
            self.netD_A.apply(weights_init_normal)
            self.netD_B.apply(weights_init_normal)

        if 'gp' in self.loss_type:
            print("using wgangp")
        else:
            print("using mse LOSS")


    def get_G_A2B(self):
        return self.netG_A2B

    def get_G_B2A(self):
        return self.netG_B2A

    def get_D_A(self):
        return self.netD_A

    def get_D_B(self):
        return self.netD_B

    def get_g_a2b_parameters(self):
        return self.netG_A2B.parameters()

    def get_g_b2a_parameters(self):
        return self.netG_B2A.parameters()

    def get_d_a_parameters(self):
        return self.netD_A.parameters()

    def get_d_b_parameters(self):
        return self.netD_B.parameters()

    def calculate_grad_penalty(self, x, x_tilde,aim="A"):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        # intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        # intp_z.requires_grad = True
        # ,_,_
        if "A".strip() in aim.strip():
            tmp_C = self.netD_A(intp_x)
        else:
            tmp_C = self.netD_B(intp_x)
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

    def criterion_identity(self,y_pre,y_real):
        # torch.nn.L1Loss()
        loss_identity = F.l1_loss(input=y_pre,target=y_real,reduction='mean')
        return loss_identity

    def criterion_cycle(self,y_pre,y_real):
        loss_cycle = F.l1_loss(input=y_pre,target=y_real,reduction='mean')
        return loss_cycle

    def reconstruct_ABA(self,x):
        tmp_b,_ = self.netG_A2B(x)
        out_a,_ = self.netG_B2A(tmp_b)
        return out_a

    def reconstruct_BAB(self, x):
        tmp_a,_ = self.netG_B2A(x)
        out_b,_ = self.netG_A2B(tmp_a)
        return out_b

    def generate_A2B(self,x):
        out_b,_ = self.netG_A2B(x)
        return out_b

    def generate_B2A(self,x):
        out_a,_ = self.netG_B2A(x)
        return out_a

    def criterion_GAN(self,y_fake,y_real,aim="A"):
        if self.loss_type.strip() == "raw".strip():
            gan_loss = F.mse_loss(input=y_fake,target=y_real)
            return gan_loss
        else:
            if "A".strip() in aim.strip():
                input_x = torch.cat((y_real, y_fake), dim=0)
                # input_z = torch.cat((z_hat, z), dim=0)
                # ,_,_
                output = self.netD_A(input_x)
                data_preds, sample_preds = output[:y_real.size(0)], output[y_real.size(0):]
                return data_preds,sample_preds
            else:
                input_x = torch.cat((y_real, y_fake), dim=0)
                # input_z = torch.cat((z_hat, z), dim=0)
                # ,_,_
                output = self.netD_B(input_x)
                data_preds, sample_preds = output[:y_real.size(0)], output[y_real.size(0):]
                return data_preds, sample_preds

        # return gan_loss

    def forward(self, x_a,x_b,disc=False,loss_lamb_id=5.0,lamb_gp=10.0,loss_lamb_cycle=10.0):
        # target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
        # target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
        if not disc:
            same_B,_ = self.netG_A2B(x_b)
            loss_identity_B = self.criterion_identity(same_B, x_b) * loss_lamb_id
            # G_B2A(A) should equal A if real A is fed
            same_A,_ = self.netG_B2A(x_a)
            loss_identity_A = self.criterion_identity(same_A, x_a) * loss_lamb_id

            if "gp" not in self.loss_type:
                fake_B,_ = self.netG_A2B(x_a)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake, self.target_real[:pred_fake.size()[0]].to(x_a.device))

                fake_A,_ = self.netG_B2A(x_b)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake, self.target_real[:pred_fake.size()[0]].to(x_b.device))
            else:
                # loss_GAN_B2A
                fake_A,_ = self.netG_B2A(x_b)
                data_preds_a, sample_preds_a = self.criterion_GAN(x_a, fake_A,aim="A".strip())

                loss_GAN_B2A = torch.mean(data_preds_a - sample_preds_a)
                C_loss_B2A = -loss_GAN_B2A + lamb_gp * self.calculate_grad_penalty(x_a.data, fake_A.data,aim="A".strip())

                # loss_GAN_A2B
                fake_B,_ = self.netG_A2B(x_a)
                data_preds_b, sample_preds_b = self.criterion_GAN(x_b, fake_B, aim="B".strip())

                loss_GAN_A2B = torch.mean(data_preds_b - sample_preds_b)
                C_loss_A2B = -loss_GAN_A2B + lamb_gp * self.calculate_grad_penalty(x_b.data, fake_B.data, aim="B".strip())

            # cycle loss
            recovered_A,_ = self.netG_B2A(fake_B)
            loss_cycle_ABA = self.criterion_cycle(recovered_A, x_a) * loss_lamb_cycle

            recovered_B,_ = self.netG_A2B(fake_A)
            loss_cycle_BAB = self.criterion_cycle(recovered_B, x_b) * loss_lamb_cycle

            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            return loss_G,loss_identity_A,loss_identity_B,loss_GAN_A2B,loss_GAN_B2A,loss_cycle_ABA,loss_cycle_BAB
        else:
            fake_B,_ = self.netG_A2B(x_a)
            fake_A,_ = self.netG_B2A(x_b)

            if "gp" not in self.loss_type:
                pred_fake_a = self.netD_A(fake_A.detach())
                loss_D_fake_a = self.criterion_GAN(pred_fake_a, self.target_fake[:pred_fake_a.size()[0]].to(x_a.device))
                pred_real_a = self.netD_A(x_a)
                loss_D_real_a = self.criterion_GAN(pred_real_a, self.target_real[:pred_real_a.size()[0]].to(x_b.device))

                loss_D_A = (loss_D_real_a + loss_D_fake_a) * 0.5

                pred_fake_b = self.netD_B(fake_B.detach())
                loss_D_fake_b = self.criterion_GAN(pred_fake_b, self.target_fake[:pred_fake_b.size()[0]].to(x_a.device))
                pred_real_b = self.netD_B(x_b)
                loss_D_real_b = self.criterion_GAN(pred_real_b, self.target_real[:pred_real_b.size()[0]].to(x_b.device))

                loss_D_B = (loss_D_real_b + loss_D_fake_b) * 0.5

                return loss_D_A,loss_D_B
            else:

                fake_A = fake_A.detach()
                data_preds_a, sample_preds_a = self.criterion_GAN(x_a, fake_A, aim="A".strip())

                loss_GAN_B2A = torch.mean(data_preds_a - sample_preds_a)
                loss_D_A = -loss_GAN_B2A + lamb_gp * self.calculate_grad_penalty(x_a.data, fake_A.data,aim="A".strip())


                fake_B = fake_B.detach()
                data_preds_b, sample_preds_b = self.criterion_GAN(x_b, fake_B, aim="B".strip())

                loss_GAN_A2B = torch.mean(data_preds_b - sample_preds_b)
                loss_D_B = -loss_GAN_A2B + lamb_gp * self.calculate_grad_penalty(x_b.data, fake_B.data,aim="B".strip())

                return loss_D_A,loss_D_B







