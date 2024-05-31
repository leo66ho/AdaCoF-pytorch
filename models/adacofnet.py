import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize


def make_model(args):
    return AdaCoFNet(args).cuda()


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleOcclusion = Subnet_occlusion()

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)


        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        #Grabbing 128 components####
        tensorPool3_128 = self.moduleUpsample3(tensorPool3)
        tensorDeconv2_128 = self.moduleDeconv2(tensorPool3_128)
        tensorUpsample2_128 = self.moduleUpsample2(tensorDeconv2_128)
        tensorCombine_128 = tensorUpsample2_128 + tensorConv2
        ######

        tensorCombine = tensorUpsample3 + tensorConv3
        

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        #Grabbing 128 components####
        #tensorDeconv_128 = self.moduleDeconv(tensorCombine_128)
        #tensorUpsample_128 = self.moduleUpsample(tensorDeconv_128)
        
        #tensorCombine_128 = tensorUpsample_128 + tensorConv2
        ######

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Occlusion = self.moduleOcclusion(tensorCombine)
        #Setting 128 weights####
        Weight1_128 = self.moduleWeight1(tensorCombine_128)
        Alpha1_128 = self.moduleAlpha1(tensorCombine_128)
        Beta1_128 = self.moduleBeta1(tensorCombine_128)
        Weight2_128 = self.moduleWeight2(tensorCombine_128)
        Alpha2_128 = self.moduleAlpha2(tensorCombine_128)
        Beta2_128= self.moduleBeta2(tensorCombine_128)
        Occlusion_128 = self.moduleOcclusion(tensorCombine_128)
        ######



        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion,Weight1_128, Alpha1_128, Beta1_128, Weight2_128, Alpha2_128, Beta2_128, Occlusion_128


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion,Weight1_128, Alpha1_128, Beta1_128, Weight2_128, Alpha2_128, Beta2_128, Occlusion_128 = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))
        ##128##
        
        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation)
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation)
        tensorAdaCoF1_128 = self.moduleAdaCoF(self.modulePad(frame0), Weight1_128, Alpha1_128, Beta1_128, self.dilation)
        tensorAdaCoF2_128 = self.moduleAdaCoF(self.modulePad(frame2), Weight2_128, Alpha2_128, Beta2_128, self.dilation)

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        frame1_128 = Occlusion_128 * tensorAdaCoF1_128 + (1 - Occlusion_128) * tensorAdaCoF2_128
        if h_padded:
            frame1_128 = frame1_128[:, :, 0:h0, :]
        if w_padded:
            frame1_128 = frame1_128[:, :, :, 0:w0]

        frame1_final = (frame1+frame1_128)/2.0


        if self.training:
            # Smoothness Terms
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

            g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
            g_Alpha2 = CharbonnierFunc(m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
            g_Beta2 = CharbonnierFunc(m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
            g_Occlusion = CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            m_Alpha1_128 = torch.mean(Weight1_128 * Alpha1_128, dim=1, keepdim=True)
            m_Alpha2_128 = torch.mean(Weight2_128 * Alpha2_128, dim=1, keepdim=True)
            m_Beta1_128 = torch.mean(Weight1_128 * Beta1_128, dim=1, keepdim=True)
            m_Beta2_128 = torch.mean(Weight2_128 * Beta2_128, dim=1, keepdim=True)

            g_Alpha1_128 = CharbonnierFunc(m_Alpha1_128[:, :, :, :-1] - m_Alpha1_128[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1_128[:, :, :-1, :] - m_Alpha1_128[:, :, 1:, :])
            g_Beta1_128 = CharbonnierFunc(m_Beta1_128[:, :, :, :-1] - m_Beta1_128[:, :, :, 1:]) + CharbonnierFunc(m_Beta1_128[:, :, :-1, :] - m_Beta1_128[:, :, 1:, :])
            g_Alpha2_128 = CharbonnierFunc(m_Alpha2_128[:, :, :, :-1] - m_Alpha2_128[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2_128[:, :, :-1, :] - m_Alpha2_128[:, :, 1:, :])
            g_Beta2_128 = CharbonnierFunc(m_Beta2_128[:, :, :, :-1] - m_Beta2_128[:, :, :, 1:]) + CharbonnierFunc(m_Beta2_128[:, :, :-1, :] - m_Beta2_128[:, :, 1:, :])
            g_Occlusion_128 = CharbonnierFunc(Occlusion_128[:, :, :, :-1] - Occlusion_128[:, :, :, 1:]) + CharbonnierFunc(Occlusion_128[:, :, :-1, :] - Occlusion_128[:, :, 1:, :])

            g_Spatial = g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

            g_Spatial_128 = g_Alpha1_128+g_Beta1_128+g_Alpha2_128+g_Beta2_128

            return {'frame1': frame1_final, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion, 'g_Spatial_128': g_Spatial_128, 'g_Occlusion_128': g_Occlusion_128}
        else:
            return frame1_final