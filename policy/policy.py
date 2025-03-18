import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from termcolor import cprint
from torch.autograd import Variable
from einops import reduce
from transformers import BertConfig,BertModel


class DSP(nn.Module):
    def __init__(
        self,
        num_action=20,
        input_dim=6,
        obs_feature_dim=512,
        action_dim=10,
        hidden_dim=512,
        nheads=8,
        num_encoder_layers=4,
        num_decoder_layers=7,
        dim_feedforward=2048,
        dropout=0.1,
        obj_dim=9,
    ):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.resnet18_cam1 = RestNet18()
        self.resnet18_cam2 = RestNet18()
        self.num_action = num_action
        config = BertConfig(hidden_size=obs_feature_dim, num_attention_heads=8, intermediate_size=obs_feature_dim * 4, num_hidden_layers=4) 
        self.action_projection = nn.Linear(obs_feature_dim, action_dim)
        self.cross_attention = BertModel(config)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        

        
    def forward(self, imgtop, imghand, actions=None, batch_size=24):

        batch_size, cam_len, height, width, channels = imgtop.shape
        imgtop = imgtop.float()/255.0
        imghand = imghand.float()/255.0

        imgtop_transformed = self.transform(imgtop.reshape(-1, channels, height, width))
        imghand_transformed = self.transform(imghand.reshape(-1, channels, height, width))
        
        imgtop_features = self.resnet18_cam1(imgtop_transformed)
        imghand_features = self.resnet18_cam2(imghand_transformed)

        readout = torch.cat((imgtop_features, imghand_features), dim=-1)
        
        if actions is not None:
            condition =  readout.unsqueeze(1)
            action_pred = torch.zeros(condition.size(0), 1, condition.size(2), device=condition.device)
        
            while action_pred.shape[1] < actions.shape[1]: 

                action_pred = self.upsample(action_pred.transpose(1, 2)).transpose(1, 2)
                input_action = torch.cat([action_pred,condition],dim=1)

                attention_output = self.cross_attention(inputs_embeds = input_action).last_hidden_state
                action_pred = attention_output[:,:action_pred.shape[1],:]  
            
            action_pred = action_pred[:,:actions.shape[1],:]
            action_pred = self.action_projection(action_pred)

            loss = F.mse_loss(action_pred, actions, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            return loss
        else:
            with torch.no_grad():
                condition =  readout.unsqueeze(1)
                action_pred = torch.zeros(condition.size(0), 1, condition.size(2), device=condition.device)

                while action_pred.shape[1] < self.num_action: 

                    action_pred = self.upsample(action_pred.transpose(1, 2)).transpose(1, 2)
                    input_action = torch.cat([action_pred,condition],dim=1)

                    attention_output = self.cross_attention(inputs_embeds = input_action).last_hidden_state
                    action_pred = attention_output[:,:action_pred.shape[1],:]

                action_pred = action_pred[:,:self.num_action,:]
                action_pred = self.action_projection(action_pred)

                action_pred = action_pred[:,:self.num_action,:] 
                return action_pred
            
class RestNetBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.GroupNorm(out_channels//16,out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn2 = nn.GroupNorm(out_channels//16,out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride[0],
                               padding=1)
        self.bn1 = nn.GroupNorm(out_channels//16,out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride[1],
                               padding=1)
        self.bn2 = nn.GroupNorm(out_channels//16,out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=stride[0],
                      padding=0), nn.GroupNorm(out_channels//16,out_channels))

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):

    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out