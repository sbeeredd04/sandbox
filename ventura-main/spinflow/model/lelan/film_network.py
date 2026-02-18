"""
Source: https://github.com/NHirose/learning-language-navigation
"""
import torch
import torch.nn as nn

def create_conv_layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )

class InitialFeatureExtractor(nn.Module):
    def __init__(self):
        super(InitialFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(
            create_conv_layer(3, 128, 5, 2, 2),
            create_conv_layer(128, 128, 3, 2, 1),
            create_conv_layer(128, 128, 3, 2, 1),
        )
        
    def forward(self, x):
        return self.layers(x)
    
class IntermediateFeatureExtractor(nn.Module):
    def __init__(self):
        super(IntermediateFeatureExtractor, self).__init__()
        
        self.layers = nn.Sequential(       
            create_conv_layer(128, 256, 3, 2, 1),
            create_conv_layer(256, 512, 3, 2, 1),
            create_conv_layer(512, 1024, 3, 2, 1),
            create_conv_layer(1024, 1024, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.layers(x)
    
class FiLMTransform(nn.Module):
    def __init__(self):
        super(FiLMTransform, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.film_transform = FiLMTransform()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film_transform(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x
    
class FinalClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(FinalClassifier, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        feature_map = x
        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc_layers(x)
        
        return x, feature_map

class FiLMNetwork(nn.Module):
    def __init__(self, num_res_blocks, num_classes, num_channels, question_dim):
        super(FiLMNetwork, self).__init__()
        question_feature_dim = question_dim

        self.film_param_generator = nn.Linear(question_feature_dim, 2 * num_res_blocks * num_channels)
        self.initial_feature_extractor = InitialFeatureExtractor()
        self.residual_blocks = nn.ModuleList()
        self.intermediate_feature_extractor = IntermediateFeatureExtractor()
        
        for _ in range(num_res_blocks):
            self.residual_blocks.append(ResidualBlock(num_channels + 2, num_channels))
            
        # self.final_classifier = FinalClassifier(num_channels, num_classes)
        # # Turn off gradients for the final classifier as it is not used
        # for param in self.final_classifier.parameters():
        #     param.requires_grad = False
            
    
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        
    def forward(self, x, question):
        batch_size = x.size(0)
        device = x.device
        
        x = self.initial_feature_extractor(x)
        film_params = self.film_param_generator(question).view(
            batch_size, self.num_res_blocks, 2, self.num_channels)
        
        dx, dy = x.size(2), x.size(3)
        coord_x = torch.arange(-1, 1 + 0.00001, 2 / (dx-1)).to(device)
        coord_x = coord_x.view(dx, 1).expand(batch_size, 1, dx, dy)
        coord_y = torch.arange(-1, 1 + 0.00001, 2 / (dy-1)).to(device)
        coord_y = coord_y.view(1, dy).expand(batch_size, 1, dx, dy)
        for i, res_block in enumerate(self.residual_blocks):
            beta = film_params[:, i, 0, :]
            gamma = film_params[:, i, 1, :]
            x = torch.cat([x, coord_x, coord_y], 1)
            x = res_block(x, beta, gamma)
        
        features = self.intermediate_feature_extractor(x)
        
        return features