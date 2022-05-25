#定义模型
class PLNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv2d(16, 12, kernel_size=(6, 6), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(6, 1), # kernel_size, stride    
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv2d(16, 12, kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(7, 1), # kernel_size, stride    
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv2d(16, 12, kernel_size=(4, 4), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(8, 1), # kernel_size, stride    
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv2d(16, 12, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(9, 1), # kernel_size, stride    
        )
        
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 24),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(24, 12),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(12, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        feature2 = self.conv2(img)
        feature3 = self.conv3(img)
        feature4 = self.conv4(img)
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output

#定义模型
class PLUNNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
#             nn.AvgPool2d(kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)),
            nn.Conv2d(16, 12, kernel_size=(6, 6), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.MaxPool2d(6, 1), # kernel_size, stride
            
             
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
#             nn.AvgPool2d(kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)),
            nn.Conv2d(16, 12, kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.MaxPool2d(7, 1), # kernel_size, stride
            
             
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
#             nn.AvgPool2d(kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)),
            nn.Conv2d(16, 12, kernel_size=(4, 4), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.MaxPool2d(8, 1), # kernel_size, stride
            
             
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(64, 64), padding=(0, 0), stride=(64, 64)), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
#             nn.AvgPool2d(kernel_size=(5, 5), padding=(0, 0), stride=(1, 1)),
            nn.Conv2d(16, 12, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.MaxPool2d(9, 1), # kernel_size, stride
            
             
        )
        
        
        
            
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 12),
#             nn.BatchNorm1d(8),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(12, 6),
#             nn.BatchNorm1d(5),
            nn.ReLU(),
            
#             nn.Dropout(0.1),
            nn.Linear(6, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        feature2 = self.conv2(img)
        feature3 = self.conv3(img)
        feature4 = self.conv4(img)
        
#         feature = feature1+feature2+feature3+feature4
        
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
#         print(feature1.shape,feature2.shape)    
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output
    
#定义模型
class TDNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(num_inputs, 16, kernel_size=(101,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            
            nn.Conv3d(16, 12, kernel_size=(10,3,3), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            
            # nn.Conv3d(16, 12, kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,4,4), padding=(0,0,0), stride=(1, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(num_inputs, 16, kernel_size=(101,5,5), padding=(0,0,0), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            
            nn.Conv3d(16, 12, kernel_size=(10,4,4), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            
            # nn.Conv3d(16, 12, kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,4,4), padding=(0,0,0), stride=(1, 1, 1))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(num_inputs, 16, kernel_size=(101,4,4), padding=(0,0,0), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            
            nn.Conv3d(16, 12, kernel_size=(10,5,5), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            
            # nn.Conv3d(16, 12, kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,4,4), padding=(0,0,0), stride=(1, 1, 1))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(num_inputs, 16, kernel_size=(101,3,3), padding=(0,0,0), stride=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            
            nn.Conv3d(16, 12, kernel_size=(10,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            
            # nn.Conv3d(16, 12, kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,4,4), padding=(0,0,0), stride=(1, 1, 1))
        )
        
        
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 24),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(24, 12),
            nn.ReLU(),
            
            nn.Linear(12, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        # print(feature1.size())
        feature2 = self.conv2(img)
        # print(feature2.size())
        feature3 = self.conv3(img)
        # print(feature3.size())
        feature4 = self.conv4(img)
        # print(feature4.size())
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output

#定义模型
class TDUNNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(num_inputs, 24, kernel_size=(110,1,1), padding=(0,0,0), stride=(1, 1, 1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            # nn.Dropout(0.5),
            
            nn.Conv3d(24, 12, kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,6,6), padding=(0,0,0), stride=(1, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(num_inputs, 24, kernel_size=(110,1,1), padding=(0,0,0), stride=(1, 1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            # nn.Dropout(0.5),
            
            nn.Conv3d(24, 12, kernel_size=(1,5,5), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,7,7), padding=(0,0,0), stride=(1, 1, 1)) 
         )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(num_inputs, 24, kernel_size=(110,1,1), padding=(0,0,0), stride=(1, 1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            # nn.Dropout(0.5),
            
            nn.Conv3d(24, 12, kernel_size=(1,4,4), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,8,8), padding=(0,0,0), stride=(1, 1, 1)) 
         )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(num_inputs, 24, kernel_size=(110,1,1), padding=(0,0,0), stride=(1, 1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            # nn.Dropout(0.5),
            
            nn.Conv3d(24, 12, kernel_size=(1,3,3), padding=(0,0,0), stride=(1, 1, 1)),
            # nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,9,9), padding=(0,0,0), stride=(1, 1, 1)) 
         )
        
        
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 24),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(24, 12),
            nn.ReLU(),
            
            nn.Linear(12, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        feature2 = self.conv2(img)
        feature3 = self.conv3(img)
        feature4 = self.conv4(img)
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output
    
#定义模型
class LSNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(6, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(6, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride            
         )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(5, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(7, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride            
         )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(4, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(8, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride            
         )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(9, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride            
         )
        
        
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 24),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(24, 12),
            nn.ReLU(),
            
            nn.Linear(12, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        feature2 = self.conv2(img)
        feature3 = self.conv3(img)
        feature4 = self.conv4(img)
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output

#定义模型
class LSUNNORMAL(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(6, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(6, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride
         )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(5, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(7, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride
         )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(4, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(8, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride
         )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=(11, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(0, 0), stride=(11, 1)),
            
            nn.Conv2d(16, 12, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(9, 1), padding=(0, 0), stride=(1, 1)), # kernel_size, stride
         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(num_inputs, 12, kernel_size=(11*4, 121), padding=(5, 0), stride=(1, 1)), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.MaxPool2d(kernel_size=(88, 1), padding=(0, 0), stride=(1, 1)),
#         )
        
        self.fc = nn.Sequential(
            nn.Linear(48*1*1, 24),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(24, 12),
            nn.ReLU(),
            
            nn.Linear(12, num_outputs),
        )

    def forward(self, img):
        feature1 = self.conv1(img)
        feature2 = self.conv2(img)
        feature3 = self.conv3(img)
        feature4 = self.conv4(img)
        feature = torch.cat((feature1,feature2,feature3,feature4),1)
        x = torch.flatten(feature,1)
        output = self.fc(x)
        return output   
