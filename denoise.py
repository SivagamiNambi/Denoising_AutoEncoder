import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import  Adam
import helper
import numpy as np
import cv2



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3 )
        self.conv4 = nn.ConvTranspose2d(8, 16, 3 )
        self.conv5 = nn.ConvTranspose2d(16, 32, 3,  stride=2, padding=1 ,output_padding=1 )
        self.conv6 = nn.ConvTranspose2d(32, 1, 3,  stride=2, padding=1, output_padding=1 )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        code = F.relu(self.conv3(x))

        #decoder
        x = F.relu(self.conv4(code))
        x = F.relu(self.conv5(x))
        x = F.sigmoid(self.conv6(x))
        return x


def train(epoch):
    model = Net()
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = F.mse_loss
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()
    train_loader, val_loader, test_loader = helper.load_data()

    for e in range(epoch):
        train_loss = 0.0
        for data in train_loader:
            input, labels = data
            noisy_input = input + 0.5 * torch.randn(*input.shape)

            if torch.cuda.is_available():
                input = input.cuda()
                noisy_input = noisy_input.cuda()

            output = model(noisy_input)
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()*input.size(0)

        loss = train_loss/len(train_loader)
        print('Epoch: '+str(e) + '  Loss :'+str(float(loss)))
    torch.save(model.state_dict(), 'model/trained.h5')


def display():
    model = Net()
    model.load_state_dict(torch.load('model/trained.h5'))
    train_loader, val_loader, test_loader = helper.load_data()

    test_iter = iter(test_loader)
    test_img, labels = test_iter.next()

    noisy_test_img = test_img + 0.5*torch.randn(*test_img.shape)

    output = model(noisy_test_img)
    output = output.view(100, 1, 28, 28)
    output = output.detach().numpy()
    output_img = np.transpose(output[0], (1, 2, 0)) * 255

    input = noisy_test_img.numpy()
    input_img = np.transpose(input[0], (1, 2, 0)).astype('uint8') * 255

    org_input = test_img.numpy()
    org_input_img = np.transpose(org_input[0], (1, 2, 0))*255

    for i in range(1, 10):
         temp = np.transpose(org_input[i], (1, 2, 0)) * 255
         org_input_img = np.hstack((org_input_img, temp))

         temp =  np.transpose(output[i], (1, 2, 0))*255
         output_img = np.hstack((output_img, temp))

         temp = np.transpose(input[i], (1, 2, 0)).astype('uint8') * 255
         input_img = np.hstack((input_img, temp))

    input_img = np.vstack((org_input_img, input_img))
    input_img = np.vstack((input_img, output_img))
    helper.image_show(input_img)
    cv2.imwrite('denoised' + '.png', input_img)


if __name__ == '__main__':
    train(20)
    display()