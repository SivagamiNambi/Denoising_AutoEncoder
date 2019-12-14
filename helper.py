import torch
import torchvision
import cv2
from torchvision import transforms
import numpy as np

transforms = transforms.Compose([transforms.ToTensor()])

def image_show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data():
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)

    print('Length of trainset:', len(train_set))
    print('Length of testset:', len(test_set))
    print('Length of valset:', len(val_set))

    return train_loader, val_loader, test_loader


def check_image(train_loader, val_loader, test_loader):
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    image, label = train_iter.next()
    img = image[0].numpy()
    img = np.transpose(img, (2, 1, 0))
    image_show(img)


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()
    check_image(train_loader, val_loader, test_loader)