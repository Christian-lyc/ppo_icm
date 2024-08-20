import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
class Dataset():
    def __init__(self,traindir,testdir,args):
        self.traindir=traindir
        self.testdir=testdir
        self.args=args
        self.test_dataset = datasets.ImageFolder(self.testdir,
                                            transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])
                                            ]))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                                       num_workers=self.args.workers, pin_memory=True)
    def sample_data(self):
        train_dataset = datasets.ImageFolder(self.traindir,
                                             transforms.Compose([
                                             transforms.Resize((256, 256)),
                                             transforms.RandomCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomAutocontrast(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                            ]))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                    num_workers=self.args.workers,pin_memory=True)
        while True:
            for idx in self.train_loader:
                yield idx

    def test_sample_data(self):

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                                   num_workers=self.args.workers,pin_memory=True)
        while True:
            for idx in self.test_loader:
                yield idx
    def get_testlen(self):
        return len(self.test_loader)
