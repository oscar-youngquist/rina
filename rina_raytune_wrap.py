import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import os 
import utils
import mlmodel
from datetime import datetime
import re
import sys
import json
import matplotlib.pyplot as plt


class RINA_Tune():

    def __init__(self, options):
        if sys.platform == 'win32':
            NUM_WORKERS = 0 # Windows does not support multiprocessing
        else:
            NUM_WORKERS = 2
        print('running on ' + sys.platform + ', setting ' + str(NUM_WORKERS) + ' workers')

        # Extract some global values
        self.options = options
        self.dim_a = options["dim_a"]
        self.features = options["features"]
        self.label = options["label"]
        self.labels = options["labels"]
        self.dataset = options["dataset"]
        self.lr = options["learning_rate"]
        self.model_save_freq = options["model_save_freq"]
        self.epochs = options['num_epochs']
        self.gamma = options["gamma"]
        self.alpha = options['alpha']
        self.discrim_save_freq = options['frequency_h']
        self.sn = options['SN']
        self.train_data_path = options['train_path']
        self.test_data_path = options['test_path']
        self.output_path_base = options['output_path']
        
        print("\n***********Model output path: ", self.output_path_base)

        RawData = utils.load_data(self.train_data_path)
        Data = utils.format_data(RawData, features=self.features, output=self.label, body_offset=options["body_offset"])

        RawDataTest = utils.load_data(self.test_data_path) # expnames='(baseline_)([0-9]*|no)wind'
        self.TestData = utils.format_data(RawDataTest, features=self.features, output=self.label, body_offset=options["body_offset"])

        # Update options dict based on the shape of the data
        options['dim_x'] = Data[0].X.shape[1]
        options['dim_y'] = Data[0].Y.shape[1]
        options['num_c'] = len(Data)

        self.num_train_classes = options["num_c"]
        self.num_test_classes = len(self.TestData)

        # Make the dataloaders globally accessible
        self.Trainloader = []
        self.Adaptloader = []
        for i in range(self.num_train_classes ):
            fullset = mlmodel.MyDataset(Data[i].X, Data[i].Y, Data[i].C)
            
            l = len(Data[i].X)
            if options['shuffle']:
                trainset, adaptset = random_split(fullset, [int(2/3*l), l-int(2/3*l)])
            else:
                trainset = mlmodel.MyDataset(Data[i].X[:int(2/3*l)], Data[i].Y[:int(2/3*l)], Data[i].C) 
                adaptset = mlmodel.MyDataset(Data[i].X[int(2/3*l):], Data[i].Y[int(2/3*l):], Data[i].C)

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=options['phi_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)
            adaptloader = torch.utils.data.DataLoader(adaptset, batch_size=options['K_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)

            self.Trainloader.append(trainloader) # for training phi
            self.Adaptloader.append(adaptloader) # for LS on a

        # Make model name
        self.modelname = f"{self.dataset}_dim-a-{self.dim_a}_{'-'.join(self.features)}"

        # Build the neural nets
        self.phi_net = mlmodel.Phi_Net(options)
        self.h_net = mlmodel.H_Net_CrossEntropy(options)

        # Create the losses
        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()

        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=self.lr)
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=self.lr)

        # Create some arrays to save training statistics
        self.Loss_f = [] # combined force prediction loss
        self.Loss_c = [] # combined adversarial loss

        # Loss for each subdataset 
        self.Loss_test_nominal = [] # loss without any learning
        self.Loss_test_mean = [] # loss with mean predictor
        self.Loss_test_phi = [] # loss with NN
        for i in range(0, self.num_test_classes):
            self.Loss_test_nominal.append([])
            self.Loss_test_mean.append([])
            self.Loss_test_phi.append([])

    
    def save_config(self, output_path):
        with open(output_path, "w+") as f:
            json.dump(self.options, f)
            f.close()

    
    def train_model(self):
        # Iterate over the desired number of epochs
        for epoch in range(0, self.epochs):
            # Randomize the order in which we train over the subdatasets
            arr = np.arange(self.num_train_classes)
            np.random.shuffle(arr)

            # Running loss over all subdatasets
            running_loss_f = 0.0
            running_loss_c = 0.0

            for i in arr:
                kshot_data = None
                data = None
                with torch.no_grad():
                    adaptloader = self.Adaptloader[i]
                    kshot_data = next(iter(adaptloader))
                    trainloader = self.Trainloader[i]
                    data = next(iter(trainloader))

                self.optimizer_phi.zero_grad()

                '''
                Least-square to get $a$ from K-shot data
                '''
                X = kshot_data['input'] # K x dim_x
                Y = kshot_data['output'] # K x dim_y
                Phi = self.phi_net(X) # K x dim_a
                Phi_T = Phi.transpose(0, 1) # dim_a x K
                A = torch.inverse(torch.mm(Phi_T, Phi)) # dim_a x dim_a
                a = torch.mm(torch.mm(A, Phi_T), Y) # dim_a x dim_y
                if torch.norm(a, 'fro') > self.gamma:
                    a = a / torch.norm(a, 'fro') * self.gamma

                '''
                Batch training \phi_net
                '''
                inputs = data['input'] # B x dim_x
                labels = data['output'] # B x dim_y
                
                c_labels = data['c'].type(torch.long)
                    
                # forward + backward + optimize
                outputs = torch.mm(self.phi_net(inputs), a)
                loss_f = self.criterion(outputs, labels)
                temp = self.phi_net(inputs)
                
                loss_c = self.criterion_h(self.h_net(temp), c_labels)
                    
                loss_phi = loss_f - self.alpha * loss_c
                loss_phi.backward()
                self.optimizer_phi.step()

                '''
                Discriminator training
                '''
                if np.random.rand() <= 1.0 / self.discrim_save_freq:
                    self.optimizer_h.zero_grad()
                    temp = self.phi_net(inputs).detach()
                    
                    loss_c = self.criterion_h(self.h_net(temp), c_labels)
                    
                    loss_h = loss_c
                    loss_h.backward()
                    self.optimizer_h.step()

                '''
                Spectral normalization
                '''
                if self.sn > 0:
                    for param in self.phi_net.parameters():
                        M = param.detach().numpy()
                        if M.ndim > 1:
                            s = np.linalg.norm(M, 2)
                            if s > self.sn:
                                param.data = param / s * self.sn
                
                running_loss_f += loss_f.item()
                running_loss_c += loss_c.item()
            
            # Save statistics
            self.Loss_f.append(running_loss_f / self.num_train_classes)
            self.Loss_c.append(running_loss_c / self.num_train_classes)
            # if epoch % 10 == 0:
            #     print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch + 1, running_loss_f / self.num_train_classes, running_loss_c / self.num_train_classes))
    
            with torch.no_grad():
                for j in range(len(self.TestData)):
                    loss_nominal, loss_mean, loss_phi = mlmodel.error_statistics(self.TestData[j].X, self.TestData[j].Y, self.phi_net, self.h_net, options=self.options)
                    self.Loss_test_nominal[j].append(loss_nominal)
                    self.Loss_test_mean[j].append(loss_mean)
                    self.Loss_test_phi[j].append(loss_phi)

            if epoch % self.model_save_freq == 0:
                mlmodel.save_model(phi_net=self.phi_net, h_net=self.h_net, modelname=self.modelname + '-epoch-' + str(epoch), options=self.options)

        # Save the model after the final batch of epochs
        mlmodel.save_model(phi_net=self.phi_net, h_net=self.h_net, modelname=self.modelname + '-epoch-' + str(self.epochs), options=self.options)

    def save_training_data(self):
        fig, axs = plt.subplots(2, 1)
        
        axs[0].plot(self.Loss_f, label="train")
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('f-loss [N]')
        axs[0].grid()
        axs[0].set_title('training f loss')
        
        axs[1].plot(self.Loss_c)
        axs[1].set_title('training c loss')
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('c-loss')
        axs[1].grid()

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_path_base, "training_loss_curves"))
        plt.close(fig)

        for j in range(len(self.TestData)):
            fig, axs = plt.subplots(1, 1)
            axs.plot(self.Loss_test_mean[j], label='mean')
            axs.plot(np.array(self.Loss_test_phi[j]), label='phi*a')
            axs.set_xlabel('epoch')
            axs.set_ylabel('f-loss [N]')
            axs.legend()
            axs.grid()
            axs.set_title(f'Test data set {j} - {self.TestData[j].meta["condition"]}')
            fig.savefig(os.path.join(self.output_path_base, f'test_data_set_{j}_{self.TestData[j].meta["condition"]}.png'))
            plt.close(fig)

        total_test_losses_mean = np.mean(self.Loss_test_phi,axis=1)
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(total_test_losses_mean, label='phi*a')
        axs.legend()
        axs.grid()
        axs.set_xlabel('epoch')
        axs.set_ylabel('f-loss [N]')
        axs.set_title(f'Avg Test-Data Loss')
        fig.savefig(os.path.join(self.output_path_base, f'test_set_avg_loss.png'))
        plt.close(fig)
        

    def eval_model(self, eval_adapt_start, eval_adapt_end, eval_val_start, eval_val_end, output_path_ims, output_path_txt):
        for i, data in enumerate(self.TestData):
            # print('------------------------------')
            # print(data.meta['condition'] + ':')
            # print(len(data.X))
            file_name = "{:s}.png".format(data.meta['condition'])
            mlmodel.vis_validation(t=data.meta['steps'], x=data.X, y=data.Y, phi_net=self.phi_net, h_net=self.h_net, 
                                idx_adapt_start=eval_adapt_start, idx_adapt_end=eval_adapt_end, 
                                idx_val_start=eval_val_start, idx_val_end=eval_val_end, c=self.TestData[i].C, options=self.options,
                                output_path_prefix=output_path_ims, output_name=file_name)
            
        error_befores = []
        error_means = []
        error_learned = []
        
        for data in self.TestData:
            image_name = "{:s}_errors_hist.png".format(data.meta['condition'])
            error_1, error_2, error_3 = mlmodel.error_statistics_hist(data.X, data.Y, self.phi_net, self.h_net, self.options, output_path_ims, image_name)
            print('**** :', data.meta['condition'], '****')
            print(f'Before learning: MSE is {error_1: .2f}')
            print(f'Mean predictor: MSE is {error_2: .2f}')
            print(f'After learning phi(x): MSE is {error_3: .2f}')
            print('')

            with open(output_path_txt, "a+") as f:
                f.write('**** :{} ****\n'.format(data.meta['condition']))
                f.write(f'Before learning: MSE is {error_1: .2f}\n')
                f.write(f'Mean predictor: MSE is {error_2: .2f}\n')
                f.write(f'After learning phi(x): MSE is {error_3: .2f}\n')
                f.write('\n')
                f.close()

            error_befores.append(error_1)
            error_means.append(error_2)
            error_learned.append(error_3)

        return error_befores, error_means, error_learned