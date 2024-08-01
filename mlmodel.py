import collections
import os

import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

Model = collections.namedtuple('Model', 'phi h options')


class Phi_Net(nn.Module):
    def __init__(self, options):
        super(Phi_Net, self).__init__()

        self.fc1 = nn.Linear(options['dim_x'], options['phi_first_out'])
        self.fc2 = nn.Linear(options['phi_first_out'], options['phi_second_out'])
        self.fc3 = nn.Linear(options['phi_second_out'], options['phi_first_out'])
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(options['phi_first_out'], options['dim_a']-1)
        self.options = options
        self.device = self.options["device"]
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1).to(self.device)])
        else:
            # batch input for training
            return torch.cat([x, torch.ones([x.shape[0], 1]).to(self.device)], dim=-1)

class E2E_Phi_Net(nn.Module):
    def __init__(self, options):
        super(E2E_Phi_Net, self).__init__()

        self.fc1 = nn.Linear(options['dim_x'], options['phi_first_out'])
        self.fc2 = nn.Linear(options['phi_first_out'], options['phi_second_out'])
        self.fc3 = nn.Linear(options['phi_second_out'], options['phi_first_out'])
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(options['phi_first_out'], options['dim_y'])

        self.options = options
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
 
# Cross-entropy loss
class H_Net_CrossEntropy(nn.Module):
    def __init__(self, options):
        super(H_Net_CrossEntropy, self).__init__()
        # self.fc1 = nn.Linear(options['dim_a'], 20)
        # self.fc2 = nn.Linear(20, options['num_c'])
        self.fc1 = nn.Linear(options['dim_a'], options['discrim_hidden'])
        self.fc2 = nn.Linear(options['discrim_hidden'], options['num_c'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Cross-entropy loss
class E2E_H_Net_CrossEntropy(nn.Module):
    def __init__(self, options):
        super(E2E_H_Net_CrossEntropy, self).__init__()
        # self.fc1 = nn.Linear(options['dim_a'], 20)
        # self.fc2 = nn.Linear(20, options['num_c'])
        self.fc1 = nn.Linear(options['dim_y'], options['discrim_hidden'])
        self.fc2 = nn.Linear(options['discrim_hidden'], options['num_c'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# multi-basis functions
class Phi_Net_Multi(nn.Module):
    def __init__(self, options):
        super(Phi_Net_Multi, self).__init__()
        self.fc1 = nn.Linear(options['dim_x'], options['phi_first_out'])
        self.fc2 = nn.Linear(options['phi_first_out'], options['phi_second_out'])
        self.fc3 = nn.Linear(options['phi_second_out'], options['phi_first_out'])
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(options['phi_first_out'], (options['dim_a']-1)*options["num_legs"])
        self.options = options
        self.device = self.options["device"]
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# multi-basis functions
class H_Net_Multi_CrossEntropy(nn.Module):
    def __init__(self, options):
        super(H_Net_Multi_CrossEntropy, self).__init__()
        # self.fc1 = nn.Linear(options['dim_a'], 20)
        # self.fc2 = nn.Linear(20, options['num_c'])
        self.fc1 = nn.Linear((options['dim_a']-1)*options["num_legs"], options['discrim_hidden'])
        self.fc2 = nn.Linear(options['discrim_hidden'], options['num_c'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def _add_bias_term(net_output, options):
    if len(net_output.shape) == 1:
        # single input
        net_output = torch.cat([net_output, torch.ones(1).to(options["device"])])
    else:
        # batch input for training
        net_output = torch.cat([net_output, torch.ones([net_output.shape[0], 1]).to(options["device"])], dim=-1)

    return net_output

def _perform_least_squares(Phi, Y, options):
    Phi_T = Phi.transpose(0, 1) # dim_a x K
    A = torch.inverse(torch.mm(Phi_T, Phi)) # dim_a x dim_a
    a = torch.mm(torch.mm(A, Phi_T), Y) # dim_a x dim_y
    if torch.norm(a, 'fro') > options["gamma"]:
        a = a / torch.norm(a, 'fro') * options["gamma"]
    # push A off the device
    A.cpu()
    return a

def update_mixing_params(net, X, Y, options):
    '''
    Least-squares to get $a$ from K-shot data
    '''
    # process all the input data
    phi_total = net(X)

    a_total = []

    for leg in range(0, options["num_legs"]):
        # split into per-leg sections
        phi = phi_total[:,leg*(options["dim_a"]-1):(leg+1)*(options["dim_a"]-1)]

        # add bias term to leg-output
        phi = _add_bias_term(phi, options)

        y = Y[:,leg*options["num_joints"]:(leg+1)*options["num_joints"]]
        _a = _perform_least_squares(phi, y, options)
        a_total.append(_a)

    return a_total

def predict_residual_torques(net, x, a, options):
    phi_output = net(x)

    tau_total = []

    for leg in range(0, options["num_legs"]):
        # split into per-leg sections
        phi_out = phi_output[:,leg*(options["dim_a"]-1):(leg+1)*(options["dim_a"]-1)]
        
        # add bias term to leg-output
        phi_out = _add_bias_term(phi_out, options)
        
        _tau = torch.mm(phi_out, a[leg])
        tau_total.append(_tau)

    if len(phi_output.shape) == 1:
        tau_total = torch.cat(tau_total)
    else:
        tau_total = torch.cat(tau_total, dim=-1)

    return tau_total
    
def save_model(*, phi_net, h_net, modelname, options):
    _output_path = os.path.join(options['output_path'], 'models')
    if not os.path.isdir(_output_path):
        os.makedirs(_output_path)
    if h_net is not None:
        torch.save({
            'phi_net_state_dict': phi_net.to('cpu').state_dict(),
            'h_net_state_dict': h_net.to('cpu').state_dict(),
            'options': dict(options)
        }, os.path.join(_output_path, (modelname + '.pth')))
    else:
        torch.save({
            'phi_net_state_dict': phi_net.state_dict(),
            'h_net_state_dict': None,
            'options': dict(options)
        }, os.path.join(_output_path, (modelname + '.pth')))


def load_model(modelname):
    model = torch.load(modelname + '.pth')
    options = model['options']

    phi_net = Phi_Net(options=options)
    # h_net = H_Net_CrossEntropy(options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])
    # h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    # h_net.eval()

    return Model(phi_net, h_net, options)

def load_model_e2e(modelname):
    model = torch.load(modelname + '.pth')
    options = model['options']

    phi_net = E2E_Phi_Net(options=options)
    # h_net = H_Net_CrossEntropy(options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])
    # h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    # h_net.eval()

    return Model(phi_net, h_net, options)


def load_model_eval(modelname, modelfolder='./models/'):
    model = torch.load(modelfolder + modelname + '.pth')
    options = model['options']

    phi_net = Phi_Net(options=options)
    # h_net = H_Net_CrossEntropy(options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])
    # h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    # h_net.eval()

    return Model(phi_net, h_net, options)


class MyDataset(Dataset):

    def __init__(self, inputs, outputs, c):
        self.inputs = inputs
        self.outputs = outputs
        self.c = c

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Input = self.inputs[idx,]
        output = self.outputs[idx,]
        sample = {'input': Input, 'output': output, 'c': self.c}

        return sample


_softmax = nn.Softmax(dim=1)

## Training Helper Functions

def validation(phi_net, h_net, adaptinput: np.ndarray, adaptlabel: np.ndarray, valinput: np.ndarray, options, lam=0):
    """
    Helper function for compute the output given a sequence of data for adaptation (adaptinput)
    and validation (valinput)

    adaptinput: K x dim_x numpy, adaptlabel: K x dim_y numpy, valinput: B x dim_x numpy
    output: K x dim_y numpy, B x dim_y numpy, dim_a x dim_y numpy, B x dim_c numpy
    """
    phi_net.to(options['device'])
    h_net.to(options['device'])
    
    with torch.no_grad():       
        # Perform least squares on the adaptation set to get a
        X = torch.from_numpy(adaptinput).to(options['device']) # K x dim_x
        Y = torch.from_numpy(adaptlabel).to(options['device']) # K x dim_y
        Phi = phi_net(X) # K x dim_a
        Phi_T = Phi.transpose(0, 1) # dim_a x K
        A = torch.inverse(torch.mm(Phi_T, Phi) + lam*torch.eye(options['dim_a']).to(options['device'])) # dim_a x dim_a
        a = torch.mm(torch.mm(A, Phi_T), Y) # dim_a x dim_y
        
        # Compute NN prediction for the validation and adaptation sets
        inputs = torch.from_numpy(valinput).to(options['device']) # B x dim_x
        val_prediction = torch.mm(phi_net(inputs), a) # B x dim_y
        adapt_prediction = torch.mm(phi_net(X), a) # K x dim_y
        
        # Compute adversarial network prediction
        temp = phi_net(inputs)
        if h_net is None:
            h_output = None
        else:
            h_output = h_net(temp) # B x num_of_c (CrossEntropy-loss) or B x dim_c (c-loss) or B x (dim_y*dim_a) (a-loss) 
            if options['loss_type'] == 'crossentropy-loss':
                # Cross-Entropy
                h_output = _softmax(h_output)
            h_output = h_output.to('cpu').numpy()
        
        temp.cpu()
        A.cpu()
        inputs.cpu()
        X.cpu()
        Y.cpu()
        Phi.cpu()
        Phi_T.cpu()
    
    return adapt_prediction.to('cpu').numpy(), val_prediction.to('cpu').numpy(), a.to('cpu').numpy(), h_output

def vis_validation(*, t, x, y, phi_net, h_net, idx_adapt_start, idx_adapt_end, idx_val_start, idx_val_end, c, options, output_path_prefix, output_name, lam=0):
    """
    Visualize performance with adaptation on x[idx_adapt_start:idx_adapt_end] and validation on x[idx_val_start:idx_val_end]
    """
    adaptinput = x[idx_adapt_start:idx_adapt_end, :]
    valinput = x[idx_val_start:idx_val_end, :]
    adaptlabel = y[idx_adapt_start:idx_adapt_end, :]

    y_adapt, y_val, a, h_output = validation(phi_net, h_net, adaptinput, adaptlabel, valinput, options, lam=lam)
    # print(f'a = {a}')
    # print(f"|a| = {np.linalg.norm(a,'fro')}")

    idx_min = min(idx_adapt_start, idx_val_start)
    idx_max = max(idx_adapt_end, idx_val_end)

    plt.figure(figsize=(15, 12))

    leg_labels = ["FR", "FL", "RR", "RL"]
    joint_labels = ["HIP", "KNEE", "FOOT"]
    # axis_range = [-60, 30]

    fig, axs = plt.subplots(4, 3, figsize=(15,12))


    row = 0
    col = 0
    for idx in range(len(y[0])):
        axs[row, col].plot(t[idx_min:idx_max], y[idx_min:idx_max, idx], 'k', alpha=0.3, label='gt')
        axs[row, col].plot(t[idx_val_start:idx_val_end], y_val[:, idx], label='val')
        axs[row, col].plot(t[idx_adapt_start:idx_adapt_end], y_adapt[:, idx], label='adapt')
        axs[row, col].legend()
        axs[row, col].set_title("{:s}, {:s}".format(leg_labels[row], joint_labels[col]))
        idx += 1
        col += 1

        if idx % 3 == 0:
            row += 1
            col = 0

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    fig.savefig(os.path.join(output_path_prefix, output_name))
    
    plt.close(fig)

    # if h_output is not None:
    #     plt.subplot(1, 4, 4)
    #     if options['loss_type'] == 'c-loss':
    #         colors = ['red', 'blue', 'green']
    #         for i in range(options['dim_c']):
    #             plt.plot(h_output[:, i], color=colors[i], label='c'+str(i))
    #             plt.hlines(c[i], xmin=0, xmax=len(h_output), linestyles='--', color=colors[i], label='c'+str(i)+' gt')
    #         plt.legend()
    #         plt.title('c prediction')
    #     if options['loss_type'] == 'crossentropy-loss':
    #         plt.plot(h_output)
    #         plt.title('c prediction (after Softmax)')
    #     if options['loss_type'] == 'a-loss':
    #         a_gt = a.reshape(1, options['dim_a'] * options['dim_y'])
    #         plt.plot(h_output - np.repeat(a_gt, h_output.shape[0], axis=0))
    #         # plt.hlines(a_gt, xmin=Data['time'][idx_val_start], xmax=Data['time'][idx_val_end]-1, linestyles='--')
    #         plt.title('a prediction')
    # plt.show()

def error_statistics(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        _, prediction, _, _ = validation(phi_net, h_net, data_input, data_output, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()



        return error_1, error_2, error_3

def error_statistics_hist(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options, output_path_prefix, output_name):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        _, prediction, _, _ = validation(phi_net, h_net, data_input, data_output, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()


        # make errors histrogram
        #     make MSE values
        errors = data_output - prediction
        counts, bins = np.histogram(errors)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Prediction Error")
        plt.title("Prediction Error Hist.")
        
        if not os.path.exists(output_path_prefix):
            os.makedirs(output_path_prefix)
        
        plt.savefig(os.path.join(output_path_prefix, output_name))

        plt.close()

        return error_1, error_2, error_3
    
def validation_e2e(phi_net, h_net, valinput: np.ndarray, options, lam=0):
    """
    Helper function for compute the output given a sequence of data for adaptation (adaptinput)
    and validation (valinput)

    adaptinput: K x dim_x numpy, adaptlabel: K x dim_y numpy, valinput: B x dim_x numpy
    output: K x dim_y numpy, B x dim_y numpy, dim_a x dim_y numpy, B x dim_c numpy
    """
    with torch.no_grad():
        # Compute NN prediction for the validation and adaptation sets
        inputs = torch.from_numpy(valinput).to(options['device']) # B x dim_x
        val_prediction = phi_net(inputs) # B x dim_y
        
        # Compute adversarial network prediction
        temp = phi_net(inputs)
        if h_net is None:
            h_output = None
        else:
            h_output = h_net(temp) # B x num_of_c (CrossEntropy-loss) or B x dim_c (c-loss) or B x (dim_y*dim_a) (a-loss) 
            if options['loss_type'] == 'crossentropy-loss':
                # Cross-Entropy
                h_output = _softmax(h_output)
            h_output = h_output.to('cpu').numpy()
        
        temp.cpu()
        inputs.cpu()

    return val_prediction.to('cpu').numpy(), h_output

def vis_validation_e2e(*, t, x, y, phi_net, h_net, idx_val_start, idx_val_end, c, options, output_path_prefix, output_name, lam=0):
    """
    Visualize performance with adaptation on x[idx_adapt_start:idx_adapt_end] and validation on x[idx_val_start:idx_val_end]
    """
    valinput = x[idx_val_start:idx_val_end, :]

    y_val, h_output = validation_e2e(phi_net, h_net, valinput, options, lam=lam)
    # print(f'a = {a}')
    # print(f"|a| = {np.linalg.norm(a,'fro')}")

    idx_min = idx_val_start
    idx_max = idx_val_end

    plt.figure(figsize=(15, 12))

    leg_labels = ["FR", "FL", "RR", "RL"]
    joint_labels = ["HIP", "KNEE", "FOOT"]
    # axis_range = [-60, 30]

    fig, axs = plt.subplots(4, 3, figsize=(15,12))


    row = 0
    col = 0
    for idx in range(len(y[0])):
        axs[row, col].plot(t[idx_min:idx_max], y[idx_min:idx_max, idx], 'k', alpha=0.3, label='gt')
        axs[row, col].plot(t[idx_val_start:idx_val_end], y_val[:, idx], label='val')
        axs[row, col].legend()
        axs[row, col].set_title("{:s}, {:s}".format(leg_labels[row], joint_labels[col]))
        idx += 1
        col += 1

        if idx % 3 == 0:
            row += 1
            col = 0

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    fig.savefig(os.path.join(output_path_prefix, output_name))
    
    plt.close(fig)

def error_statistics_e2e(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        prediction, _ = validation_e2e(phi_net, h_net, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()



        return error_1, error_2, error_3

def error_statistics_hist_e2e(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options, output_path_prefix, output_name):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        prediction, _ = validation_e2e(phi_net, h_net, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()

        # make errors histrogram
        #     make MSE values
        errors = data_output - prediction
        counts, bins = np.histogram(errors)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Prediction Error")
        plt.title("Prediction Error Hist.")
        
        if not os.path.exists(output_path_prefix):
            os.makedirs(output_path_prefix)
        
        plt.savefig(os.path.join(output_path_prefix, output_name))

        plt.close()

        return error_1, error_2, error_3
    
def validation_multi_basis(phi_net, h_net, adaptinput: np.ndarray, adaptlabel: np.ndarray, valinput: np.ndarray, options, lam=0):
    """
    Helper function for compute the output given a sequence of data for adaptation (adaptinput)
    and validation (valinput)

    adaptinput: K x dim_x numpy, adaptlabel: K x dim_y numpy, valinput: B x dim_x numpy
    output: K x dim_y numpy, B x dim_y numpy, dim_a x dim_y numpy, B x dim_c numpy
    """
    phi_net.to(options['device'])
    h_net.to(options['device'])
    
    with torch.no_grad():       
        # Perform least squares on the adaptation set to get a
        X = torch.from_numpy(adaptinput).to(options['device']) # K x dim_x
        Y = torch.from_numpy(adaptlabel).to(options['device']) # K x dim_y
        a = update_mixing_params(phi_net, X, Y, options)
        
        
        # Compute NN prediction for the validation and adaptation sets
        inputs = torch.from_numpy(valinput).to(options['device']) # B x dim_x

        val_prediction = predict_residual_torques(phi_net, inputs, a, options) # B x dim_y
        
        adapt_prediction = predict_residual_torques(phi_net, X, a, options) # K x dim_y
        
        # Compute adversarial network prediction
        temp = phi_net(inputs)
        if h_net is None:
            h_output = None
        else:
            h_output = h_net(temp) # B x num_of_c (CrossEntropy-loss) or B x dim_c (c-loss) or B x (dim_y*dim_a) (a-loss) 
            if options['loss_type'] == 'crossentropy-loss':
                # Cross-Entropy
                h_output = _softmax(h_output)
            h_output = h_output.to('cpu').numpy()
        
        temp.cpu()
        inputs.cpu()
        X.cpu()
        Y.cpu()

        a = np.array([leg_a.to('cpu').numpy() for leg_a in a])
    
    return adapt_prediction.to('cpu').numpy(), val_prediction.to('cpu').numpy(), a, h_output

def vis_validation_multi(*, t, x, y, phi_net, h_net, idx_adapt_start, idx_adapt_end, idx_val_start, idx_val_end, c, options, output_path_prefix, output_name, lam=0):
    """
    Visualize performance with adaptation on x[idx_adapt_start:idx_adapt_end] and validation on x[idx_val_start:idx_val_end]
    """
    adaptinput = x[idx_adapt_start:idx_adapt_end, :]
    valinput = x[idx_val_start:idx_val_end, :]
    adaptlabel = y[idx_adapt_start:idx_adapt_end, :]

    y_adapt, y_val, a, h_output = validation_multi_basis(phi_net, h_net, adaptinput, adaptlabel, valinput, options, lam=lam)
    # print(f'a = {a}')
    # print(f"|a| = {np.linalg.norm(a,'fro')}")

    idx_min = min(idx_adapt_start, idx_val_start)
    idx_max = max(idx_adapt_end, idx_val_end)

    plt.figure(figsize=(15, 12))

    leg_labels = ["FR", "FL", "RR", "RL"]
    joint_labels = ["HIP", "KNEE", "FOOT"]
    # axis_range = [-60, 30]

    fig, axs = plt.subplots(4, 3, figsize=(15,12))


    row = 0
    col = 0
    for idx in range(len(y[0])):
        axs[row, col].plot(t[idx_min:idx_max], y[idx_min:idx_max, idx], 'k', alpha=0.3, label='gt')
        axs[row, col].plot(t[idx_val_start:idx_val_end], y_val[:, idx], label='val')
        axs[row, col].plot(t[idx_adapt_start:idx_adapt_end], y_adapt[:, idx], label='adapt')
        axs[row, col].legend()
        axs[row, col].set_title("{:s}, {:s}".format(leg_labels[row], joint_labels[col]))
        idx += 1
        col += 1

        if idx % 3 == 0:
            row += 1
            col = 0

    if not os.path.exists(output_path_prefix):
        os.makedirs(output_path_prefix)

    fig.savefig(os.path.join(output_path_prefix, output_name))
    
    plt.close(fig)

def error_statistics_multi(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        _, prediction, _, _ = validation_multi_basis(phi_net, h_net, data_input, data_output, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()



        return error_1, error_2, error_3
    
def error_statistics_hist_multi(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options, output_path_prefix, output_name):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        torch_data = torch.from_numpy(data_output).to(options['device'])
        error_1 = criterion(torch_data, 0.0*torch.from_numpy(data_output).to(options['device'])).to('cpu').item()
        error_2 = criterion(torch_data, torch.from_numpy(np.ones((len(data_output), 1))).to(options['device']) * torch.from_numpy(np.mean(data_output, axis=0)[np.newaxis, :]).to(options['device'])).to('cpu').item()

        _, prediction, _, _ = validation_multi_basis(phi_net, h_net, data_input, data_output, data_input, options=options)
        error_3 = criterion(torch_data, torch.from_numpy(prediction).to(options['device'])).to('cpu').item()


        # make errors histrogram
        #     make MSE values
        errors = data_output - prediction
        counts, bins = np.histogram(errors)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.ylabel("Prediction Error")
        plt.title("Prediction Error Hist.")
        
        if not os.path.exists(output_path_prefix):
            os.makedirs(output_path_prefix)
        
        plt.savefig(os.path.join(output_path_prefix, output_name))

        plt.close()

        return error_1, error_2, error_3