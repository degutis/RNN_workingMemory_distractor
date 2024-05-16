import numpy as np
import torch
from torch import nn
from torch import optim
import os

class LSTM(nn.Module):
    def __init__(self, input_size, n_hidden, n_layer):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, n_hidden, n_layer)
        self.out = nn.Linear(n_hidden, 2)
    def forward(self, input):
        h, _ = self.lstm(input)
        y = self.out(torch.relu(h))
        return h, y.squeeze().transpose(dim0=1, dim1=0)


class TrainTestModel:
    def __init__(self, lengthDistractor, lengthExtraDelay, seed):

        self.seed = seed
        self.bins = [np.arange(1,31),np.arange(31,61),np.arange(61,91), \
            np.arange(-29,1),np.arange(-59,-29),np.arange(-89,-59)]
#        self.bins = [np.arange(0,30),np.arange(30,60),np.arange(60,90), \
#            np.arange(90,120),np.arange(120,150),np.arange(150,180)]
        self.bins_distractor = self.bins
        self.numRuns = 300
        self.lengthDistractor = lengthDistractor
        self.lengthExtraDelay = lengthExtraDelay

    def generateTrials(self):

        numBins = len(self.bins)
        numIndepTrials = numBins*numBins

        fullTrial_matrix = np.empty((numIndepTrials,2,self.numRuns), int)
        numDistPositions = self.lengthDistractor
        additionalDelay = self.lengthExtraDelay
        numTimePoints = 1+numDistPositions+additionalDelay*2
        totalTime_per_run = numTimePoints*numIndepTrials
        #trialSequence = np.empty((self.numRuns,totalTime_per_run),int)
        # 91 will represent an empty delay by not activating the basis functions
        trialSequence = np.full((self.numRuns,totalTime_per_run),91) 
        
        for run in range(self.numRuns):
        
            targets = []
            for iter in range(0,numBins):
                for a in self.bins:
                    targets.append(int(np.random.choice(a,1)[0]))
            
            distractor = []
            for iter in range(0,numBins):
                currentTrials = []
                for a in self.bins_distractor:
                    currentTrials.append(int(np.random.choice(a,1)[0]))
                shiftedOr = currentTrials[-1*iter:]+currentTrials[:-1*iter]
                [distractor.append(i) for i in shiftedOr]    
            
            
            trialMatrix = np.empty((numIndepTrials,2))
            trialMatrix[:,0] = np.array(targets)
            trialMatrix[:,1] = np.array(distractor)

            idx = np.random.rand(*trialMatrix.shape).argsort(axis=0)[:,0]
            idx_same = np.empty((numIndepTrials,2), int)
            idx_same[:,0] = idx
            idx_same[:,1] = idx
            shuffled_trialMatrix = np.take_along_axis(trialMatrix,idx_same,axis=0)

            fullTrial_matrix[:,:,run] = shuffled_trialMatrix

            trialSequence[run,0:totalTime_per_run:numTimePoints] = shuffled_trialMatrix[:,0]
            trialSequence[run,1:totalTime_per_run:numTimePoints] = 91
            for distPosition in range(1+additionalDelay,1+numDistPositions+additionalDelay):
                trialSequence[run,distPosition:totalTime_per_run:numTimePoints] = shuffled_trialMatrix[:,1]

        return fullTrial_matrix, trialSequence, numTimePoints                  


    def make_basis_function(self,xx, mu, n_ori_chans):
        return np.power(np.cos(np.deg2rad(xx - mu)), n_ori_chans - np.mod(n_ori_chans, 2))
  
    def basisFunc(self):
        self.n_ori_chans = 8
        xx = np.linspace(1, 180, 180)
        basis_set = np.zeros((180, self.n_ori_chans))
        chan_center = np.linspace(180 / self.n_ori_chans, 180, self.n_ori_chans)

        for cc in range(self.n_ori_chans):
            basis_set[:, cc] = self.make_basis_function(xx, chan_center[cc], self.n_ori_chans)
        
        basis_set = np.concatenate([basis_set, np.zeros((1,8))],axis=0)

        return basis_set
    
    def loss_fn(self, outputs, targets): 
        total_err = torch.zeros(1)
        resp_tp = np.arange(0,outputs.size(1),self.numTimePoints)
        predicted_angle = np.empty((outputs.size(0),resp_tp.size))
        for i in range(outputs.size(0)): #batch
            batch_err = torch.zeros(1)
            for idx, j in enumerate(resp_tp):
                batch_err = self.circular_mean_sq_error(targets[i,j],outputs[i,j+self.numTimePoints-1,:],resp_tp)
                predicted_angle[i,idx] = self.calc_predicted_angle(outputs[i,j+self.numTimePoints-1,:].detach().numpy())
            total_err += batch_err
        return total_err, predicted_angle
   

    def circular_mean_sq_error(self,angle,angle_hat,resp_tp):
        d = (np.sin(angle)-angle_hat[1])**2 + (np.cos(angle)-angle_hat[0])**2
        return d/len(resp_tp)

    def calc_predicted_angle(self,angle_hat):

        sin = angle_hat[1]
        cos = angle_hat[0]        
        return  np.arctan2(sin,cos)  # sine cosine   


    def trainModel(self):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.matrix, self.trialSequence, self.numTimePoints = self.generateTrials()

        self.matrix_orig = self.matrix+89
        self.trialSequence_orig = self.trialSequence+89

        self.matrix = np.radians(self.matrix*2)
        self.trialSequence = np.radians(self.trialSequence*2)

        self.batch_size = 20
        self.n_layer = 2
        self.n_timestep = self.trialSequence.shape[1] # total timepoints within a run
        self.learning_rate = 1e-3
        self.n_sequence = self.trialSequence.shape[0]
        self.n_hidden = 150
        self.n_iter = 1200 # iterations of SGD

        self.basis_set = self.basisFunc()
        self.input_size = self.n_ori_chans

        track_loss = np.zeros(self.n_iter)
        self.model2 = LSTM(self.input_size, self.n_hidden, self.n_layer)
        optimizer = optim.Adam(self.model2.parameters(), lr=self.learning_rate)

        accuracy_total = np.empty((self.batch_size,len(self.bins)*len(self.bins),self.n_iter))
        predicted_angle_total = np.empty((self.batch_size,len(self.bins)*len(self.bins),self.n_iter))
        stim_angle = np.empty((self.batch_size,len(self.bins)*len(self.bins),self.n_iter))
        self.stim_index = np.arange(0, self.trialSequence.shape[1],self.numTimePoints)

        dist_angle = np.empty((self.batch_size,len(self.bins)*len(self.bins),self.n_iter))

        # Loop over iterations
        t = 0
        for i in range(self.n_iter):
            if (i + 1) %  100 == 0: # print progress every 100 iterations
                print('%.2f%% iterations of Adam completed...loss = %.2f'  % (100* (i + 1) / self.n_iter, loss*10))
            input_mat = torch.zeros(self.n_timestep,self.batch_size, self.input_size)
            batch_idx = np.random.choice(self.n_sequence, self.batch_size)
            stim = self.trialSequence[batch_idx,:].transpose()
            stim_orig = self.trialSequence_orig[batch_idx,:].transpose()

            input_mat = torch.from_numpy(self.basis_set[stim_orig,:]).to(torch.float32)

            #input_mat[:,:,0] = torch.from_numpy(np.sin(stim))
            #input_mat[:,:,1] = torch.from_numpy(np.cos(stim))

            hidden, outputs = self.model2.forward(input_mat)  #LSTM model
            loss, predicted_angle = self.loss_fn(outputs, stim.transpose())
            predicted_angle_total[:,:,i] = predicted_angle
            stim2 = self.trialSequence[batch_idx, :]
            stim_angle[:,:,i] = stim2[:,self.stim_index]
            dist_angle[:,:,i] = stim2[:,self.stim_index+2]
            
            track_loss[t] = loss
        # Compute gradients
            optimizer.zero_grad()
            loss.backward()
        # Update weights
            optimizer.step()
            t += 1    

    def forwardPassModel(self):
        
        self.trainModel()

        n_timestep = self.trialSequence.shape[1] # total timepoints within a run
        n_sequence = self.trialSequence.shape[0]
        batch_size = 300
        input_mat = torch.zeros(self.n_timestep, batch_size, self.input_size)
        self.stim_full = self.trialSequence.transpose()

        self.stim_full_orig = self.trialSequence_orig.transpose()

        #input_mat[:,:,0] = torch.from_numpy(np.sin(self.stim_full))
        #input_mat[:,:,1] = torch.from_numpy(np.cos(self.stim_full))

        input_mat = torch.from_numpy(self.basis_set[self.stim_full_orig,:]).to(torch.float32)

        self.hidden, self.outputs = self.model2.forward(input_mat)  #LSTM model
        self.loss, self.predicted_angle = self.loss_fn(self.outputs, self.stim_full.transpose())

        stim_a = self.trialSequence[:,self.stim_index]
        dist_a = self.trialSequence[:,self.stim_index+2]

        # Behaviorally relevant data converted into degrees
        self.stimuli = np.rad2deg(stim_a/2)
        self.distractor = np.rad2deg(dist_a/2)
        self.predicted_angle = np.rad2deg(self.predicted_angle/2)
        self.stim_full_inDegrees = np.rad2deg(self.stim_full/2)+89 # shifting output to [0 180)

        path2save = os.path.join("results","RNN_models", f"150_units_lengthDelay_{self.numTimePoints}_seed_{self.seed}.pt")
        torch.save(self.model2.state_dict(), path2save)



