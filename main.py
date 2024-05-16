# %% Import libraries and define functions
import pickle as pkl
import os
import pandas as pd

import model_new as model
import neuron_change
import plotting_subspaces
import behavior

#seeds = [1,2,3,4,5,6,7,8,9,10]
#seeds = [1,11,12,13,14,15,16,17,18,19]
seeds = [1111]

lengthDistractor = 8
lengthExtraDelay = 2

def openRNNFile(lengthDistractor, seed):
        path2load = os.path.join("..","results","RNN_models",f'distractorLength_{lengthDistractor}',f"RNN_model_150_lengthDistractor_{lengthDistractor}_seed_{seed}")
        with open(path2load,"rb") as openRNN:
                RNN_model = pkl.load(openRNN)
                return RNN_model   

# %% Initilize, train, and test the model

path2save = os.path.join("..","results","RNN_models", f'distractorLength_{lengthDistractor}')
os.makedirs(path2save,exist_ok=True)

for seed in seeds:
        newModel = model.TrainTestModel(lengthDistractor,lengthExtraDelay,seed)
        newModel.forwardPassModel()

        name2save = os.path.join(path2save,f"RNN_model_150_lengthDistractor_{lengthDistractor}_seed_{seed}")
        with open(name2save, "wb") as handle:
                pkl.dump(newModel,handle, protocol=pkl.HIGHEST_PROTOCOL)


# %% Behavior
path2save = os.path.join("..","results","figures", "behavior")
path2save2 = os.path.join("..","results","tables", "behavior")
os.makedirs(path2save,exist_ok=True)
os.makedirs(path2save2,exist_ok=True)

for seed in seeds:
        RNN_model = openRNNFile(lengthDistractor,seed)
        beh_matrix = behavior.runBehavioral(RNN_model.stimuli, RNN_model.distractor, RNN_model.predicted_angle, seed)

        name2save = os.path.join(path2save2,f"behavior_RNN_model_150_lengthDistractor_{lengthDistractor}_seed_{seed}")
        beh_matrix.to_csv(name2save)



# %% Behavior across models
path2save2 = os.path.join("..","results","tables","behavior")

behavior.combineBehavior(path2save2)



# %% Subspace Analysis

path2save = os.path.join("results","figures", "subspaces", f'distractorLength_{lengthDistractor}')
os.makedirs(path2save,exist_ok=True)

for seed in seeds:
        
        RNN_model = openRNNFile(lengthDistractor,seed)
        plotting_subspaces.plotSubspace_singleSub(RNN_model.stim_full_inDegrees, RNN_model.hidden, RNN_model.numTimePoints, 7, RNN_model.seed)
        angles_sub = plotting_subspaces.angles_plot(RNN_model.stim_full_inDegrees, RNN_model.hidden, RNN_model.numTimePoints, 7, RNN_model.seed)
        plotting_subspaces.dynamicSubspacesPlot_2D(RNN_model.stim_full_inDegrees, RNN_model.hidden, RNN_model.numTimePoints, 7, RNN_model.seed)


# %% Plot neurons
path2save = os.path.join("results","figures", "neuron_change", f'distractorLength_{lengthDistractor}')
os.makedirs(path2save,exist_ok=True)

for seed in seeds:
        RNN_model = openRNNFile(lengthDistractor,seed)
        neuron_change.plotNeurons_2d(RNN_model.stim_full_inDegrees, RNN_model.hidden,RNN_model.numTimePoints, 4, RNN_model.seed)                     

# %%
