# main
# This file aims to perform sentiment classification using various binarized embeddings


# Internal Imports
from utils import binary_embedding as embed

# External Imports
import os
import random
#import h5py
import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import nltk
import re
import string
import tensorflow_hub as hub

import lava.lib.dl.slayer as slayer

# Globals
NETWORK_TYPE="SNN"# One of: ANN, SNN
DATASET="emotion"# One of: imdb, emotion  #To be added: topic
DATA_FORMAT="word"# One of word, sentence
RATE_CODING=True# Enable rate-coded embeddings rather than binary. Only affects SNN models
SPIKE_RAND=False# Enable non-deterministic spikes. Only affects SNN models
MODEL_SIZE=[0, 256, 128, (2 if DATASET == "imdb" else 6)]# Set the first value to zero to auto-select the size of the input layer
NUM_EPOCHS=50# Number of training iterations
BATCH_SIZE=15# Number of samples per batch
LEARN_RATE=1e-5# Learning rate (for training)
MAX_TOKENS=(512 if DATASET == "imdb" else 32)# IMDB-512, Emotion-32 # Only affects the non-rate-coded word-level SNN
MAX_TIMESTEP=10#100# Doesn't affect the non-rate-coded word-level SNN
DATA_SPLITS={
    "train":.6,
    "valid":.1,
    "test":.3
}
TRAIN_MODEL=True
TEST_MODEL=True
VALIDATION_SET=True# Only affects model training
PROFILING=True# Turn this on to get operation counts (MACs for ANNs & Spike Counts for SNNs)
CUSTOM_NAME=""# Leave this blank to use a default name for both the model and log
MODEL_DIR="models/"
LOG_LOC="output/log/"
GRAPH_LOC="output/graph/"
GRAPH_VERTICALLY=True
RANDOM_SEED=2500
CLS_AMOUNT=50
# Generate some globals (Below lines shouldn't be modified)
MODEL_NAME=(CUSTOM_NAME if CUSTOM_NAME else f"{NETWORK_TYPE.lower()}_{DATA_FORMAT}_{'rate_' if RATE_CODING and NETWORK_TYPE == 'SNN' else ''}{'rand_' if SPIKE_RAND and NETWORK_TYPE == 'SNN' else ''}{DATASET}{'_{}'.format(MAX_TOKENS if (DATA_FORMAT == 'word' and not RATE_CODING) else MAX_TIMESTEP) if NETWORK_TYPE == 'SNN' else ''}")
MODEL_LOC=f"{MODEL_DIR}{MODEL_NAME}.pt"
LOG_NAME=(CUSTOM_NAME if CUSTOM_NAME else f"{NETWORK_TYPE.lower()}_{DATA_FORMAT}_{'rate_' if RATE_CODING and NETWORK_TYPE == 'SNN' else ''}{'rand_' if SPIKE_RAND and NETWORK_TYPE == 'SNN' else ''}{DATASET}{'_{}'.format(MAX_TOKENS if (DATA_FORMAT == 'word' and not RATE_CODING) else MAX_TIMESTEP) if NETWORK_TYPE == 'SNN' else ''}")
# Set library-specific globals & states
matplotlib.rcParams.update({'font.size': 14})
random.seed(RANDOM_SEED)
torch.set_printoptions(precision=3, sci_mode=False)




'''
----------Network----------
'''
class SNN(torch.nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        # Parameters
        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,#0.25
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : True,
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}
        blocks = []
        for dim_index in range(len(MODEL_SIZE)-2):
            blocks.append(slayer.block.cuba.Dense(neuron_params_drop, MODEL_SIZE[dim_index], MODEL_SIZE[dim_index+1]))
        blocks.append(slayer.block.cuba.Dense(neuron_params, MODEL_SIZE[-2], MODEL_SIZE[-1]))
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, spike):
        count = [torch.sum(torch.abs((spike[..., 1:]) > 0).to(torch.int64))] # Initialize the running counter & count the initial spikes
        for block in self.blocks:
            spike = block(spike)
            if PROFILING:
                count.append(torch.sum(torch.abs((spike[..., 1:]) > 0).to(torch.int64)))#.item()) # Count all the spikes in this layer & append it to the running count
        if PROFILING:
            return spike, torch.LongTensor(count).to(spike.device)#torch.FloatTensor(count).reshape((1, -1)).to(spike.device)
        else:
            return spike, 0

    def save(self, location):
        torch.save(self.state_dict(), location)

    def load(self, location):
        state_dict = torch.load(location)
        self.load_state_dict(state_dict)

    def grad_flow(self, path = "output/trained/"):
        # Monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()
        return grad

    def export_hdf5(self, filename):
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i,b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))



'''
----------ANN Network----------
'''
class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        blocks = []
        for dim_index in range(len(MODEL_SIZE)-2):
            blocks.append(torch.nn.Linear(MODEL_SIZE[dim_index], MODEL_SIZE[dim_index+1]))
            blocks.append(torch.nn.ReLU())
        blocks.append(torch.nn.Linear(MODEL_SIZE[-2], MODEL_SIZE[-1]))
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        # Now run inference
        for block in self.blocks:
            x = block(x)
        if PROFILING:
            device = torch.device("cuda")
            mac_counts = [0]
            for i in range(1, len(MODEL_SIZE)):
                mac_counts.append(MODEL_SIZE[i-1] * (MODEL_SIZE[i]))
            return x, torch.tensor(mac_counts).to(device)
        else:
            return x, 0

    def save(self, location):
        torch.save(self.state_dict(), location)

    def load(self, location):
        state_dict = torch.load(location)
        self.load_state_dict(state_dict)



'''
----------Dataset----------
'''
class WordDataset(Dataset):
    def __init__(self, pairs, embed_loc="embed/binarized/glove-64bit.txt", embed_dim=64, time_per_token=1, spike_freq=1, max_tokens=MAX_TOKENS, spike_rand=False, log_tokens=False):
        super(WordDataset, self).__init__()
        self.num_samples = len(pairs[pairs.columns[0]])
        self.events = []
        self.times = []
        self.labels = []
        self.vocab = set()
        self.num_repeats = int(np.floor(time_per_token / spike_freq))
        self.spike_count = 0# This will be used to record the average input spike count
        self.time_per_token = time_per_token
        self.spike_freq = spike_freq
        self.max_tokens = max_tokens
        self.max_timestep = (max_tokens * self.num_repeats) + self.num_repeats
        # Load the embedding
        self.embedding = embed.load(embed_loc)
        self.embed_dim = embed_dim
        self.max_in_neuron_id = embed_dim
        #self.max_out_neuron_id = len(self.embedding) + 1
        # Initialize the token log
        if log_tokens:
            with open(LOG_LOC+"token_log.txt", "w") as writefile:
                writefile.close()
        # Process everything
        column = "text"
        if (True):
            arr = []
            time_arr = []
            tok_arr = []
            for item in tqdm(pairs[column], desc=f"WordDataset: Formatting \"{column}\" column..."):
                event, time, tokens = make_spikes_from_text(item, self.embedding, self.embed_dim, self.num_repeats, self.spike_freq, self.max_tokens)# MODIFY THIS
                self.spike_count += len(event.x)
                if log_tokens:
                    with open(LOG_LOC+"token_log.txt", "a") as writefile:
                        writefile.write(item+"\n\t"+str(tokens)+"\n\n")
                        writefile.close()
                arr.append(event)
                time_arr.append(time)
                tok_arr.append(tokens)
                self.vocab.update(set(tokens) - self.vocab)
            self.events = arr
            self.times = time_arr
            #self.labels = tok_arr
        self.vocab = list(self.vocab)
        self.labels = torch.from_numpy(pairs["label"].values)

    def __getitem__(self, index):
        #print(type(self.events[0][index]), self.times[0][index])
        # Modify the below line to make it output input & target separately
        #return [col[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze() for col, time_col in zip(self.events, self.times)] + [col[index] for col in self.labels]#time_col[index]
        return (
            self.events[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze(),
            self.labels[index]
        )

    def __len__(self):
        return self.num_samples


class RealWordDataset(Dataset):
    def __init__(self, pairs, embed_loc="embed/real_valued/glove.6B.200d.txt", embed_dim=200, log_tokens=False):
        super(RealWordDataset, self).__init__()
        self.num_samples = len(pairs[pairs.columns[0]])
        self.max_timestep = MAX_TIMESTEP
        # Load the embedding
        self.embedding, self.word2index, self.index2word = load_real_embedding(embed_loc)
        self.embed_dim = embed_dim
        self.max_in_neuron_id = embed_dim
        # Initialize the token log
        if log_tokens:
            with open(LOG_LOC+"token_log.txt", "w") as writefile:
                writefile.close()
        # Process everything
        column = "text"
        if (True):
            arr = []
            for item in tqdm(pairs[column], desc=f"RealWordDataset: Formatting \"{column}\" column..."):
                vector, tokens = average_embedding(
                    item,
                    lambda x: (self.embedding[self.word2index[x]] if x in self.word2index.keys() else np.zeros(embed_dim))
                )
                if log_tokens:
                    with open(LOG_LOC+"token_log.txt", "a") as writefile:
                        writefile.write(item+"\n\t"+str(tokens)+"\n\n")
                        writefile.close()
                arr.append(vector)
            self.vectors = arr
        # If the network is an SNN, convert it to events
        if NETWORK_TYPE == "SNN":
            # If rate-coding is enabled, do that
            if RATE_CODING:
                arr = []
                for item in tqdm(self.vectors, desc=f"RealWordDataset: Rate-coding \"{column}\" column..."):
                    arr.append(rate_code(item, self.max_timestep))
                self.events = arr
        # Set the labels
        self.labels = torch.from_numpy(pairs["label"].values)
        # If we're profiling, get the average input spike count
        if PROFILING and NETWORK_TYPE == "SNN":
            self.spike_count = sum([len(event.x) for event in self.events])/self.num_samples
            #print(self.spike_count)

    def __getitem__(self, index):
        if NETWORK_TYPE == "SNN":
            return (
                self.events[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze(),
                self.labels[index]
            )
        else: # NETWORK_TYPE == "ANN":
            return (
                torch.tensor(self.vectors[index]),
                self.labels[index]
            )

    def __len__(self):
        return self.num_samples


class SentenceDataset(Dataset):
    def __init__(self, pairs, embed_loc="", embed_dim=200):
        super(RealSentenceDataset, self).__init__()
        self.num_samples = len(pairs[pairs.columns[0]])
        # Load the embedding
        self.embed = hub.load(embed_loc)
        self.embed_dim = embed_dim
        self.max_in_neuron_id = embed_dim
        # Process everything
        print("RealSentenceDataset: Formatting \"text\" column...")
        self.vectors = self.embed(pairs["text"].values).numpy().tolist()
        # Set the labels
        self.labels = torch.from_numpy(pairs["label"].values)

    def __getitem__(self, index):
        return (
            self.events[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze(),
            self.labels[index]
        )

    def __len__(self):
        return self.num_samples


class RealSentenceDataset(Dataset):
    def __init__(self, pairs, embed_loc="https://tfhub.dev/google/universal-sentence-encoder/4", embed_dim=512):
        super(RealSentenceDataset, self).__init__()
        self.num_samples = len(pairs[pairs.columns[0]])
        self.max_timestep = MAX_TIMESTEP
        # Load the embedding
        self.embed = hub.load(embed_loc)
        self.embed_dim = embed_dim
        self.max_in_neuron_id = embed_dim
        # Process everything
        column = "text"
        print(f"RealSentenceDataset: Formatting \"{column}\" column...")
        self.vectors = self.embed(pairs["text"].values).numpy()
        # If the network is an SNN, convert it to events
        if NETWORK_TYPE == "SNN":
            # If rate-coding is enabled, do that
            if RATE_CODING:
                arr = []
                for item in tqdm(self.vectors, desc=f"RealWordDataset: Rate-coding \"{column}\" column..."):
                    arr.append(rate_code(item, self.max_timestep))
                self.events = arr
        # Set the labels
        self.labels = torch.from_numpy(pairs["label"].values)
        # If we're profiling, get the average input spike count
        if PROFILING and NETWORK_TYPE == "SNN":
            self.spike_count = sum([len(event.x) for event in self.events])/self.num_samples
            #print(self.spike_count)

    def __getitem__(self, index):
        if NETWORK_TYPE == "SNN":
            return (
                self.events[index].fill_tensor(torch.zeros(1, 1, self.max_in_neuron_id, self.max_timestep)).squeeze(),
                self.labels[index]
            )
        else: # NETWORK_TYPE == "ANN":
            return (
                torch.tensor(self.vectors[index]),
                self.labels[index]
            )

    def __len__(self):
        return self.num_samples


def make_spikes_from_text(text, embedding, embed_dim, num_repeats, spike_freq, max_tokens, spike_rand=0, return_neuron_timings=False):
    spike_rand = (0 if not SPIKE_RAND else .05)
    # Tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)[:max_tokens]
    # For every token, embed it (if possible) & add it to the 'active_neurons' array
    active_tokens = []
    active_neurons_per_token = []
    counter = 0
    for token in tokens:
        embed_token = embed.embed_item(token, embedding)[0]
        if (embed_token):
            active_tokens.append(token)
            binary = embed_token
            active_neurons = []
            counter = 0
            while(binary):
                if (binary & 1):
                    active_neurons.append(counter)
                binary = binary >> 1
                counter += 1
            active_neurons_per_token.append(active_neurons)
    # Add an EOS token
    active_neurons_per_token.append(np.ones(embed_dim).tolist())
    # Now, build an event object using the active neurons
    neurons = []
    times = []
    time_counter = 0
    for active_neurons in active_neurons_per_token:
        for i in range(num_repeats):
            if not spike_rand:
                neurons.extend(active_neurons)
                times.extend([time_counter for x in range(len(active_neurons))])
            else:
                temp = random.sample(active_neurons, int(np.floor(len(active_neurons) * spike_rand)))
                neurons.extend(temp)
                times.extend([time_counter for x in range(len(temp))])
            time_counter += spike_freq
    # Return the event object
    if not return_neuron_timings:
        return slayer.io.Event(neurons, None, np.zeros(len(neurons)), times), time_counter, active_tokens
    else:
        return neurons, times


def rate_code(embed_vector, duration, spike_rand=SPIKE_RAND):
    neurons = []
    times = []
    # Constrain the vector to (0, 1)
    embed_vector = (embed_vector + 1)/2
    if spike_rand:
        for t in range(duration):
            active_neurons = [i for i in range(len(embed_vector)) if embed_vector[i] <= random.random()]
            neurons.extend(active_neurons)
            times.extend(([t] * len(active_neurons)))
    else:
        threshold = 1# This controls the relative rates of each spiking neuron
        resonators = np.zeros(len(embed_vector))
        for t in range(duration):
            resonators = resonators + embed_vector
            active_neurons = [i for i in range(len(resonators)) if resonators[i] >= threshold]
            for i in active_neurons:
                resonators[i] = resonators[i] - threshold
            neurons.extend(active_neurons)
            times.extend(([t] * len(active_neurons)))
    return slayer.io.Event(neurons, None, np.zeros(len(neurons)), times)


'''
----------Misc. Functions----------
'''
def clean_text(text):
    # All lowercase
    text = text.lower()
    # Replace underscores
    text = text.replace("_", " ")
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Remove whitespace
    text = " ".join(text.split())
    # Return the clean text
    return text


def load_real_embedding(embed_loc):
    word_to_index = {}
    index_to_word = []
    embeddings = []

    with open(embed_loc, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)

            word_to_index[word] = len(index_to_word)
            index_to_word.append(word)
            embeddings.append(vector)

    embeddings = np.stack(embeddings)

    return embeddings, word_to_index, index_to_word


def average_embedding(text, embed_func):
    # Tokenize the text
    tokens = nltk.tokenize.word_tokenize(text)
    # For every token, embed it (if possible) & add it to the running total
    active_tokens = []
    token_sum = []
    counter = 0
    for token in tokens:
        embed_token = embed_func(token)
        if embed_token.any():
            active_tokens.append(token)
            counter += 1
            if not len(token_sum):
                token_sum = embed_token
            else:
                token_sum = token_sum + embed_token
    # Return all values
    return (token_sum * (1/counter)), active_tokens


def visualize_spikes(dataset, filename):
    figure = plt.figure(figsize=(10,10))
    #figure.supxlabel = "Time (ms)"
    #figure.supylabel = "Neuron ID"
    anim = dataset.anim(figure)
    anim.save(filename, animation.PillowWriter(fps=24), dpi=300)


def load_data():
    # Get the dataset
    data = {"train":{}, "test":{}}
    if DATASET == "imdb":
        data = load_dataset("imdb")# IMDB Sentiment analysis dataset
        keys = ["train", "test"]
        cols = ["text","label"]
        df = pd.DataFrame(columns=cols)
        for col in cols:
            series = pd.Series(name=col)
            for key in keys:
                series = pd.concat([series, pd.Series(data[key][col])], ignore_index=True)
            #df = pd.concat([df, series], axis=1, ignore_index=True)
            df[col] = series
        df["text"] = df["text"].apply(clean_text)
        data = df.sample(frac=1, random_state=RANDOM_SEED)
        #data = torch.utils.data.ConcatDataset([data["train"], data["test"]])
    elif DATASET == "emotion":
        data = load_dataset("emotion")# Emotion classification dataset
        keys = ["train", "validation", "test"]
        cols = ["text","label"]
        df = pd.DataFrame(columns=cols)
        for col in cols:
            series = pd.Series(name=col)
            for key in keys:
                series = pd.concat([series, pd.Series(data[key][col])], ignore_index=True)
            #df = pd.concat([df, series], axis=1, ignore_index=True)
            df[col] = series
        df["text"] = df["text"].apply(clean_text)
        data = df.sample(frac=1, random_state=RANDOM_SEED)
        #data = torch.utils.data.ConcatDataset([data["train"], data["validation"], data["test"]])
    elif DATASET == "topic":
        data = load_dataset("")# Topic <classification> dataset
    #print(data)
    # At this point we have a shuffled dataframe with the data
    # Split the data into train, test, and valid sets
    train_data, temp_data = train_test_split(data, test_size=1-DATA_SPLITS["train"], random_state=RANDOM_SEED)
    test_data, valid_data = train_test_split(temp_data, test_size=(DATA_SPLITS["valid"]/(1-DATA_SPLITS["train"])), random_state=RANDOM_SEED)

    # Return the splits
    return train_data, valid_data, test_data



def graph_stats():
    # Read in the data
    df = pd.read_csv(LOG_LOC+LOG_NAME+".csv")
    # Normalize the loss values to between 0 and 1
    df["train_loss"] = (df["train_loss"] - df["train_loss"].min()) / (df["train_loss"].max() - df["train_loss"].min())
    df["valid_loss"] = (df["valid_loss"] - df["valid_loss"].min()) / (df["valid_loss"].max() - df["valid_loss"].min())
    # Initialize the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,10), sharex=GRAPH_VERTICALLY)
    # Plot the training loss
    ax1.plot(df["train_loss"], label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.tick_params(bottom=True, labelbottom=True)
    # Plot the validation loss
    ax2.plot(df["valid_loss"], label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.tick_params(bottom=True, labelbottom=True)
    # Plot the performance metrics
    ax3.plot(df["valid_accuracy"], label="Validation Accuracy")
    if (MODEL_SIZE[-1] > 2):
        ax3.plot(df["valid_mrr"], label="Validation MRR")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Performance")
    ax3.legend()
    ax3.set_yticks([0, .2, .4, .6, .8, 1])
    ax3.tick_params(bottom=True, labelbottom=True)
    # Save the figure
    plt.subplots_adjust(hspace=.5)
    plt.savefig(GRAPH_LOC+LOG_NAME+".jpg")
    plt.close(fig)



'''
----------Training Loop----------
'''
def train_model():
    global MODEL_SIZE
    device = torch.device("cuda")

    # Load the data
    train_data, valid_data, _ = load_data()
    # Determine which dataset/network encoding methods to use
    if DATA_FORMAT == "word":
        if NETWORK_TYPE == "SNN":
            if RATE_CODING:
                train_data = RealWordDataset(train_data, log_tokens=True)
                valid_data = RealWordDataset(valid_data)
            else:
                train_data = WordDataset(train_data, log_tokens=True)
                valid_data = WordDataset(valid_data)
            visualize_spikes(train_data.events[0], "output/data_vis/input.gif")
        else: # NETWORK_TYPE == "ANN":
            train_data = RealWordDataset(train_data, log_tokens=True)
            valid_data = RealWordDataset(valid_data)
    else: # DATA_FORMAT == "sentence":
        if NETWORK_TYPE == "SNN":
            if RATE_CODING:
                train_data = RealSentenceDataset(train_data)
                valid_data = RealSentenceDataset(valid_data)
            else:
                train_data = SentenceDataset(train_data)
                valid_data = SentenceDataset(valid_data)
            visualize_spikes(train_data.events[0], "output/data_vis/input.gif")
        else: # NETWORK_TYPE == "ANN":
            train_data = RealSentenceDataset(train_data)
            valid_data = RealSentenceDataset(valid_data)
    # Generate dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # Prepare the model
    if (MODEL_SIZE[0] == 0):
        MODEL_SIZE[0] = train_data.embed_dim
    model = (SNN() if NETWORK_TYPE == "SNN" else ANN())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=1e-5)
    error = (slayer.loss.SpikeRate(true_rate=.75, false_rate=.05) if NETWORK_TYPE == "SNN" else torch.nn.CrossEntropyLoss())
    error = error.to(device)

    # Initialize CSV file
    with open(LOG_LOC+LOG_NAME+".csv", "w") as writefile:
        writefile.write("train_loss,valid_loss,valid_accuracy,valid_mrr")
        if PROFILING:
            writefile.write(",operation count")
        writefile.close()

    # Initialize the CSV for comparative training results
    if not os.path.isfile(LOG_LOC+"train_results.csv"):
        with open(LOG_LOC+"train_results.csv", "w") as writefile:
            writefile.write("model,avg_train_loss,avg_valid_loss,valid_accuracy,valid_mrr")
            if PROFILING:
                writefile.write(",train_ops_per_sample,train_ops_per_layer")
            writefile.close()

    # Initialize terminal output
    print("\n\n\n\n\n")

    # Start training
    epoch_counts = []
    best_train_loss = 0.0
    best_accuracy = 0.0
    best_loss = 0.0
    best_mrr = 0.0
    for epoch in range(NUM_EPOCHS):
        # Move the print head back up to the top
        print("\x1b[1A\x1b[1A\x1b[1A\x1b[1A\x1b[1A", end="")
        # Record epoch (& send print head down 2 rows)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}\n\n")
        stats = []
        counts = torch.zeros(len(MODEL_SIZE)).to(torch.int64).to(device)
        # Run the training set
        running_loss = 0
        for i, (input, target) in enumerate(train_loader):
            # Move the print head up 2 rows
            print("\x1b[1A\x1b[1A", end="")
            # Record information
            print("\r" + " " * CLS_AMOUNT + "\r", end="")
            print(f"Set: Training\tSample: {i+1}/{len(train_loader)}")
            # Forward Pass
            output, count = model(input.to(device))
            # Compute Loss
            loss = error(output, target.to(device))
            # Zero out the gradients
            optimizer.zero_grad()
            # Backpropagate
            loss.backward()
            loss = loss.cpu()
            # Take an optimizer step
            optimizer.step()
            # Record the loss
            running_loss += loss.item()
            print(f"Loss: {loss:.3f}\tAverage: {running_loss/(i+1):.3f}")
            # Add the operation counts to the running total
            if PROFILING:
                counts = torch.add(counts, count)
        # Append the running loss to the stat block
        stats.append(running_loss)
        # Update the counts
        if PROFILING:
            if NETWORK_TYPE == "SNN":
                counts = counts / train_data.num_samples
            else: # NETWORK_TYPE == "ANN":
                counts = counts / len(train_loader)
            epoch_counts.append(counts.cpu().tolist())
        # Send the print head down a row
        print("\n")
        train_loss = running_loss
        running_loss = 0
        running_accuracy = 0
        running_sum_rank = 0
        # Run the vaidation set
        if (VALIDATION_SET):
            for i, (input, target) in enumerate(valid_loader):
                # Move the print head up 4 rows
                print("\x1b[1A\x1b[1A\x1b[1A\x1b[1A", end="")
                # Record information
                print("\r" + " " * CLS_AMOUNT + "\r", end="")
                print(f"Set: Validation\tSample: {i+1}/{len(valid_loader)}")
                # Forward Pass
                output, _ = model(input.to(device))
                # Compute Loss
                loss = error(output, target.to(device)).cpu()
                # Record the loss
                running_loss += loss.item()
                print(f"Loss: {loss:.3f}\tAverage: {running_loss/(i+1):.3f}")
                # Compute Accuracy & MRR
                accuracy_rec = []
                sum_rank = 0
                if NETWORK_TYPE == "SNN":
                    for sample, ground_truth in zip(output, target):
                        output_counts = []
                        counter = 0
                        for label in sample:
                            output_counts.append({"label":counter, "count":torch.sum(torch.abs((label) > 0).to(torch.int64)).cpu().item()})
                            counter += 1
                        output_counts = sorted(output_counts, key=lambda x: x["count"], reverse=True)
                        ranking = [i for i,_ in enumerate(output_counts) if _["label"] == ground_truth][0] + 1
                        sum_rank += 1/ranking
                        accuracy_rec.append(1 if ranking == 1 else 0)
                        if (BATCH_SIZE == 1):
                            print(f"Target: {target}\tAccuracy: {accuracy_rec}\tRank: {ranking}")
                            print(f"Counts: {output_counts}")
                else: # NETWORK_TYPE == "ANN":
                    output_prob = F.softmax(output, dim=1).cpu()
                    sorted_values, sorted_indices = torch.sort(output_prob, descending=True, dim=1)
                    accuracy_rec = torch.eq(sorted_indices[:,0], target).int().tolist()
                    rankings = torch.where(torch.eq(sorted_indices, target.view(-1, 1)))[1] + 1
                    sum_rank += torch.sum(1 / rankings).item()
                    if (BATCH_SIZE == 1):
                        print(f"Target: {target[0].item()}\tAccuracy: {accuracy_rec}\tRank: {rankings[0].item()}")
                        print(f"Probabilities: {[f'{prob:.3f}' for prob in output_prob[0].tolist()]}")
                accuracy = sum(accuracy_rec)/len(accuracy_rec)
                running_accuracy += sum(accuracy_rec)
                running_sum_rank += sum_rank
                # Record Accuracy & MRR
                print(f"Acc: {accuracy:.3f}\tTotal: {running_accuracy/((i+1)*BATCH_SIZE):.3f}")
                print(f"MRR: {sum_rank/BATCH_SIZE:.3f}\tTotal:{running_sum_rank/((i+1)*BATCH_SIZE):.3f}")
                if (BATCH_SIZE == 1):
                    print("\x1b[1A\x1b[1A", end="")
            # Add the running loss, accuracy, & MRR to the stat block
            stats.append(running_loss)
            stats.append(running_accuracy/(len(valid_loader)*BATCH_SIZE))
            stats.append(running_sum_rank/(len(valid_loader)*BATCH_SIZE))
            if not epoch:
                best_loss = running_loss
                best_accuracy = running_accuracy/(len(valid_loader)*BATCH_SIZE)
                best_mrr = running_sum_rank/(len(valid_loader)*BATCH_SIZE)
                best_train_loss = train_loss
            '''# If the validation accuracy sets a new record, save the model
            if (running_accuracy/(len(valid_loader)*BATCH_SIZE)) > best_accuracy:
                model.save(MODEL_LOC)
                best_accuracy = (running_accuracy/(len(valid_loader)*BATCH_SIZE))'''
            # If the validation loss sets a new record, save the model
            if running_loss <= best_loss:
                model.save(MODEL_LOC)
                best_loss = running_loss
                best_accuracy = running_accuracy/(len(valid_loader)*BATCH_SIZE)
                best_mrr = running_sum_rank/(len(valid_loader)*BATCH_SIZE)
                best_train_loss = train_loss
        # Write the stats to the CSV
        with open(LOG_LOC+LOG_NAME+".csv", "a") as writefile:
            writefile.write("\n"+",".join([str(x) for x in stats]))
            writefile.close()
        # Graph all stats
        graph_stats()
    # Move the print head down 1 row
    print("")
    # Write the epoch counts to screen
    if PROFILING:
        #print("Input:", train_data.spike_count, "spikes per epoch")
        #for i in range(len(epoch_counts)):
        #    print(f"Epoch {i+1}:", epoch_counts[i])
        print("Avg. spikes per sample:", np.mean(epoch_counts, axis=0))
    # Write the overall training stats to file
    stats = [LOG_NAME,best_train_loss/len(train_loader),best_loss/len(valid_loader),best_accuracy,best_mrr]
    if PROFILING:
        temp = np.mean(epoch_counts, axis=0)
        stats.append(np.sum(temp))
        stats.append([float(f"{number:.3f}") for number in temp])
    with open(LOG_LOC+"train_results.csv", "a") as writefile:
        writefile.write("\n"+",".join([str(x) for x in stats]))
        writefile.close()



'''
----------Testing Loop----------
'''
def test_model():
    global MODEL_SIZE
    device = torch.device("cuda")

    # Load the data
    _, _, test_data = load_data()
    # Determine which dataset/network encoding methods to use
    if DATA_FORMAT == "word":
        if NETWORK_TYPE == "SNN":
            if RATE_CODING:
                test_data = RealWordDataset(test_data, log_tokens=True)
            else:
                test_data = WordDataset(test_data, log_tokens=True)
            visualize_spikes(test_data.events[0], "output/data_vis/input.gif")
        else: # NETWORK_TYPE == "ANN":
            test_data = RealWordDataset(test_data, log_tokens=True)
    else: # DATA_FORMAT == "sentence":
        if NETWORK_TYPE == "SNN":
            if RATE_CODING:
                test_data = RealSentenceDataset(test_data)
            else:
                test_data = SentenceDataset(test_data)
            #visualize_spikes(train_data.events[0], "output/data_vis/input.gif")
        else: # NETWORK_TYPE == "ANN":
            test_data = RealSentenceDataset(test_data)
    # Generate dataloaders
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    # Prepare the model
    if (MODEL_SIZE[0] == 0):
        MODEL_SIZE[0] = test_data.embed_dim
    model = (SNN() if NETWORK_TYPE == "SNN" else ANN())
    print("Testing: Loading trained model:",LOG_NAME)
    model.load(MODEL_LOC)
    model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=1e-5)
    error = (slayer.loss.SpikeRate(true_rate=.75, false_rate=.05) if NETWORK_TYPE == "SNN" else torch.nn.CrossEntropyLoss())
    error = error.to(device)

    # Initialize CSV (if it doesn't exist)
    if not os.path.isfile(LOG_LOC+"test_results.csv"):
        with open(LOG_LOC+"test_results.csv", "w") as writefile:
            writefile.write("model,loss,accuracy,mrr")
            if PROFILING:
                writefile.write(",ops_per_sample,operation_count")
            writefile.close()

    # Initialize terminal output
    print("\n")

    # Start testing
    running_loss = 0
    running_accuracy = 0
    running_sum_rank = 0
    op_counts = torch.zeros(len(MODEL_SIZE)).to(torch.int64).to(device)
    for i, (input, target) in enumerate(test_loader):
        # Move the print head up 4 rows
        print("\x1b[1A\x1b[1A\x1b[1A\x1b[1A", end="")
        # Record information
        print("\r" + " " * CLS_AMOUNT + "\r", end="")
        print(f"Set: Testing\tSample: {i+1}/{len(test_loader)}")
        # Forward Pass
        output, count = model(input.to(device))
        # Compute Loss
        loss = error(output, target.to(device))
        # Add the operation counts
        if PROFILING:
            op_counts = torch.add(op_counts, count)
        # Record the loss
        running_loss += loss.item()
        print("\r" + " " * CLS_AMOUNT + "\r", end="")
        print(f"Loss: {loss:.3f}\tAverage: {running_loss/(i+1):.3f}")
        # Compute Accuracy & MRR
        accuracy_rec = []
        sum_rank = 0
        if NETWORK_TYPE == "SNN":
            for sample, ground_truth in zip(output, target):
                output_counts = []
                counter = 0
                for label in sample:
                    output_counts.append({"label":counter, "count":torch.sum(torch.abs((label) > 0).to(torch.int64)).cpu().item()})
                    counter += 1
                output_counts = sorted(output_counts, key=lambda x: x["count"], reverse=True)
                ranking = [i for i,_ in enumerate(output_counts) if _["label"] == ground_truth][0] + 1
                sum_rank += 1/ranking
                accuracy_rec.append(1 if ranking == 1 else 0)
                if (BATCH_SIZE == 1):
                    print(f"Target: {target}\tAccuracy: {accuracy_rec}\tRank: {ranking}")
                    print(f"Counts: {output_counts}")
        else: # NETWORK_TYPE == "ANN":
            output_prob = F.softmax(output, dim=1).cpu()
            sorted_values, sorted_indices = torch.sort(output_prob, descending=True, dim=1)
            accuracy_rec = torch.eq(sorted_indices[:,0], target).int().tolist()
            rankings = torch.where(torch.eq(sorted_indices, target.view(-1, 1)))[1] + 1
            sum_rank += torch.sum(1 / rankings).item()
            if (BATCH_SIZE == 1):
                print(f"Target: {target[0].item()}\tAccuracy: {accuracy_rec}\tRank: {rankings[0].item()}")
                print(f"Probabilities: {[f'{prob:.3f}' for prob in output_prob[0].tolist()]}")
        accuracy = sum(accuracy_rec)/len(accuracy_rec)
        running_accuracy += sum(accuracy_rec)
        running_sum_rank += sum_rank
        # Record Accuracy & MRR
        print("\r" + " " * CLS_AMOUNT + "\r", end="")
        print(f"Acc: {accuracy:.3f}\tTotal: {running_accuracy/((i+1)*BATCH_SIZE):.3f}")
        print("\r" + " " * CLS_AMOUNT + "\r", end="")
        print(f"MRR: {sum_rank/BATCH_SIZE:.3f}\tTotal:{running_sum_rank/((i+1)*BATCH_SIZE):.3f}")
        if (BATCH_SIZE == 1):
            print("\x1b[1A\x1b[1A", end="")
    stats = [LOG_NAME,running_loss]
    stats.append(running_accuracy/(len(test_loader)*BATCH_SIZE))
    stats.append(running_sum_rank/(len(test_loader)*BATCH_SIZE))
    if PROFILING:
        final_op_counts = []
        if NETWORK_TYPE == "SNN":
            final_op_counts = (op_counts.cpu()/test_data.num_samples).tolist()
        else: # NETWORK_TYPE == "ANN":
            final_op_counts = (op_counts.cpu()/len(test_loader)).tolist()
        stats.append(np.sum(final_op_counts))
        stats.append([float(f"{number:.3f}") for number in final_op_counts])
    # Write the stats to the CSV
    with open(LOG_LOC+"test_results.csv", "a") as writefile:
        writefile.write("\n"+",".join([str(x) for x in stats]))
        writefile.close()




if __name__=="__main__":
    if TRAIN_MODEL:
        train_model()
    if TEST_MODEL:
        test_model()