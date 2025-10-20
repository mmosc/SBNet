
# Neural Network for Cross-Modal Music Similarity Classification
This is the code for the multimedia mining search and retrieval (MMSR) WS25 course at the Johannes Kepler University Linz. 

The code aims at developing and training a neural network (NN) architecture that, given two tracks, classifies them as 
"sharing at least one genre" (1, positive output) or "not sharing any music genre" (0, negative output). 
The task is therefore formulated as binary classification.

You will use this code to train the algorithm on a set $S_\text{class}$ of 1000 tracks. The pre-trained architecture will then be used as feature
extractor to compute representations of the set $S_\text{retr}$ of 4000 tracks to be used in the actual MMRS task, i.e., the retrieval of similar tracks, 
given a query track. Notice that $S_\text{class}$ and $S_\text{retr}$ do not share any track, i.e., their intersection is empty. 

## Cross-Modal Music Similarity Classification
### Dataset
The dataset creation is shown in the notebook `/data/create_dataset.ipynb`. As you can see, we started from an initial set of tracks $S = S_\text{class} \cup S_\text{retr}$. 
We randomly sampled 1000 tracks to form $S_\text{class}$. We created all possible combinations of two tracks out of 
$S_\text{class}$, resulting in $499,500$ pairs. Each pair is labelled either with a 1, if the two tracks share at least one genre
or with a 0, if the two tracks do not share any genre.

We then shuffled the resulting data, and selected 80% of pairs as training set, 10% as validation set, and 10% as test set.
The data for training, evaluating, and testing the NN is shared with you in the folder `binary_classification` available [here](https://cloud.cp.jku.at/index.php/s/zaoS2nLtDkNfHBx). 

Notice that the files to be used for the actual retrieval task (i.e., the set $S_\text{retr}$) are shared with you in the folder `retrieval` available [here](https://cloud.cp.jku.at/index.php/s/zaoS2nLtDkNfHBx). 


### Models
The code for the actual NN models is in `ssnet_fop/binary_classification_model.py`. All models rely on the `SingleBranchGeneral` class, 
which is a subclass of PyTorch `nn.Module`. All models have a `forward` method that, given the feature vectors of 
track $i$ (feature 1) and track $j$ (feature 2)
 - Processes each of them with an embedding module (see below for description)
 - Computes the cosine similarity of the resulting embeddings
 - Returns the cosine similarity as logits to compute the probability that the two tracks share at least one genre.

The feature embedding modules are multi-layer perceptrons (MLP) with fully connected linear layers, followed by batch normalization, 
ReLU activation function and dropout regularization. The all embedding modules are children classes of `EmbedBranchGeneral`. 
In all embedding modules, the same final layer `self.fc_shared` is shared between the two input modalities. 
We currently support three options:
- `EmbedBranchDownproject`: Before being passed as input to `self.fc_shared`, feature 1 and feature 2 are passed to a modality-specific layer, `self.fc_i` and `self.fc_j`. 
 This means that the first layer is not shared between the two modalities.
- `EmbedBranchPadding`: Before being passed as input to `self.fc_shared`, feature 1 and feature 2 are passed to a same layer, shared between modalities. 
Since the modalities might be of different dimensionality, the algorithm pads with zeros the modality of lower dimension, to ensure
That both modalities are compatible with the input dimension of `self.fc_shared`. In this case, one between `self.fc_i` and `self.fc_j` is the 
identity operator, while the other is the padding operator.


### Training 
Models are trained with binary cross entropy loss. The code for training the NN is in `ssnet_fop/main.py`. You can run it as follows:
```bash
cd ssnet_fop
python main.py
```

All arguments of this script are optional, but be aware of the default values.  
```bash
usage: main.py [-h] [--seed S] [--cuda] [--save_dir SAVE_DIR] [--lr LR] [--batch_size BATCH_SIZE] [--max_num_epoch MAX_NUM_EPOCH] [--intermediate_emb INTERMEDIATE_EMB] [--dim_embed DIM_EMBED] [--feature_i FEATURE_I] [--feature_j FEATURE_J]
               [--merging_technique MERGING_TECHNIQUE]

options:
  --help, --help            show this help message and exit
  --seed S              Random Seed. Default 1
  --device                Device to be used, either cuda (if a GPU is available) or cpu. Default cuda
  --save_dir SAVE_DIR   Directory for saving checkpoints. Default model
  --lr LR               learning rate. Default: 1e-5
  --batch_size BATCH_SIZE
                        Batch size for training. Default 128
  --max_num_epoch MAX_NUM_EPOCH
                        Max number of epochs to train, number. Default 500
  --intermediate_emb INTERMEDIATE_EMB
                        Intermediate Layer. Default 256
  --dim_embed DIM_EMBED
                        Embedding Size. Default 128
  --feature_i FEATURE_I
                        feature i (first modality). Default mfcc_bow
  --feature_j FEATURE_J
                        feature j (second modality). Default mfcc_bow
  --merging_technique MERGING_TECHNIQUE
                        whether to downproject or pad if there is a dimension mismatch. Default downproject

```
### Evaluation



## Project Structure and Files Description
```bash
.
├── README.md
├── data
│ ├── todo folder
│ ├── todo folder
│ ├── create_dataset.ipynb
│ └── inspect.ipynb
├── ssnet_fop
│ ├── best_fc2_mfcc_bow_mfcc_bow_model
│ │ └── checkpoint.pth.tar
│ ├── fc2_mfcc_bow_mfcc_bow_model
│ ├── main.py
│ ├── online_evaluation.py
│ ├── output
│ ├── retrieval_model.py
```
### Dataset Folder
For training, evaluating, and testing the model with our scripts, the dataset should be stored in the `data` folder. 
The data for training, evaluating, and testing the NN is shared with you in the folder `binary_classification` available [here](https://cloud.cp.jku.at/index.php/s/zaoS2nLtDkNfHBx). 


### Trained Model Folder
The pre-trained model should be stored in the `ssnet_fop/best_fc2<modality_1>_<modality_2>_model` folder. Such a model is 
also stored any time you re-run the code to train a new model. Pre-trained model instances are available (not yet ready) [here]().
The name shares the same convention of the model folder, and allows you to identify what features were used during training.
## Setup
### Environment

We recommend using a conda environment to run the code. Once you installed conda, you can run the following commands to reproduce our setup
```bash
conda env -n sbnet python=3.12
conda activate sbnet
pip install pandas==2.3.3 scikit-learn==1.7.2 scipy==1.16.2 tqdm==4.67.1 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 
```
