import numpy as np
import os.path
import pandas as pd
import random
import skopt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import confusion_matrix, fbeta_score, precision_recall_curve, auc, average_precision_score
from skopt.space import Real, Integer, Categorical
from typing import List, Tuple, Union
from torch.utils.data import DataLoader, Dataset


# EimDataset:
# - Used here, the entire input DataFrame, a list of categories, and an output DataFrame are provided
# - The input DataFrame is split into a categorical and a continuous DataFrame
# - The 'cats' DataFrame is converted to an int64 NumPy ndarray and stored on the EmbeddingDataset object as the cats attribute
# - The 'conts' DataFrame is converted to a float32 NumPy ndarray and stored on the EmbeddingDataset object as the conts attribute
# - The 'y' DataFrame is converted to a float32 NumPy array and stored on the EmbeddingDataset object as the y attribute
# - If no cats are provided, the cats attribute is an array of zeros the length of the conts list, this is similar for conts
# - When asked for an item, returns: [cats, conts, y] for that particular item
class EimDataset(Dataset):
    """
    Subclass of a PyTorch Dataset. Categorical and continuous variables are stored 
    as NumPy arrays, as is the output variable. If no categorical variables are 
    provided, an array of zeros as long as the number of continuous variables is 
    stored for each observation's categorical variables. This is similar for 
    continuous variables. If an output variable (y) is not provided, a zero is 
    stored for the output for each observation.
    
    When asked for an item a list like [categorical_vars, continuous_vars, output] 
    is returned for the item.
    """    
    cats: np.ndarray
    conts: np.ndarray
    y: np.ndarray
    
    def __init__(self, cats: np.ndarray, conts: np.ndarray, y: np.ndarray) -> 'EimDataset':
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n,1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n,1))
        self.y = np.zeros((n,1)) if y is None else y[:,None].astype(np.float32)
        
    def __len__(self) -> int: return len(self.y)

    def __getitem__(self, idx) -> List[np.ndarray]:
        return [self.cats[idx], self.conts[idx], self.y[idx]]
    
    @classmethod
    def from_data_frames(cls, df_cat: pd.DataFrame, df_cont: pd.DataFrame, y: pd.Series=None):
        cat_cols = [c.values for n,c in df_cat.items()]
        cont_cols = [c.values for n,c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame, cat_flds: List[str], cont_flds: List[str], y=None):
        return cls.from_data_frames(df[cat_flds], df[cont_flds], y)


# EimModelData exposes path, trn_dl, val_dl, and test_dl as attributes on the object. On creation, it:
# - Takes the train_input, train_y, and cats and creates an EmbeddingDataset
# - Takes the test_input, test_y, and cats and creates an EmbeddingDataset
# - Wraps these EmbeddingDatasets in DataLoaders, passing on bs to the DataLoader
# - Stories these DataLoaders and path as attributes on the created object
class EimModelData():
    ### This class provides training and validation dataloaders
    ### Which we will use in our model    
    path: Path
    trn_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader
    
    def __init__(self, path: Path, trn_ds: Dataset, val_ds: Dataset, bs: int, test_ds: Dataset=None) -> 'EimModelData': 
        self.path = path
        self.trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=1)
        self.val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True, num_workers=1)
        self.test_dl = DataLoader(test_ds, bs, shuffle=False, num_workers=1) if test_ds is not None else None
    
    @classmethod
    def from_data_frames(cls, path: Path, trn_df: pd.DataFrame, val_df: pd.DataFrame, trn_y: Union[pd.DataFrame, pd.Series], 
                         val_y: Union[pd.DataFrame, pd.Series], cat_flds: List[str], cont_flds: List[str], 
                         bs: int, test_df: pd.DataFrame=None) -> 'EimModelData':
        test_ds = EimDataset.from_data_frame(test_df, cat_flds, cont_flds) if test_df is not None else None
        return cls(path, EimDataset.from_data_frame(trn_df, cat_flds, cont_flds, trn_y),
                   EimDataset.from_data_frame(val_df, cat_flds, cont_flds, val_y), bs, test_ds=test_ds)

    @classmethod
    def from_data_frame(cls, path: Path, val_idxs: Union[List[int], np.ndarray], trn_idxs: Union[List[int], np.ndarray], 
                        df: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], cat_flds: List[str], cont_flds: List[str], 
                        bs: int, test_df: pd.DataFrame=None) -> 'EimModelData':
        val_df, val_y = df.iloc[val_idxs], y[val_idxs]
        trn_df, trn_y = df.iloc[trn_idxs], y[trn_idxs]
        return cls.from_data_frames(path, trn_df, val_df, trn_y, val_y, cat_flds, cont_flds, bs, test_df)
    

class EimModel(nn.Module):
    """
    EiM learning model that supports embedding layers for categorical features.
    """
    
    def __init__(self, emb_szs: List[Tuple[int, int]], n_cont: int, emb_drop: float, out_sz: int, szs: List[int], 
                 drops: List[float], y_range: List[float]=None, use_bn: bool=False, classify: bool=None) -> 'EimModel':
        """
        Creates a neural network model that supports embedding layers for categorical features. Dropout and batch norm 
        layers can be used following embedding layers and fully connected layers. Can be used for regression or 
        classification.
        
        Parameters
        ----------
        emb_szs : list of (int, int) tuples
            The list of embedding sizes to use. Order of tuples should correspond to the order in which
            categorical features will be passed to the model in training. The first item in the tuple 
            should be the number of unique values for the category, and the second item should be the 
            number of embedding dimensions to use for the catgory.
        n_cont : int
            The number of continuous features that will be passed to the model.
        emb_drop : float
            The dropout probability that should be used for the embedding layers.
        out_sz : int
            The number of nodes in the output later.
        szs : list of ints
            The number of nodes in each hidden, fully connected layer.
        drops : list of floats
            The dropout probability for each fully connected layer.
        y_range : list of (float, float)
            For regression, the min (y_range[0]) and max (y_range[1]) values that the output can take.
        use_bn : bool
            If true, batch norm layers will be used following the embedding and fully connected layers.
        classify : bool
            If true, model will be configured for classification.
        
        Construction:
        - Starts by creating an embedding layer (in self.embs) for each size given in emb_szs
        - Initializes the weights of each embedding layer uniformly randomly (The width of this 
          distribution is inversely proportional to the size of the embedding dimension--see 
          `emb_init`)
        - Adds up the size of all embedding dimensions
        - Adds to this count the number of continuous variables
        - Adds this total to the beginning of the szs list (in order to create the whole input layer in front of the first hidden layer)
        - Creates a list of linear layers as specified by szs--these are stored in self.lins
        - Creates a list of batch norm layers to follow each hidden layer--these are stored in self.bns
        - Performs Kaiming initialization on each batch norm layer
        - Creates a dropout layer after each hidden layer and after the embedding portion of the input layer
        - Creates a batch norm layer to follow the continuous portion of the input layer
        """
        super().__init__() ## inherit from nn.Module parent class
        self.embs = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs]) ## construct embeddings
        for emb in self.embs: emb_init(emb) ## initialize embedding weights
        n_emb = sum(e.embedding_dim for e in self.embs) ## get embedding dimension needed for 1st layer
        szs = [n_emb+n_cont] + szs ## add input layer to szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)]) ## create linear layers input, l1 -> l1, l2 ...
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]]) ## batchnormalization for hidden layers activations
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data) ## init weights with kaiming normalization
        self.outp = nn.Linear(szs[-1], out_sz) ## create linear from last hidden layer to output
        nn.init.kaiming_normal_(self.outp.weight.data) ## do kaiming initialization
        
        self.emb_drop = nn.Dropout(emb_drop) ## embedding dropout, will zero out weights of embeddings
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in [drops] * len(self.lins)])
        self.bn = nn.BatchNorm1d(n_cont) # batch norm for continous data
        self.use_bn,self.y_range = use_bn,y_range 
        self.classify = classify
        
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Calculate a forward pass for the provided input data.
        
        Parameters
        ----------
        x_cat : PyTorch Tensor
            The categorical inputs, with observations in the rows.
        x_cont PyTorch Tensor
            The categorical inputs, with observations in the rows. The ordering of the observations 
            should match the ordering of observations in `x_cat`.
        
        Forward pass:
        - Passes each column of the categorical input through its corresponding embedding layer; 
          concatenates all these results into a matrix
        - Applies dropout to output of embedding layers
        - Applies batch norm to continuous inputs
        - Concatenates output of embedding dropout and batch norm-ed continuous inputs
        - Passes this matrix through each linear/batch norm/dropout layer
        - If we are classifying, applies sigmoid activation
        - If we are regressing, applies sigmoid activation and then scales to range of output values
        """        
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)] # takes necessary emb vectors 
        x = torch.cat(x, 1) ## concatenate along axis = 1 (columns - side by side) # this is our input from cats
        x = self.emb_drop(x) ## apply dropout to elements of embedding tensor
        x2 = self.bn(x_cont) ## apply batchnorm to continous variables        
        x = torch.cat([x, x2], 1) ## concatenate cats and conts for final input
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x)) ## dotprod + non-linearity
            if self.use_bn: x = b(x) ## apply batchnorm activations
            x = d(x) ## apply dropout to activations
        x = self.outp(x) # we defined this externally just not to apply dropout to output
        if self.classify:
            x = torch.sigmoid(x) # for classification
        elif y_range:
            x = torch.sigmoid(x) ## scales the output between 0,1
            x = x*(self.y_range[1] - self.y_range[0]) ## scale output
            x = x + self.y_range[0] ## shift output
        return x


# Validation
def eim_validate(model, model_data, criterion, epochs):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = []
    for data in iter(model_data.val_dl):
        # get inputs
        x_cats, x_conts, y = data

        # wrap with variable
        x_cats = torch.LongTensor(x_cats).to(device)
        x_conts = torch.FloatTensor(x_conts).to(device)
        y = torch.FloatTensor(y).to(device)
        x_cats.requires_grad = False
        x_conts.requires_grad = False
        y.requires_grad = False
        
        outputs = model(x_cats, x_conts)
        loss = criterion(outputs, y)
        running_loss.append(loss.cpu().detach().data)    
    return np.mean(running_loss)
    

# Training:
# - Gets next batch from DataLoader
# - Breaks batch into cats, conts, and output
# - Converts these into tensors
# - Specifies that gradients for cats and conts should not be computed
# - Performs forward/backward pass
def eim_train(model, model_data, optimizer, criterion, epochs, patience=50, 
              print_output=True, save_best=False, save_path=Path('.', 'best_model.pkl')):

    model.train()    
    running_losses = {'train': [], 'validation': []}
    best_loss = float('inf')
    es_patience = patience
    es_counter = 0
    epoch_counter = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(epochs):
        epoch_counter += 1

        for data in iter(model_data.trn_dl):            

            # Get inputs
            x_cats, x_conts, y = data

            # Wrap with variables
            x_cats = torch.LongTensor(x_cats).to(device)
            x_conts = torch.FloatTensor(x_conts).to(device)
            y = torch.FloatTensor(y).to(device)
            x_cats.requires_grad = False
            x_conts.requires_grad = False
            y.requires_grad = False

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass + backward pass + optimization
            outputs = model(x_cats, x_conts)
            train_loss = criterion(outputs, y)

            running_losses['train'].append(train_loss.cpu().detach().numpy())

            train_loss.backward()
            optimizer.step()

        # Validate after each epoch
        validation_loss = eim_validate(model, model_data, criterion, epochs)
        running_losses['validation'].append(validation_loss)

        # Track best loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            loss_improved = True
            if save_best:
                torch.save(model.state_dict(), save_path)
        else:
            loss_improved = False

        # Simple early stopping
        if loss_improved:
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
#                 print("Early stopping")
                break

        # Print progress
        if print_output:
            print("Epoch: {}/{}, Train Loss: {}, Validation Loss: {}"
                  .format(epoch_counter, epochs, train_loss, validation_loss), end='\r')
            
    return running_losses


def preprocess(data: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    data.fillna('missing', inplace=True)
    for c in cats:
        data[c] = data[c].astype('category').cat.codes
    return data


def EimDataPreprocess(data, cats, inplace=True):
    ### Each categorical column should have indices as values 
    ### Which will be looked up at embedding matrix and used in modeling
    ### Make changes inplace
    if inplace:
        for c in cats:
            data[c].replace({val:i  for i, val in enumerate(data[c].unique())}, inplace=True)
        return data
    else:
        data_copy = data.copy()
        for c in cats:
            data_copy[c].replace({val:i  for i, val in enumerate(data_copy[c].unique())}, inplace=True)
        return data_copy
        
        
def get_embs_dims(data, cats, minimum=2):
    cat_sz = [len(data[c].unique()) for c in cats]
    return [(c, max(min(50, (c+1)//2 + 1), minimum)) for c in cat_sz]
    
    
def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)

    
class ProgressCallback():
    def __init__(self, n_calls):
        self.n_calls = n_calls

    def __call__(self, res):
        calls = len(res['func_vals'])
        best = np.min(res['func_vals'])
        print('Completed optimization trial {}/{}. Best loss so far: {:.6f}'.format(calls, self.n_calls, best))


class CheckpointSaver():
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        from skopt import dump
        dump(res, self.checkpoint_path, **self.dump_options)


class CustomEstimator():
    def __init__(self, cats, conts, emb_szs, embedding_dropout, layers, dropouts, y_range, use_batch_norm, 
                 lr, wd, epochs=2, optimization_n=10, random_seed=42):
        self.cats = cats
        self.conts = conts
        self.emb_szs = emb_szs
        self.embedding_dropout = embedding_dropout
        self.layers = list(layers)
        self.dropouts = dropouts
        self.y_range = y_range
        self.epochs = epochs
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.wd = wd
        self.opt_dimensions = dimensions = [
            Real(1e-8, 0.5, name='learning_rate'),
            Real(1e-5, 1e-1, name='weight_decay'),
            Categorical([(60, 60), (60, 30), (60, 30, 15), (30, 15)], name='layers'),
            Real(0., 0.75, name='dropouts'),
            Real(0., 0.75, name='embedding_dropout'),
        ]
        self.dimension_names = ['learning_rate', 'weight_decay', 'layers', 'dropouts', 'embedding_dropout']
        self.optimization_n = optimization_n
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.opt_results = None
        self.random_seed = random_seed
    
    
    def fit(self, train_input, valid_input, train_y, valid_y):
        print('Fitting CustomEstimator')
        self.model_data = EimModelData.from_data_frames('./tmp', train_input, valid_input, train_y, 
                                                        valid_y, self.cats, self.conts, bs=len(train_input))
        self._optimize()
        
        # Build model with optimal parameters
        hypes = dict(zip(self.dimension_names, self.opt_results.x))
        
        print('Final fit of CustomEstimator with tuned hyperparameters')
        sys.stdout.flush()
        
        self.model = EimModel(self.emb_szs, len(self.conts), hypes['embedding_dropout'], 1, list(hypes['layers']), 
                              hypes['dropouts'], y_range=self.y_range, classify=True, use_bn=self.use_batch_norm)
        self.model.to(self.device)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        crit = F.binary_cross_entropy
        self.model.train()        
        self.losses = eim_train(self.model, self.model_data, opt, crit, self.epochs, print_output=False, save_best=True, save_path='cv_fit_model.pkl')
        self.model.load_state_dict(torch.load(Path('.', 'cv_fit_model.pkl')))
    
    
    def score(self):
        from sklearn.metrics import precision_recall_curve
        
        self.model.eval()
        y, y_hat = self._get_test_outputs()
        precision, recall, thresholds = precision_recall_curve(y, y_hat)
        fbeta = self._get_best_fbeta(precision, recall, thresholds, y, y_hat)
        
        return fbeta
    
    
    def _objective(self, dimensions):
        _dims = dict(zip(self.dimension_names, dimensions))

        # Reset as much as possible
        torch.cuda.empty_cache()
        seed_everything(self.random_seed)

        # Build model
        _emb_model = EimModel(self.emb_szs, len(self.conts), _dims['embedding_dropout'], 1, list(_dims['layers']), _dims['dropouts'], y_range=self.y_range, 
                              classify=True, use_bn=self.use_batch_norm)
        _emb_model.to(self.device)
        _emb_model.train()

        _opt = torch.optim.SGD(_emb_model.parameters(), lr=_dims['learning_rate'], weight_decay=_dims['weight_decay'])
        _crit = F.binary_cross_entropy
        _losses = eim_train(_emb_model, self.model_data, _opt, _crit, self.epochs, patience=25, 
                            print_output=False, save_best=True, save_path=Path('.', 'cv_fit_optimize.pkl'))

        return(min(_losses['validation']))
    
    
    def _optimize(self):
        print("Optimizing CustomEstimator")
        sys.stdout.flush()
        n_calls = self.optimization_n
#         self.opt_results = skopt.gp_minimize(self._objective, self.opt_dimensions, n_calls=n_calls, random_state=42, callback=[ProgressCallback(n_calls)])
        self.opt_results = skopt.gp_minimize(self._objective, self.opt_dimensions, n_calls=n_calls, random_state=42)
    
    
    def _get_test_outputs(self):
        self.model.eval()
        val_outputs = None
        ys = None
        for data in iter(self.model_data.val_dl):
            
            # get inputs
            x_cats, x_conts, y = data

            # wrap with variable
            x_cats = torch.LongTensor(x_cats).to(self.device)
            x_conts = torch.FloatTensor(x_conts).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            x_cats.requires_grad = False
            x_conts.requires_grad = False
            y.requires_grad = False

            outputs = self.model(x_cats, x_conts).cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            if val_outputs is not None:
                val_outputs = np.concatenate((val_outputs, outputs))
                ys = np.concatenate((ys, y))
            else:
                val_outputs = outputs
                ys = y
        return (ys, val_outputs)
    
    
    def _get_best_fbeta(self, precision, recall, thresholds, y, y_hat):
        best_fbeta = 0.0
        best_fbeta_thresh = 0.0
        for thresh in thresholds:
            y_hat_thresh = threshold_array(thresh, y_hat)
            fb = fbeta_score(y, y_hat_thresh, 0.5, average='weighted')
            if fb >= best_fbeta:
                best_fbeta = fb
                best_fbeta_thresh = thresh
        return best_fbeta


class BaselineEstimator():
    def __init__(self):
        pass
    
    def fit(self, train_input, valid_input, train_y, valid_y):
        self.valid_y = np.array(valid_y)
    
    def score(self):
        y = self.valid_y
        y_hat = np.zeros(y.shape)
        
        precision, recall, thresholds = precision_recall_curve(y, y_hat)
        fbeta = self._get_best_fbeta(precision, recall, [0.5], y, y_hat)
        
        return fbeta        
    
    def _get_best_fbeta(self, precision, recall, thresholds, y, y_hat):
        best_fbeta = 0.0
        best_fbeta_thresh = 0.0
        for thresh in thresholds:
            y_hat_thresh = threshold_array(thresh, y_hat)
            fb = fbeta_score(y, y_hat_thresh, 0.5, average='weighted')
            if fb >= best_fbeta:
                best_fbeta = fb
                best_fbeta_thresh = thresh
        return best_fbeta
    
    
def paired_ttest_5x2cv(estimator1, estimator2, X_train, 
                       X_valid,
                       y_train,
                       y_valid):
    """
    Adapted from http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
    """
    
    from scipy import stats
    
    variance_sum = 0.
    first_diff = None

    def score_diff(X_1, X_2, y_1, y_2):

        estimator1.fit(X_1, X_2, y_1, y_2)
        estimator2.fit(X_1, X_2, y_1, y_2)
        est1_score = estimator1.score()
        est2_score = estimator2.score()
        score_diff = est1_score - est2_score
        return {
            'custom': est1_score,
            'baseline': est2_score,
            'score_diff': score_diff
        }
    
    scores = {
        'custom': [],
        'baseline': []
    }
    
    train_idxs, valid_idxs = [], []
    
    for i in range(5):
        
        train_idxs.append(np.random.choice(np.array(y_train.index), len(y_train.index)//2, replace=False))
        valid_idxs.append(np.random.choice(np.array(y_valid.index), len(y_valid.index)//2, replace=False))

    for i in range(5):
        
        X_1, X_2  = X_train.iloc[train_idxs[i]], X_valid.iloc[valid_idxs[i]]
        y_1, y_2  = y_train.iloc[train_idxs[i]], y_valid.iloc[valid_idxs[i]]

        print('5xCV Iteration {}'.format(i+1))
        sys.stdout.flush()

        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1['score_diff'] + score_diff_2['score_diff']) / 2.
        score_var = ((score_diff_1['score_diff'] - score_mean)**2 +
                     (score_diff_2['score_diff'] - score_mean)**2)
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1['score_diff']
        
        scores['custom'].append(score_diff_1['custom'])
        scores['custom'].append(score_diff_2['custom'])
        scores['baseline'].append(score_diff_1['baseline'])
        scores['baseline'].append(score_diff_2['baseline'])

    scores['means'] = {
       'custom': np.mean(scores['custom']),
       'baseline': np.mean(scores['baseline'])
    }
    
    numerator = first_diff
    denominator = np.sqrt(1/5. * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5)*2.
    return {
        't_stat': float(t_stat),
        'pvalue': float(pvalue),
        'scores': scores
    }


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def threshold_array(threshold, array):
    array = array.copy()
    array[array >= threshold] = 1.
    array[array < threshold] = 0.
    return array


def get_best_f05(ys, y_hats, thresholds):
    best_fbeta = 0.0
    best_fbeta_thresh = 0.0
    for thresh in thresholds:
        test_outputs_thresh = threshold_array(thresh, y_hats)
        fb = fbeta_score(ys, test_outputs_thresh, 0.5, average='weighted')
        if fb >= best_fbeta:
            best_fbeta = fb
            best_fbeta_thresh = thresh

    return best_fbeta, best_fbeta_thresh


def break_ties(xs, ys):
    zipped = list(zip(xs, ys))
    unique = np.unique(zipped, axis=0)
    unique_xs, unique_ys = unique[:, 0], unique[:, 1]
    
    new_xs, new_ys = [], []
    
    for i in range(len(unique_xs)):
        x = unique_xs[i]
        y = unique_ys[i]
        
        # Get indices of matching xs
        matching_idxs = np.argwhere(unique_xs == x)
        matching_ys = unique_ys[matching_idxs]
        
        if y == np.max(matching_ys):
            new_xs.append(x)
            new_ys.append(y)
    
    return new_xs, new_ys


def plot_pr_curve(ys, y_hats, classifier_name):
    import matplotlib.pyplot as plt
    average_precision = average_precision_score(ys, y_hats)

    precision, recall, thresholds = precision_recall_curve(ys, y_hats)
    pr_auc = auc(recall, precision)
    best_fbeta, best_fbeta_thresh = get_best_f05(ys, y_hats, thresholds)

    step_kwargs = ({'step': 'post'})
    plt.step(recall, precision, alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plot_title = """{}
    Precision-Recall Curve
    Average Precision: {:.2f}
    Best $F_{{0.5}}$: {:.2f} at $P(Reaction)={:.2f}$)"""
    plt.title(plot_title
              .format(classifier_name, average_precision, pr_auc, best_fbeta, best_fbeta_thresh));