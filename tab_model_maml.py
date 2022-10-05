import torch
import numpy as np
from pytorch_tabnet.utils import PredictDataset, filter_weights
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import DataLoader
import warnings

from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.utils import (
    PredictDataset,
    create_explain_matrix,
    validate_eval_set,
    create_dataloaders,
    define_device,
    ComplexEncoder,
    check_input,
    check_warm_start
)

class TaNetRegressorMAML(TabNetRegressor):
    def __post_init__(self):
        super(TaNetRegressorMAML, self).__post_init__()
    # add maml variable maml=False
    def maml_fit(
        self,
        X_train,
        y_train,

        ## Dataset for maml#########
        X_train_maml,
        y_train_maml,   
        eval_set_maml=None,  
        wk_weight_list= None,  
        ## Dataset for maml#########

        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
        from_unsupervised=None,
        warm_start=False,
        augmentations=None,

    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
            dict for custom weights per class
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        from_unsupervised: unsupervised trained model
            Use a previously self supervised model as starting weights
        warm_start: bool
            If True, current model parameters are used to start training
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last      
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")
        self.augmentations = augmentations

        if self.augmentations is not None:
            # This ensure reproducibility
            self.augmentations._set_seed()

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_input(X_train)
        check_warm_start(warm_start, from_unsupervised)

        self.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        ## For target : Validate and reformat eval set depending on target training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)


        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )


        ## For maml
        ############################################################
        maml_train_dataloader, maml_valid_dataloaders = self.self._maml_construct_loaders(
            X_train_maml, y_train_maml, eval_set_maml
        )

        
        ############################################################

        if from_unsupervised is not None:
            # Update parameters to match self pretraining
            self.__update__(**from_unsupervised.get_params())

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()
        self._update_network_params()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        if from_unsupervised is not None:
            self.load_weights_from_unsupervised(from_unsupervised)
            warnings.warn("Loading weights from unsupervised pretraining")
        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs ##########################################################################
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch_maml(maml_train_dataloader)#################


            # Apply predict epoch to all eval sets #################################
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        # compute feature importance once the best model is defined
        self.feature_importances_ = self._compute_feature_importances(X_train)

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()
        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    def _train_epoch_maml(self, train_loader):
        """
        [For maml train]

        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()

        for batch_idx, (X, y) in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            
            batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        if self.augmentations is not None:
            X, y = self.augmentations(X, y)

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs


    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()

        for batch_idx, (X, y) in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)

            batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        if self.augmentations is not None:
            X, y = self.augmentations(X, y)

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)

        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()

        # compute model output
        scores, _ = self.network(X)

        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()

        return scores
    '''
    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders
    '''

    def _maml_construct_loaders(self, X_train_maml, y_train_maml, eval_set_maml):

        """Generate dataloaders for maml train and eval set.

        Parameters
        ----------
        X_train_maml : dictionary 
            Train set.
        y_train_maml : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        train_dataloader_list : list of train_dataloaders per meta-task
        valid_dataloader_list : list of valid_dataloaders per meta-task
        """
        train_dataloader_list = []
        valid_dataloaders_list = []
        for i in range(len(X_train_maml)):
            X_train = X_train_maml[i]
            y_train = y_train_maml[i]
            eval_set = eval_set_maml[i]
            y_train_mapped = self.prepare_target(y_train)
            for i, (X, y) in enumerate(eval_set):
                y_mapped = self.prepare_target(y)
                eval_set[i] = (X, y_mapped)

            train_dataloader, valid_dataloaders = create_dataloaders(
                X_train,
                y_train_mapped,
                eval_set,
                self.updated_weights,
                self.batch_size,
                self.num_workers,
                self.drop_last,
                self.pin_memory,
            )
            train_dataloader_list.append(train_dataloader)
            valid_dataloaders_list.append(valid_dataloaders)
        return train_dataloader_list, valid_dataloaders_list


    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders
    
    # def fit(
    #     self,
    #     X_train,
    #     y_train,
    #     eval_set=None,
    #     eval_name=None,
    #     eval_metric=None,
    #     loss_fn=None,
    #     weights=0,
    #     max_epochs=100,
    #     patience=10,
    #     batch_size=1024,
    #     virtual_batch_size=128,
    #     num_workers=0,
    #     drop_last=True,
    #     callbacks=None,
    #     pin_memory=True,
    #     from_unsupervised=None,
    #     warm_start=False,
    #     augmentations=None,

    # ):
    #     """Train a neural network stored in self.network
    #     Using train_dataloader for training data and
    #     valid_dataloader for validation.

    #     Parameters
    #     ----------
    #     X_train : np.ndarray
    #         Train set
    #     y_train : np.array
    #         Train targets
    #     eval_set : list of tuple
    #         List of eval tuple set (X, y).
    #         The last one is used for early stopping
    #     eval_name : list of str
    #         List of eval set names.
    #     eval_metric : list of str
    #         List of evaluation metrics.
    #         The last metric is used for early stopping.
    #     loss_fn : callable or None
    #         a PyTorch loss function
    #     weights : bool or dictionnary
    #         0 for no balancing
    #         1 for automated balancing
    #         dict for custom weights per class
    #     max_epochs : int
    #         Maximum number of epochs during training
    #     patience : int
    #         Number of consecutive non improving epoch before early stopping
    #     batch_size : int
    #         Training batch size
    #     virtual_batch_size : int
    #         Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
    #     num_workers : int
    #         Number of workers used in torch.utils.data.DataLoader
    #     drop_last : bool
    #         Whether to drop last batch during training
    #     callbacks : list of callback function
    #         List of custom callbacks
    #     pin_memory: bool
    #         Whether to set pin_memory to True or False during training
    #     from_unsupervised: unsupervised trained model
    #         Use a previously self supervised model as starting weights
    #     warm_start: bool
    #         If True, current model parameters are used to start training
    #     """
    #     # update model name

    #     self.max_epochs = max_epochs
    #     self.patience = patience
    #     self.batch_size = batch_size
    #     self.virtual_batch_size = virtual_batch_size
    #     self.num_workers = num_workers
    #     self.drop_last = drop_last      
    #     # self.input_dim = X_train.shape[1]

    #     # If maml=True self.input_dim is X_train[0].sape[1] ( X_train is list of workload X_train )
    #     if self.maml==True:
    #         self.input_dim = X_train[0].shape[1]    # ( X_train is list of workload X_train )
    #     else:
    #         self.input_dim = X_train.shape[1]

    #     self._stop_training = False
    #     self.pin_memory = pin_memory and (self.device.type != "cpu")
    #     self.augmentations = augmentations

    #     if self.augmentations is not None:
    #         # This ensure reproducibility
    #         self.augmentations._set_seed()

    #     eval_set = eval_set if eval_set else []

    #     if loss_fn is None:
    #         self.loss_fn = self._default_loss
    #     else:
    #         self.loss_fn = loss_fn

    #     check_input(X_train)
    #     check_warm_start(warm_start, from_unsupervised)

    #     self.update_fit_params(
    #         X_train,
    #         y_train,
    #         eval_set,
    #         weights,
    #     )

    #     # Validate and reformat eval set depending on training data
    #     eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)



    #     train_dataloader, valid_dataloaders = self._construct_loaders(
    #         X_train, y_train, eval_set
    #     )

    #     if from_unsupervised is not None:
    #         # Update parameters to match self pretraining
    #         self.__update__(**from_unsupervised.get_params())

    #     if not hasattr(self, "network") or not warm_start:
    #         # model has never been fitted before of warm_start is False
    #         self._set_network()
    #     self._update_network_params()
    #     self._set_metrics(eval_metric, eval_names)
    #     self._set_optimizer()
    #     self._set_callbacks(callbacks)

    #     if from_unsupervised is not None:
    #         self.load_weights_from_unsupervised(from_unsupervised)
    #         warnings.warn("Loading weights from unsupervised pretraining")
    #     # Call method on_train_begin for all callbacks
    #     self._callback_container.on_train_begin()

    #     # Training loop over epochs
    #     for epoch_idx in range(self.max_epochs):

    #         # Call method on_epoch_begin for all callbacks
    #         self._callback_container.on_epoch_begin(epoch_idx)

    #         # self._train_epoch(train_dataloader)
    #         # If maml=True use self._train_epoch_maml
    #         if self.maml==True:
    #             self._train_epoch_maml(train_dataloader)
    #         else:
    #             self._train_epoch(train_dataloader)


    #         # Apply predict epoch to all eval sets
    #         for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
    #             self._predict_epoch(eval_name, valid_dataloader)

    #         # Call method on_epoch_end for all callbacks
    #         self._callback_container.on_epoch_end(
    #             epoch_idx, logs=self.history.epoch_metrics
    #         )

    #         if self._stop_training:
    #             break

    #     # Call method on_train_end for all callbacks
    #     self._callback_container.on_train_end()
    #     self.network.eval()

    #     # compute feature importance once the best model is defined
    #     self.feature_importances_ = self._compute_feature_importances(X_train)

    # def predict(self, X):
    #     """
    #     Make predictions on a batch (valid)

    #     Parameters
    #     ----------
    #     X : a :tensor: `torch.Tensor`
    #         Input data

    #     Returns
    #     -------
    #     predictions : np.array
    #         Predictions of the regression problem
    #     """
    #     self.network.eval()
    #     dataloader = DataLoader(
    #         PredictDataset(X),
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #     )

    #     results = []
    #     for batch_nb, data in enumerate(dataloader):
    #         data = data.to(self.device).float()
    #         output, M_loss = self.network(data)
    #         predictions = output.cpu().detach().numpy()
    #         results.append(predictions)
    #     res = np.vstack(results)
    #     return self.predict_func(res)

    # def _train_epoch_maml(self, train_loader):
    #     """
    #     [For maml train]

    #     Trains one epoch of the network in self.network

    #     Parameters
    #     ----------
    #     train_loader : a :class: `torch.utils.data.Dataloader`
    #         DataLoader with train set
    #     """
    #     self.network.train()

    #     for batch_idx, (X, y) in enumerate(train_loader):
    #         self._callback_container.on_batch_begin(batch_idx)
            
    #         batch_logs = self._train_batch(X, y)

    #         self._callback_container.on_batch_end(batch_idx, batch_logs)

    #     epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
    #     self.history.epoch_metrics.update(epoch_logs)

    #     return


    # def _train_epoch(self, train_loader):
    #     """
    #     Trains one epoch of the network in self.network

    #     Parameters
    #     ----------
    #     train_loader : a :class: `torch.utils.data.Dataloader`
    #         DataLoader with train set
    #     """
    #     self.network.train()

    #     for batch_idx, (X, y) in enumerate(train_loader):
    #         self._callback_container.on_batch_begin(batch_idx)

    #         batch_logs = self._train_batch(X, y)

    #         self._callback_container.on_batch_end(batch_idx, batch_logs)

    #     epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
    #     self.history.epoch_metrics.update(epoch_logs)

    #     return

    # def _train_batch(self, X, y):
    #     """
    #     Trains one batch of data

    #     Parameters
    #     ----------
    #     X : torch.Tensor
    #         Train matrix
    #     y : torch.Tensor
    #         Target matrix

    #     Returns
    #     -------
    #     batch_outs : dict
    #         Dictionnary with "y": target and "score": prediction scores.
    #     batch_logs : dict
    #         Dictionnary with "batch_size" and "loss".
    #     """
    #     batch_logs = {"batch_size": X.shape[0]}

    #     X = X.to(self.device).float()
    #     y = y.to(self.device).float()

    #     if self.augmentations is not None:
    #         X, y = self.augmentations(X, y)

    #     for param in self.network.parameters():
    #         param.grad = None

    #     output, M_loss = self.network(X)

    #     loss = self.compute_loss(output, y)
    #     # Add the overall sparsity loss
    #     loss = loss - self.lambda_sparse * M_loss

    #     # Perform backward pass and optimization
    #     loss.backward()
    #     if self.clip_value:
    #         clip_grad_norm_(self.network.parameters(), self.clip_value)
    #     self._optimizer.step()

    #     batch_logs["loss"] = loss.cpu().detach().numpy().item()

    #     return batch_logs

    # def _predict_epoch(self, name, loader):
    #     """
    #     Predict an epoch and update metrics.

    #     Parameters
    #     ----------
    #     name : str
    #         Name of the validation set
    #     loader : torch.utils.data.Dataloader
    #             DataLoader with validation set
    #     """
    #     # Setting network on evaluation mode
    #     self.network.eval()

    #     list_y_true = []
    #     list_y_score = []

    #     # Main loop
    #     for batch_idx, (X, y) in enumerate(loader):
    #         scores = self._predict_batch(X)
    #         list_y_true.append(y)
    #         list_y_score.append(scores)

    #     y_true, scores = self.stack_batches(list_y_true, list_y_score)

    #     metrics_logs = self._metric_container_dict[name](y_true, scores)
    #     self.network.train()
    #     self.history.epoch_metrics.update(metrics_logs)
    #     return

    # def _predict_batch(self, X):
    #     """
    #     Predict one batch of data.

    #     Parameters
    #     ----------
    #     X : torch.Tensor
    #         Owned products

    #     Returns
    #     -------
    #     np.array
    #         model scores
    #     """
    #     X = X.to(self.device).float()

    #     # compute model output
    #     scores, _ = self.network(X)

    #     if isinstance(scores, list):
    #         scores = [x.cpu().detach().numpy() for x in scores]
    #     else:
    #         scores = scores.cpu().detach().numpy()

    #     return scores
    # '''
    # def _construct_loaders(self, X_train, y_train, eval_set):
    #     """Generate dataloaders for train and eval set.

    #     Parameters
    #     ----------
    #     X_train : np.array
    #         Train set.
    #     y_train : np.array
    #         Train targets.
    #     eval_set : list of tuple
    #         List of eval tuple set (X, y).

    #     Returns
    #     -------
    #     train_dataloader : `torch.utils.data.Dataloader`
    #         Training dataloader.
    #     valid_dataloaders : list of `torch.utils.data.Dataloader`
    #         List of validation dataloaders.

    #     """
    #     # all weights are not allowed for this type of model
    #     y_train_mapped = self.prepare_target(y_train)
    #     for i, (X, y) in enumerate(eval_set):
    #         y_mapped = self.prepare_target(y)
    #         eval_set[i] = (X, y_mapped)

    #     train_dataloader, valid_dataloaders = create_dataloaders(
    #         X_train,
    #         y_train_mapped,
    #         eval_set,
    #         self.updated_weights,
    #         self.batch_size,
    #         self.num_workers,
    #         self.drop_last,
    #         self.pin_memory,
    #     )
    #     return train_dataloader, valid_dataloaders
    # '''

    # def _construct_loaders(self, X_train, y_train, eval_set):
        # """Generate dataloaders for train and eval set.

        # Parameters
        # ----------
        # X_train : np.array
        #     Train set.
        # y_train : np.array
        #     Train targets.
        # eval_set : list of tuple
        #     List of eval tuple set (X, y).

        # Returns
        # -------
        # train_dataloader : `torch.utils.data.Dataloader`
        #     Training dataloader.
        # valid_dataloaders : list of `torch.utils.data.Dataloader`
        #     List of validation dataloaders.

        # """
        # # all weights are not allowed for this type of model
        # y_train_mapped = self.prepare_target(y_train)
        # for i, (X, y) in enumerate(eval_set):
        #     y_mapped = self.prepare_target(y)
        #     eval_set[i] = (X, y_mapped)

        # train_dataloader, valid_dataloaders = create_dataloaders(
        #     X_train,
        #     y_train_mapped,
        #     eval_set,
        #     self.updated_weights,
        #     self.batch_size,
        #     self.num_workers,
        #     self.drop_last,
        #     self.pin_memory,
        # )
        # return train_dataloader, valid_dataloaders
