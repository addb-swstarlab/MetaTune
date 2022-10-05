import torch
import numpy as np
from pytorch_tabnet.utils import PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import DataLoader
import warnings


class TaNetRegressorMAML(TabNetRegressor):
    def __post_init__(self):
        return super(TaNetRegressorMAML).__post_init__()
        
    def fit(
        self,
        X_train,
        y_train,
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

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, y_train, eval_set
        )

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

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
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

# class TabNetRegressor(TabModel):
#     def __post_init__(self):
#         super(TabNetRegressor, self).__post_init__()
#         self._task = 'regression'
#         self._default_loss = torch.nn.functional.mse_loss
#         self._default_metric = 'mse'

#     def prepare_target(self, y):
#         return y

#     def compute_loss(self, y_pred, y_true):
#         return self.loss_fn(y_pred, y_true)

#     def update_fit_params(
#         self,
#         X_train,
#         y_train,
#         eval_set,
#         weights
#     ):
#         if len(y_train.shape) != 2:
#             msg = "Targets should be 2D : (n_samples, n_regression) " + \
#                   f"but y_train.shape={y_train.shape} given.\n" + \
#                   "Use reshape(-1, 1) for single regression."
#             raise ValueError(msg)
#         self.output_dim = y_train.shape[1]
#         self.preds_mapper = None

#         self.updated_weights = weights
#         filter_weights(self.updated_weights)

#     def predict_func(self, outputs):
#         return outputs

#     def stack_batches(self, list_y_true, list_y_score):
#         y_true = np.vstack(list_y_true)
#         y_score = np.vstack(list_y_score)
#         return y_true, y_score