from logging import getLogger
from pathlib import Path
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseRunner
from .hooks import build_hooks
from .models import build_model
from .optim import build_optimizer, build_scheduler
from ..config import Config
from ..data_bundle.base import BaseDataBundle

logger = getLogger('__main__')


class TorchRunner(BaseRunner):
    """Runner class of PyTorch implementation.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.device = torch.device(config.device)
        self.hooks = build_hooks(config)
        self.initialize()

    def initialize(self):
        self.fold_scores = []
        self.model_states = {}
        self.optimizer_states = {}

    def run(self, data_bundle: BaseDataBundle):
        self.initialize()
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {len(data_bundle)}')
        logger.info(f'  Num Epochs = {self.config.n_epochs}')
        logger.info(f'  Batch size = {self.config.train_data_loader.batch_size}')
        logger.info(f'  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}')

        for fold, (train_loader, valid_loader) in enumerate(data_bundle.generate_folds_data()):
            logger.info(f'\n============================ fold {fold + 1} ============================')
            logger.info('|         |------- train -------|------- valid -------|        |')
            logger.info('|  epoch  |   loss   |  metric  |   loss   |  metric  |  time  |')
            logger.info('|--------------------------------------------------------------|')
            self.fold_initialize()
            self.train_single_fold(fold, train_loader, valid_loader)
            self.save_states(fold)

    def fold_initialize(self):
        self.model = build_model(self.config)
        self.model.zero_grad()
        self.model.to(self.device)

        self.optimizer = build_optimizer(self.config, self.model)
        self.epoch_scheduler, self.batch_scheduler = build_scheduler(self.config, self.optimizer)

    def train_single_fold(self, fold: int, train_loader: DataLoader, valid_loader: DataLoader):
        for epoch in range(self.config.n_epochs):
            epoch_start_time = time.time()
            self.train_single_epoch(epoch, train_loader)
            self.epoch_end_fn()

            train_loss, train_score = self.evaluate(train_loader)
            valid_loss, valid_score = self.evaluate(valid_loader)
            self.fold_scores.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_score': train_score,
                'valid_score': valid_score
            })
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(
                f'| {epoch:<3.0f}     '
                f'| {train_loss:<1.6f} '
                f'| {train_score:<1.6f} '
                f'| {valid_loss:<1.6f} '
                f'| {valid_score:<1.6f} '
                f'| {epoch_elapsed_time:<.2f}   |'
            )

    def epoch_end_fn(self):
        if self.epoch_scheduler is not None:
            self.epoch_scheduler.step()

    def train_single_epoch(self, epoch: int, train_loader: DataLoader):
        self.model.train()
        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f'epoch {epoch + 1}',
                            leave=False)
        for i, (x_batch, y_batch) in progress_bar:
            x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
            y_pred = self.model(**x_batch)
            loss = self.hooks.loss_fn(y_pred, y_batch.to(self.device))
            loss.backward()
            progress_bar.set_postfix(loss=loss.item())

            if (
                self.config.gradient_accumulation_steps is None
                or (i + 1) % self.config.gradient_accumulation_steps == 0
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.batch_end_fn()

    def batch_end_fn(self):
        if self.batch_scheduler is not None:
            self.batch_scheduler.step()

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            preds = []
            for x_batch, y_batch in data_loader:
                x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
                outputs = self.model(**x_batch)
                loss = self.hooks.loss_fn(outputs, y_batch.to(self.device))
                total_loss += loss.item()
                pred = self.hooks.post_forward_fn(outputs).detach().cpu()
                preds.append(pred)
        total_loss /= len(data_loader)
    
        y_true = data_loader.dataset.target
        y_pred = torch.cat(preds, dim=0).numpy()
        score = self.hooks.metric_fn(y_true, y_pred)
        return total_loss, score

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preds = []
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch[0].items()}
                outputs = self.model(**batch)
                pred = self.hooks.post_forward_fn(outputs).detach().cpu()
                preds.append(pred)
        return torch.cat(preds, dim=0).numpy()

    def save_states(self, fold: int):
        self.model_states[f'model{fold}'] = self.model.state_dict()
        self.optimizer_states[f'optimizer{fold}'] = self.optimizer.state_dict()

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        torch.save(self.model_states, output_dir / 'model.pt')
        torch.save(self.optimizer_states, output_dir / 'optimizer.pt')
