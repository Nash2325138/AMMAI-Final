import os
import time
import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils.util import get_lr, gen_roc_plot, gen_acc_thres_plot
from evaluator import Evaluator


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, losses, metrics, optimizer, resume, config,
                 data_loader, valid_data_loaders=[], lr_scheduler=None, train_logger=None,
                 show_all_loss=False, log_step=20,
                 finetune_fc_epoch=None, finetune_first_conv_epoch=None):
        super(Trainer, self).__init__(
            model, losses, metrics, optimizer, resume,
            config, data_loader, valid_data_loaders, train_logger)
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.show_all_loss = show_all_loss
        self.evaluator = Evaluator()

    def _eval_metrics(self, data_input, model_output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data_input, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        log = {}
        if self.do_validation:
            for idx in range(len(self.valid_data_loaders)):
                valid_log = self.verify(self.valid_data_loaders[idx], epoch=epoch)
                log = {**log, **valid_log}

        np.random.seed()
        self.model.train()
        self.logger.info(f'Current lr: {get_lr(self.optimizer)}')
        epoch_start_time = time.time()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data_input in enumerate(self.data_loader):
            self.writer.set_step(self.train_iteration_count, self.data_loader.name)
            self.train_iteration_count += 1
            for key in data_input.keys():
                # Dataloader yeilds something that's not tensor, e.g data_input['video_id']
                if torch.is_tensor(data_input[key]):
                    data_input[key] = data_input[key].to(self.device)
            batch_start_time = time.time()

            self.optimizer.zero_grad()
            model_output = self.model(data_input)

            loss = self._get_loss(data_input, model_output)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('total_loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(data_input, model_output)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
                    f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
                    f'loss_total: {loss.item():.6f}, '
                    f'BT: {time.time() - batch_start_time:.2f}s'
                )
                self._write_tfboard(data_input, model_output)

        train_log = {
            'epoch_time': time.time() - epoch_start_time,
            'loss': total_loss / len(self.data_loader),
            f'{self.data_loader.name}_metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        log = {**log, **train_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _get_loss(self, data_input, model_output):
        losses = []
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight <= 0.0:
                continue
            loss = loss_instance(data_input, model_output) * loss_weight
            losses.append(loss)
            self.writer.add_scalar(f'{loss_name}', loss.item())
        loss = sum(losses)
        return loss

    def _valid_epoch(self, epoch, valid_loader_idx):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        loader = self.valid_data_loaders[valid_loader_idx]
        with torch.no_grad():
            for batch_idx, data_input in enumerate(loader):
                self.writer.set_step(self.valid_iteration_counts[valid_loader_idx], loader.name)
                self.valid_iteration_counts[valid_loader_idx] += 1
                for key in data_input.keys():
                    # Dataloader yeilds something that's not tensor, e.g data_input['video_id']
                    if torch.is_tensor(data_input[key]):
                        data_input[key] = data_input[key].to(self.device)
                model_output = self.model(data_input)

                loss = self._get_loss(data_input, model_output)

                self.writer.add_scalar('total_loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(data_input, model_output)
                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self._write_tfboard(data_input, model_output)

        return {
            f'{loader.name}_loss': total_val_loss / len(loader),
            f'{loader.name}_metrics': (total_val_metrics / len(loader)).tolist()
        }

    def _write_tfboard(self, data_input, model_output, n=4):
        face_tensor = data_input['face_tensor'][: n]
        self.writer.add_image('face_tensor', make_grid(face_tensor, nrow=4, normalize=True))

    def verify(self, data_loader, load_from=None, save_to=None, epoch=0):

        def collect():
            self.model.eval()
            self.logger.info(f'Number of examples is around {data_loader.batch_size * len(data_loader)}')
            with torch.no_grad():
                for batch_idx, data_input in enumerate(data_loader):
                    for key in data_input.keys():
                        value = data_input[key]
                        data_input[key] = value.to(self.device) if torch.is_tensor(value) else value

                    if isinstance(self.model, torch.nn.DataParallel):
                        model = self.model.module
                    source_embedding = model.embedding(data_input['f1'])
                    target_embedding = model.embedding(data_input['f2'])
                    self.evaluator.extend(source_embedding, target_embedding, data_input['is_same'])
                    if batch_idx % 50 == 0:
                        self.logger.info(f'Entry {batch_idx * data_loader.batch_size} done.')

        def draw_curves():
            self.logger.info('Drawing curves...')
            roc_curve_tensor = gen_roc_plot(fprs, tprs, return_tensor=True,
                                            save_to=os.path.join(self.checkpoint_dir, 'ROC_curve.png'))
            acc_curve_tensor = gen_acc_thres_plot(thresholds, accs, return_tensor=True,
                                                  save_to=os.path.join(self.checkpoint_dir, 'acc_curve.png'))
            self.writer.add_image('Curves', make_grid([roc_curve_tensor, acc_curve_tensor], nrow=2))

        self.writer.set_step(epoch, 'inference')
        if load_from:
            self.evaluator.load_from(load_from)
        else:
            self.evaluator.clear()
            collect()
        if save_to:
            self.evaluator.save_to(save_to)

        tprs, fprs, accs, thresholds = self.evaluator.calculate_roc(n_thres=100, strategy='cosine')
        auc = self.evaluator.calculate_auc(tprs=tprs, fprs=fprs)
        draw_curves()
        return {"valid_acc": accs.max(), "valid_auc": auc}
