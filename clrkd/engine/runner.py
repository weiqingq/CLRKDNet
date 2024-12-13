import time
import cv2
import torch
from tqdm import tqdm
import numpy as np
import random
import os

from clrkd.models.registry import build_net
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrkd.datasets import build_dataloader
from clrkd.utils.recorder import build_recorder
from clrkd.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


import torch.nn as nn
def at_loss(x, y):
    """
    Calculate the mean squared error between two sets of attention maps, averaged over the batch.

    :param x: Attention maps from the student model [batch_size, 1, height, width].
    :param y: Attention maps from the teacher model [batch_size, 1, height, width].
    :return: Mean squared error averaged over the batch.
    """
    # Calculate squared differences and then mean over all dimensions except the batch dimension
    mse_loss = nn.MSELoss(reduction='mean')
    return mse_loss(x,y)

def distill_attention_loss(student_attention_maps, teacher_attention_maps):
    """
    Calculate distillation loss based on attention maps from student and teacher models,
    considering batch-wise processing.

    :param student_attention_maps: List of attention maps from the student model.
    :param teacher_attention_maps: List of attention maps from the teacher model.
    :return: Total loss calculated from the difference between student and teacher attention maps,
             averaged over the batch.
    """
    # Calculate attention map loss for each corresponding layer and average over all maps
    at_losses = [at_loss(s_map, t_map) for s_map, t_map in zip(student_attention_maps, teacher_attention_maps)]
    total_at_loss = sum(at_losses) / len(at_losses)  # Average the losses from all layers

    return total_at_loss


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

        if self.cfg.distillation:
            print(self.cfg.teacher_model_path)
            self.recorder.logger.info('Loading Teacher model...')
            self.recorder.logger.info('Teacher weight Path: ' + str(self.cfg.teacher_model_path))

            self.teacher = build_net(self.cfg.teacher_model_cfg)
            self.teacher = MMDataParallel(self.teacher, device_ids=range(self.cfg.gpus)).cuda()
            load_network(self.teacher, self.cfg.teacher_model_path)
            self.teacher.eval()  # Set the teacher to evaluation mode

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            if self.cfg.distillation:
                output, s_attention, s_logits, s_priors = self.net(data, True)
            else:
                output= self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum() 

            if self.cfg.distillation:
                with torch.no_grad():
                    output_t, t_attention, t_logits, t_priors = self.teacher(data, True)

                mse_loss_fn = nn.MSELoss(reduction='mean')
                mse_sum_loss = nn.MSELoss(reduction='sum')  # Use sum reduction
                distill_att_loss = distill_attention_loss(s_attention, t_attention)

                distill_log_loss, distill_priors_loss = 0, 0
                for s_logit, t_logit in zip(s_logits, t_logits):
                    distill_log_loss += mse_loss_fn(s_logit, t_logit)  # Mean loss for each pair
                for s_prior, t_prior in zip(s_priors, t_priors):
                    distill_priors_loss += mse_sum_loss(s_prior, t_prior)/len(s_priors[0])  # Sum loss for each pair

                # distill_log_loss = mse_loss_fn(s_logits, t_logits)
                # distill_priors_loss = mse_sum_loss(s_priors, t_priors)/len(s_priors)

                att_loss_weight, log_loss_weight = self.cfg.att_loss_weight, self.cfg.log_loss_weight
                priors_loss_weight = self.cfg.priors_loss_weight

                loss_att = distill_att_loss * att_loss_weight
                loss += loss_att
                loss_log = distill_log_loss * log_loss_weight
                loss += loss_log
                loss_prior = distill_priors_loss * priors_loss_weight
                loss += loss_prior

                output['loss_stats']['distill_att_loss'] = loss_att
                output['loss_stats']['distill_log_loss'] = loss_log
                output['loss_stats']['distill_priors_loss'] = loss_prior

                output['loss_stats']['loss'] = loss

            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

        if not metric:
            return
        if metric > self.metric:
            self.metric = metric
            self.best_epoch = self.recorder.epoch
            self.save_ckpt(is_best=True)
        self.recorder.logger.info('Best metric: ' + str(self.metric) + '  epoch_best:' + str(self.best_epoch))


    def inference(self, image):
        self.net.eval()
        with torch.no_grad():
            output= self.net(image)
            output = self.net.module.heads.get_lanes(output)
        return output


    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)