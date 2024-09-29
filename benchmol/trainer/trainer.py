import os.path
import sys

import numpy as np
import torch
from tqdm import tqdm

from metrics.metric import comprehensive_evaluate
from model_pools.base_utils import save_checkpoint
from utils.public_utils import get_tqdm_desc, is_left_better_right


class Trainer:
    def __init__(self, model, data_type, train_loader, valid_loader, test_loader, task_type, criterion, optimizer, label_empty=None, device="cpu", logger=None):
        assert task_type in ["classification", "regression"]
        assert data_type in ["feature", "sequence", "graph", "geometry", "image"]
        self.model = model
        self.data_type = data_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.task_type = task_type
        self.criterion = criterion
        self.optimizer = optimizer
        self.label_empty = label_empty
        self.device = device
        self.log = print if logger is None else logger.info

        self.model.to(device)

    def get_image_logits(self, images):
        if len(images.shape) == 4:  # (batch, 3, 224, 224)
            bs, c, h, w = images.shape
            y_logit = self.model(images)
        elif len(images.shape) == 5:  # (batch, n_view, 3, 224, 224)
            bs, n_view, c, h, w = images.shape
            images = images.reshape(bs * n_view, c, h, w)
            y_logit = self.model(images).reshape(bs, n_view, -1).mean(1)
        elif len(images.shape) == 6:  # (batch, n_conf, n_view, 3, 224, 224)
            bs, n_conf, n_view, c, h, w = images.shape
            images = images.reshape(bs * n_conf * n_view, c, h, w).to(self.device)
            y_logit = self.model(images).reshape(bs, n_conf, n_view, -1).mean(2).feat.mean(1)
        else:
            raise NotImplementedError
        return y_logit

    def train_epoch(self, tqdm_desc):
        self.model.train()
        accu_loss = torch.zeros(1).to(self.device)
        self.optimizer.zero_grad()

        data_loader = tqdm(self.train_loader, total=len(self.train_loader))
        step = 0
        for step, data_package in enumerate(data_loader):
            if self.data_type == "image":
                images, labels = data_package["images"].to(self.device), data_package["labels"].to(self.device)
                y_logit = self.get_image_logits(images)
            elif self.data_type == "graph":
                data_package = data_package.to(self.device)
                labels = data_package.y
                y_logit = self.model(data_package)
            elif self.data_type == "geometry":  # {"coords": coords, "edge_attr": edge_attr, "edge_index": edge_index}
                data_package = data_package.to(self.device)
                labels = data_package.y
                y_logit = self.model(data_package)
            elif self.data_type == "sequence":
                data_package = {item: data_package[item].to(self.device) for item in data_package.keys() if
                                isinstance(data_package[item], torch.Tensor)}
                labels = data_package["label"]
                y_logit = self.model(data_package)
            elif self.data_type == "feature":
                feats, labels = data_package["feats"].to(self.device), data_package["labels"].to(self.device).double()
                y_logit = self.model(feats)
            else:
                raise NotImplementedError

            if self.label_empty is not None:
                is_valid = labels != self.label_empty
                loss_mat = self.criterion(y_logit.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                loss = self.criterion(y_logit, labels.view(y_logit.shape).float()).mean()

            loss.backward()

            # # logger
            accu_loss += loss.detach()
            data_loader.desc = "{} total loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

            if not torch.isfinite(loss):
                self.log('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return accu_loss.item() / (step + 1)

    @torch.no_grad()
    def evaluate_epoch(self, type, tqdm_desc=""):
        assert type in ["train", "valid", "test"]
        if type == "test":
            assert self.test_loader is not None

        self.model.eval()

        if type == "train":
            data_loader = self.train_loader
        elif type == "valid":
            data_loader = self.valid_loader
        elif type == "test":
            data_loader = self.test_loader
        else:
            raise NotImplementedError

        accu_loss = torch.zeros(1).to(self.device)  # 累计损失
        y_true_list, y_logit_list = [], []
        data_loader = tqdm(data_loader, desc=tqdm_desc)
        for step, data_package in enumerate(data_loader):
            with torch.no_grad():
                if self.data_type == "image":
                    images, labels = data_package["images"].to(self.device), data_package["labels"].to(self.device)
                    y_logit = self.get_image_logits(images)
                elif self.data_type == "graph":
                    data_package = data_package.to(self.device)
                    labels = data_package.y
                    y_logit = self.model(data_package)
                elif self.data_type == "geometry":  # {"coords": coords, "edge_attr": edge_attr, "edge_index": edge_index}
                    data_package = data_package.to(self.device)
                    labels = data_package.y
                    y_logit = self.model(data_package)
                elif self.data_type == "sequence":
                    data_package = {item: data_package[item].to(self.device) for item in data_package.keys() if
                                    isinstance(data_package[item], torch.Tensor)}
                    labels = data_package["label"]
                    y_logit = self.model(data_package)
                elif self.data_type == "feature":
                    feats, labels = data_package["feats"].to(self.device), data_package["labels"].to(self.device).double()
                    y_logit = self.model(feats)
                else:
                    raise NotImplementedError

                if self.label_empty is not None:
                    is_valid = labels != self.label_empty
                    loss_mat = self.criterion(y_logit.double(), labels)
                    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                else:
                    loss = self.criterion(y_logit, labels.view(y_logit.shape).float()).mean()

                # # logger
                accu_loss += loss.detach()
                data_loader.desc = "{} total loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

            if len(y_logit.size()) <= 1:  # 表示 y_logit 是一个一维向量。
                num = len(y_logit) if len(y_logit.size()) == 1 else 1
                y_logit = y_logit.reshape(num, -1)

            y_true_list.append(labels.view(y_logit.shape).cpu())
            y_logit_list.append(y_logit.cpu())

        y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
        y_logit = torch.cat(y_logit_list, dim=0).cpu().numpy()

        avg_loss = accu_loss.item() / (step + 1)
        results, data_dict = comprehensive_evaluate(y_true, y_logit, self.task_type, return_data_dict=True)
        return avg_loss, results, data_dict

    def train(self, num_epochs, eval_metric, valid_select, min_value, dataset="UNKOWN",
              save_data_prediction_dict=False, save_finetune_ckpt=False, save_dir=None,
              cache_detail_logs=False):
        if save_finetune_ckpt or save_finetune_ckpt:
            assert save_dir is not None
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        results = {
            'highest_valid': min_value,
            'final_train': min_value,
            'final_test': min_value,
            'highest_train': min_value,
        }
        if cache_detail_logs:
            results['highest_valid_desc'] = None
            results["final_train_desc"] = None
            results["final_test_desc"] = None

        for epoch in range(num_epochs):
            tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(dataset, epoch)
            loss = self.train_epoch(tqdm_desc=tqdm_train_desc)
            # evaluate
            train_loss, train_results, train_data_dict = self.evaluate_epoch(type="train", tqdm_desc=tqdm_eval_train_desc)
            val_loss, val_results, val_data_dict = self.evaluate_epoch(type="valid", tqdm_desc=tqdm_eval_val_desc)
            test_loss, test_results, test_data_dict = self.evaluate_epoch(type="test", tqdm_desc=tqdm_eval_test_desc)

            train_result = train_results[eval_metric]
            valid_result = val_results[eval_metric]
            test_result = test_results[eval_metric]

            self.log({"dataset": dataset, "epoch": epoch, "Loss": train_loss, 'metric': eval_metric,
                      'train': train_result, 'valid': valid_result, 'test': test_result})

            if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
                results['highest_train'] = train_result

            if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
                results['highest_valid'] = valid_result
                results['final_train'] = train_result
                results['final_test'] = test_result

                if cache_detail_logs:
                    results['highest_valid_desc'] = val_results
                    results['final_train_desc'] = train_results
                    results['final_test_desc'] = test_results

                if save_finetune_ckpt:
                    save_checkpoint({"model": self.model}, {"optimizer": self.optimizer}, None, results, epoch, save_dir, "", name_post='valid_best')

                if save_data_prediction_dict:
                    data_prediction_dict = {
                        "train_data_dict": train_data_dict,
                        "val_data_dict": val_data_dict,
                        "test_data_dict": test_data_dict,
                    }
                    np.savez(f"{save_dir}/prediction_dict.npz", **data_prediction_dict)

        # self.log("final results: {}\n".format(results))
        return results


