import torch
from utils.evaluate import metric as utils_evaluate_metric
from utils.evaluate import metric_multitask as utils_evaluate_metric_multitask
from utils.evaluate import metric_reg as utils_evaluate_metric_reg
from utils.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask


def comprehensive_evaluate(y_true, y_logit, task_type, return_data_dict=False):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(len(y_true), -1)

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_logit))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()

            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
            else:
                return utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_logit}
                return utils_evaluate_metric_reg(y_true, y_logit), data_dict
            else:
                return utils_evaluate_metric_reg(y_true, y_logit)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_logit))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()

            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1), data_dict
            else:
                return utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_logit}
                return utils_evaluate_metric_reg_multitask(y_true, y_logit, num_tasks=y_true.shape[1]), data_dict
            else:
                return utils_evaluate_metric_reg_multitask(y_true, y_logit, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")
