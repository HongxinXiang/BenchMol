import os

import timm
import torch

from benchmol.model_pools.base_utils import get_timm_model_names, get_predictor


class ImageModelFactory(torch.nn.Module):
    def __init__(self, model_name, head_arch, num_tasks, pretrained=False, head_arch_params=None, **kwargs):
        super(ImageModelFactory, self).__init__()

        assert model_name in get_timm_model_names()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        self.pretrained = pretrained
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params

        # create base model
        self.model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        # some attributes of base model
        self.classifier_name = self.model.default_cfg["classifier"]
        self.in_features = self.get_in_features()
        # self-defined head for prediction
        self_defined_head = self.create_self_defined_head()
        self.set_self_defined_head(self_defined_head)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        return self.model

    def get_in_features(self):
        if type(self.classifier_name) == str:
            if "." not in self.classifier_name and isinstance(getattr(self.model, self.classifier_name), torch.nn.modules.linear.Identity):
                in_features = self.model.num_features
            else:
                classifier = self.model
                for item in self.classifier_name.split("."):
                    classifier = getattr(classifier, item)
                in_features = classifier.in_features
        elif type(self.classifier_name) == tuple or type(self.classifier_name) == list:
            in_features = []
            for item_name in self.classifier_name:
                classifier = self.model
                for item in item_name.split("."):
                    classifier = getattr(classifier, item)
                in_features.append(classifier.in_features)
        else:
            raise Exception("{} is undefined.".format(self.classifier_name))
        return in_features

    def create_self_defined_head(self):
        if type(self.in_features) == list or type(self.in_features) == tuple:
            assert len(self.classifier_name) == len(self.in_features)
            head_predictor = []
            for item_in_features in self.in_features:
                single_predictor = get_predictor(arch=self.head_arch, in_features=item_in_features,
                                                 num_tasks=self.num_tasks,
                                                 inner_dim=self.head_arch_params["inner_dim"],
                                                 dropout=self.head_arch_params["dropout"],
                                                 activation_fn=self.head_arch_params["activation_fn"])
                head_predictor.append(single_predictor)
        elif type(self.classifier_name) == str:
            head_predictor = get_predictor(arch=self.head_arch, in_features=self.in_features, num_tasks=self.num_tasks,
                                           inner_dim=self.head_arch_params["inner_dim"],
                                           dropout=self.head_arch_params["dropout"],
                                           activation_fn=self.head_arch_params["activation_fn"])
        else:
            raise Exception("error type in classifier_name ({}) and in_features ({})".format(type(self.classifier_name),
                                                                                             type(self.in_features)))
        return head_predictor

    def set_self_defined_head(self, self_defined_head):
        if type(self.classifier_name) == list or type(self.classifier_name) == tuple:
            for predictor_idx, item_classifier_name in enumerate(self.classifier_name):
                classifier = self.model
                if "." in item_classifier_name:
                    split_classifier_name = item_classifier_name.split(".")
                    for i, item in enumerate(split_classifier_name):
                        classifier = getattr(classifier, item)
                        if i == len(split_classifier_name) - 2:
                            setattr(classifier, split_classifier_name[-1], self_defined_head[predictor_idx])
                else:
                    setattr(self.model, item_classifier_name, self_defined_head[predictor_idx])
        elif "." in self.classifier_name:
            classifier = self.model
            split_classifier_name = self.classifier_name.split(".")
            for i, item in enumerate(split_classifier_name):
                classifier = getattr(classifier, item)
                if i == len(split_classifier_name) - 2:
                    setattr(classifier, split_classifier_name[-1], self_defined_head)
        else:
            setattr(self.model, self.classifier_name, self_defined_head)

    def from_pretrained(self, pratrain_path, model_key=None, consistency=False, logger=None):
        log = logger.info if logger is not None else print
        if os.path.isfile(pratrain_path):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(pratrain_path))
            checkpoint = torch.load(pratrain_path)

            ckpt_model_state_dict = checkpoint[model_key]
            if consistency:  # model and ckpt_model_state_dict is consistent.
                self.model.load_state_dict(ckpt_model_state_dict)
                log("load all the parameters of pre-trianed model.")
            else:
                ckp_keys = list(ckpt_model_state_dict)
                cur_keys = list(self.model.state_dict())
                model_sd = self.model.state_dict()

                len_ckp_keys, len_cur_keys = len(ckp_keys), len(cur_keys)
                min_len = min(len_ckp_keys, len_cur_keys)
                ckp_keys = ckp_keys[:min_len]
                cur_keys = cur_keys[:min_len]

                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    if model_sd[cur_key].shape == ckpt_model_state_dict[ckp_key].shape:
                        model_sd[cur_key] = ckpt_model_state_dict[ckp_key]
                    else:
                        log(f"shape is inconsistency in {cur_key}; checkpoint is "
                            f"{ckpt_model_state_dict[ckp_key].shape}, current model is {model_sd[cur_key].shape}")

                self.model.load_state_dict(model_sd)
                log("load the first {} parameters. layer number: model({}), pretrain({})".format(min_len, len_cur_keys, len_ckp_keys))
        else:
            log("=> no checkpoint found at '{}'".format(pratrain_path))
