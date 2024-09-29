import torch
from torch import nn

from model_pools.geom3d.UniMol.unimol_tools.models.unimol import UniMolModel, molecule_architecture


class UniMol(nn.Module):
    def __init__(self, num_tasks=2, data_type="molecule", return_repr=False, return_atomic_reprs=False, remove_hs=True, use_pretrained=True):
        super().__init__()
        self.return_repr = return_repr
        self.return_atomic_reprs = return_atomic_reprs
        self.args = molecule_architecture()
        self.params = {"remove_hs": remove_hs}
        self.model = UniMolModel(output_dim=num_tasks, data_type=data_type, use_pretrained=use_pretrained, **self.params)
        self.dictionary = self.model.dictionary
        if return_atomic_reprs:
            self.model.classification_head = nn.Identity()

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type):
        ret_dict = self.model(src_tokens, src_distance, src_coord, src_edge_type, return_repr=self.return_repr,
                              return_atomic_reprs=self.return_atomic_reprs)
        if self.return_repr and not self.return_atomic_reprs:
            return ret_dict["cls_repr"]
        return ret_dict


if __name__ == '__main__':
    from model_pools.geom3d.UniMol.unimol_tools.data.conformer import coords2unimol
    import lmdb, pickle

    args = molecule_architecture()
    print(args)

    params = {"remove_hs": True}
    num_tasks = 1
    model = UniMolModel(output_dim=num_tasks, data_type="molecule", **params)
    # model.classification_head = torch.nn.Identity()
    print(model)

    dictionary = model.dictionary

    lmdb_path = "./UniMol/example_data/bace/test.lmdb"
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)

        data_dict = coords2unimol(data["atoms"], data["coordinates"][0], dictionary, max_atoms=256)

        src_tokens, src_distance, src_coord, src_edge_type = torch.from_numpy(data_dict["src_tokens"]).unsqueeze(
            0), torch.from_numpy(data_dict["src_distance"]).unsqueeze(0), torch.from_numpy(
            data_dict["src_coord"]).unsqueeze(0), torch.from_numpy(data_dict["src_edge_type"]).unsqueeze(0)

        y_pred = model(src_tokens, src_distance, src_coord, src_edge_type, return_repr=False, return_atomic_reprs=False)

        print(123)