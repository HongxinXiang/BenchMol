import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool

from model_pools.base_utils import get_predictor
from model_pools.geom3d import *
from model_pools.geom3d.NequIP.model import model_from_config
from model_pools.geom3d.unimol import UniMol


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


class GeometryModelFactory(torch.nn.Module):
    """Reference: https://github.com/chao1224/Geom3D/blob/e44b11d4959aeb757789a2bbe5080b4ccdb485a1/examples_3D/finetune_QM9.py
    """
    def __init__(self, model_name, head_arch, num_tasks, head_arch_params=None, pretrain_gnn_path=None,
                 model_key=None, emb_dim=300, args=None, **kwargs):
        super(GeometryModelFactory, self).__init__()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params
        self.pretrain_gnn_path = pretrain_gnn_path
        self.model_key = model_key
        self.emb_dim = emb_dim

        self.args = args
        self.node_class, self.edge_class = 119, 5  # for molecules
        self.model, self.graph_pred_linear = self.get_model(self.args, self.num_tasks, self.node_class, self.edge_class)

    def forward(self, batch):
        if self.model_name == "SchNet":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch.batch)

        elif self.model_name in ["UniMol", "UniMol_no_pretrain"]:
            molecule_3D_repr = self.model(
                src_tokens=batch.src_tokens,
                src_distance=batch.src_distance,
                src_coord=batch.src_coord,
                src_edge_type=batch.src_edge_type
            )

        elif self.model_name == "DimeNet":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch.batch)

        elif self.model_name == "DimeNetPlusPlus":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch.batch)

        elif self.model_name == "TFN":
            x_one_hot = F.one_hot(batch.x, num_classes=self.node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=self.edge_class)
            node_3D_repr = self.model(
                x=x,
                positions=batch.coords,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif self.model_name == "SE3_Transformer":
            x_one_hot = F.one_hot(batch.x, num_classes=self.node_class).float()
            x = torch.cat([x_one_hot, batch.x.unsqueeze(1)], dim=1).float()
            edge_attr_one_hot = F.one_hot(batch.edge_attr[:, 0], num_classes=self.edge_class)
            node_3D_repr = self.model(
                x=x,
                positions=batch.coords,
                edge_index=batch.edge_index,
                edge_feat=edge_attr_one_hot,
            )
            molecule_3D_repr = global_max_pool(node_3D_repr, batch.batch)

        elif self.model_name == "EGNN":
            device = batch.x.device
            x_one_hot = F.one_hot(batch.x, num_classes=self.node_class).float()
            x = preprocess_input(
                x_one_hot,
                batch.x,
                charge_power=self.args.EGNN_charge_power,
                charge_scale=self.node_class,
                device=device
            )
            node_3D_repr = self.model(
                x=x,
                positions=batch.coords,
                edge_index=batch.edge_index,  # default is full_edge_index
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif self.model_name == "SphereNet":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch.batch)

        elif self.model_name == "SEGNN":
            molecule_3D_repr = self.model(batch)

        elif self.model_name == "PaiNN":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch.edge_index, batch.batch)  # radius_edge_index

        elif self.model_name == "GemNet":
            molecule_3D_repr = self.model(batch.x, batch.coords, batch)

        elif self.model_name in ["NequIP", "Allegro"]:
            data = {
                "edge_index": batch.radius_edge_index,
                "pos": batch.coords,
                "atom_types": batch.x,
                "batch": batch.batch,
            }
            out = self.model(data)
            molecule_3D_repr = out["total_energy"].squeeze()

        elif self.model_name == "Equiformer":
            molecule_3D_repr = self.model(node_atom=batch.x, pos=batch.coords, batch=batch.batch)

        if self.graph_pred_linear is not None:
            pred = self.graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_repr.squeeze()

        return pred

    def get_model(self, args, num_tasks, node_class, edge_class):
        if self.model_name == "SchNet":
            model = SchNet(
                hidden_channels=self.emb_dim,
                num_filters=args.SchNet_num_filters,
                num_interactions=args.SchNet_num_interactions,
                num_gaussians=args.SchNet_num_gaussians,
                cutoff=args.SchNet_cutoff,
                readout=args.SchNet_readout,
                node_class=node_class,
            )
            graph_pred_linear = None

        elif self.model_name == "UniMol":
            self.emb_dim = 512  # UniMol 的预训练模型必须是 512
            model = UniMol(return_repr=True, return_atomic_reprs=False, remove_hs=True, use_pretrained=True)
            graph_pred_linear = None

        elif self.model_name == "UniMol_no_pretrain":
            self.emb_dim = 512  # UniMol 的预训练模型必须是 512
            model = UniMol(return_repr=True, return_atomic_reprs=False, remove_hs=True, use_pretrained=False)
            graph_pred_linear = None

        elif self.model_name == "DimeNet":
            model = DimeNet(
                node_class=node_class,
                hidden_channels=self.emb_dim,
                out_channels=num_tasks,
                num_blocks=6,
                num_bilinear=8,
                num_spherical=7,
                num_radial=6,
                cutoff=10.0,
                envelope_exponent=5,
                num_before_skip=1,
                num_after_skip=2,
                num_output_layers=3,
            )
            graph_pred_linear = nn.Identity()

        elif self.model_name == "DimeNetPlusPlus":
            model = DimeNetPlusPlus(
                node_class=node_class,
                hidden_channels=self.emb_dim,
                out_channels=num_tasks,
                num_blocks=args.DimeNetPlusPlus_num_blocks,
                int_emb_size=args.DimeNetPlusPlus_int_emb_size,
                basis_emb_size=args.DimeNetPlusPlus_basis_emb_size,
                out_emb_channels=args.DimeNetPlusPlus_out_emb_channels,
                num_spherical=args.DimeNetPlusPlus_num_spherical,
                num_radial=args.DimeNetPlusPlus_num_radial,
                cutoff=args.DimeNetPlusPlus_cutoff,
                envelope_exponent=args.DimeNetPlusPlus_envelope_exponent,
                num_before_skip=args.DimeNetPlusPlus_num_before_skip,
                num_after_skip=args.DimeNetPlusPlus_num_after_skip,
                num_output_layers=args.DimeNetPlusPlus_num_output_layers,
            )
            graph_pred_linear = nn.Identity()

        elif self.model_name == "TFN":
            # This follows the dataset construction in oriGINal implementation
            # https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/QM9.py#L187
            atom_feature_size = node_class + 1
            model = TFN(
                atom_feature_size=atom_feature_size,
                edge_dim=edge_class,
                num_layers=args.TFN_num_layers,
                num_channels=args.TFN_num_channels,
                num_degrees=args.TFN_num_degrees,
                num_nlayers=args.TFN_num_nlayers,
            )
            latent_dim = args.TFN_num_channels * args.TFN_num_degrees
            graph_pred_linear = get_predictor(arch=self.head_arch, in_features=latent_dim, num_tasks=self.num_tasks,
                                              inner_dim=self.head_arch_params["inner_dim"],
                                              dropout=self.head_arch_params["dropout"],
                                              activation_fn=self.head_arch_params["activation_fn"])

        elif self.model_name == "SE3_Transformer":
            # This follows the dataset construction in oriGINal implementation
            # https://github.com/FabianFuchsML/se3-transformer-public/blob/master/experiments/qm9/QM9.py#L187
            atom_feature_size = node_class + 1
            model = SE3Transformer(
                atom_feature_size=atom_feature_size,
                edge_dim=edge_class,
                num_layers=args.SE3_Transformer_num_layers,
                num_channels=args.SE3_Transformer_num_channels,
                num_degrees=args.SE3_Transformer_num_degrees,
                num_nlayers=args.SE3_Transformer_num_nlayers,
                div=args.SE3_Transformer_div,
                n_heads=args.SE3_Transformer_n_heads,
            )
            latent_dim = (
                    args.SE3_Transformer_num_channels * args.SE3_Transformer_num_degrees
            )
            graph_pred_linear = get_predictor(arch=self.head_arch, in_features=latent_dim, num_tasks=self.num_tasks,
                                              inner_dim=self.head_arch_params["inner_dim"],
                                              dropout=self.head_arch_params["dropout"],
                                              activation_fn=self.head_arch_params["activation_fn"])

        elif self.model_name == "EGNN":
            in_node_nf = node_class * (1 + args.EGNN_charge_power)
            model = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=0,
                hidden_nf=self.emb_dim,
                n_layers=args.EGNN_n_layers,
                positions_weight=args.EGNN_positions_weight,
                attention=args.EGNN_attention,
                node_attr=args.EGNN_node_attr,
            )
            graph_pred_linear = None

        elif self.model_name == "SphereNet":
            model = SphereNet(
                hidden_channels=self.emb_dim,
                out_channels=num_tasks,
                energy_and_force=False,
                cutoff=args.SphereNet_cutoff,
                num_layers=args.SphereNet_num_layers,
                int_emb_size=args.SphereNet_int_emb_size,
                basis_emb_size_dist=args.SphereNet_basis_emb_size_dist,
                basis_emb_size_angle=args.SphereNet_basis_emb_size_angle,
                basis_emb_size_torsion=args.SphereNet_basis_emb_size_torsion,
                out_emb_channels=args.SphereNet_out_emb_channels,
                num_spherical=args.SphereNet_num_spherical,
                num_radial=args.SphereNet_num_radial,
                envelope_exponent=args.SphereNet_envelope_exponent,
                num_before_skip=args.SphereNet_num_before_skip,
                num_after_skip=args.SphereNet_num_after_skip,
                num_output_layers=args.SphereNet_num_output_layers,
            )
            graph_pred_linear = nn.Identity()

        elif self.model_name == "PaiNN":
            model = PaiNN(
                n_atom_basis=self.emb_dim,  # default is 64
                n_interactions=args.PaiNN_n_interactions,
                n_rbf=args.PaiNN_n_rbf,
                cutoff=args.PaiNN_radius_cutoff,
                max_z=node_class,
                n_out=num_tasks,
                readout=args.PaiNN_readout,
            )
            graph_pred_linear = model.create_output_layers()

        elif self.model_name == "SEGNN":  #TODO: unsupported
            model = SEGNN(
                node_class,
                num_tasks,
                hidden_features=self.emb_dim,
                N=args.SEGNN_radius,
                lmax_h=args.SEGNN_N,
                lmax_pos=args.SEGNN_lmax_pos,
                norm=args.SEGNN_norm,
                pool=args.SEGNN_pool,
                edge_inference=args.SEGNN_edge_inference
            )
            graph_pred_linear = nn.Identity()

        elif self.model_name == "NequIP":  #TODO: unsupported
            config = dict(
                model_builders=[
                    "SimpleIrrepsConfig",
                    "EnergyModel",
                ],
                dataset_statistics_stride=1,
                chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],

                r_max=args.NequIP_radius_cutoff,
                num_layers=5,

                chemical_embedding_irreps_out="64x0e",

                l_max=1,
                parity=True,
                num_features=64,

                nonlinearity_type="gate",
                nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
                nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
                resnet=False,
                num_basis=8,
                BesselBasis_trainable=True,
                PolynomialCutoff_p=6,
                invariant_layers=3,
                invariant_neurons=64,
                avg_num_neighbors=8,
                use_sc=True,
                compile_model=False,
            )
            model = model_from_config(config=config, initialize=True)
            graph_pred_linear = None

        elif self.model_name == "Allegro":  #TODO: unsupported
            config = dict(
                model_builders=[
                    "Geom3D.models.Allegro.model.Allegro",
                ],
                dataset_statistics_stride=1,
                chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],
                default_dtype="float32",
                allow_tf32=False,
                model_debug_mode=False,
                equivariance_test=False,
                grad_anomaly_mode=False,
                _jit_bailout_depth=2,
                _jit_fusion_strategy=[("DYNAMIC", 3)],
                r_max=args.NequIP_radius_cutoff,
                num_layers=5,
                l_max=1,
                num_features=64,
                nonlinearity_type="gate",
                nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
                nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
                num_basis=8,
                BesselBasis_trainable=True,
                PolynomialCutoff_p=6,
                invariant_layers=3,
                invariant_neurons=64,
                avg_num_neighbors=8,
                use_sc=True,

                parity="o3_full",
                mlp_latent_dimensions=[512],
            )
            model = model_from_config(config=config, initialize=True)
            graph_pred_linear = None

        elif self.model_name == "GemNet":  #TODO: unsupported
            model = GemNet(
                # node_class=93,
                node_class=node_class,
                num_spherical=args.GemNet_num_spherical,
                num_radial=args.GemNet_num_radial,
                num_blocks=args.GemNet_num_blocks,
                emb_size_atom=self.emb_dim,
                emb_size_edge=self.emb_dim,
                emb_size_trip=args.GemNet_emb_size_trip,
                emb_size_quad=args.GemNet_emb_size_quad,
                emb_size_rbf=args.GemNet_emb_size_rbf,
                emb_size_cbf=args.GemNet_emb_size_cbf,
                emb_size_sbf=args.GemNet_emb_size_sbf,
                emb_size_bil_quad=args.GemNet_emb_size_bil_quad,
                emb_size_bil_trip=args.GemNet_emb_size_bil_trip,
                num_before_skip=args.GemNet_num_before_skip,
                num_after_skip=args.GemNet_num_after_skip,
                num_concat=args.GemNet_num_concat,
                num_atom=args.GemNet_num_atom,
                cutoff=args.GemNet_cutoff,
                int_cutoff=args.GemNet_int_cutoff,
                triplets_only=args.GemNet_triplets_only,
                direct_forces=args.GemNet_direct_forces,
                envelope_exponent=args.GemNet_envelope_exponent,
                extensive=args.GemNet_extensive,
                forces_coupled=args.GemNet_forces_coupled,
                output_init=args.GemNet_output_init,
                activation=args.GemNet_activation,
                scale_file=args.GemNet_scale_file,
                num_targets=num_tasks,
            )
            graph_pred_linear = nn.Identity()

        elif self.model_name == "Equiformer": #TODO: unsupported
            if args.Equiformer_hyperparameter == 0:
                # This follows the hyper in Equiformer_l2
                model = EquiformerEnergy(
                    irreps_in=args.Equiformer_irreps_in,
                    max_radius=args.Equiformer_radius,
                    node_class=node_class,
                    number_of_basis=args.Equiformer_num_basis,
                    irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                    irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                    fc_neurons=[64, 64],
                    irreps_feature='512x0e',
                    irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                    rescale_degree=False, nonlinear_message=False,
                    irreps_mlp_mid='384x0e+192x1e+96x2e',
                    norm_layer='layer',
                    alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
            elif args.Equiformer_hyperparameter == 1:
                # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
                model = EquiformerEnergy(
                    irreps_in=args.Equiformer_irreps_in,
                    max_radius=args.Equiformer_radius,
                    node_class=node_class,
                    number_of_basis=args.Equiformer_num_basis,
                    irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                    irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                    fc_neurons=[64, 64], basis_type='bessel',
                    irreps_feature='512x0e',
                    irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                    rescale_degree=False, nonlinear_message=True,
                    irreps_mlp_mid='384x0e+192x1e+96x2e',
                    norm_layer='layer',
                    alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
            else:
                raise NotImplementedError
            graph_pred_linear = nn.Identity()

        else:
            raise Exception("3D model {} not included.".format(self.model_name))

        if graph_pred_linear is None:
            graph_pred_linear = get_predictor(arch=self.head_arch, in_features=self.emb_dim, num_tasks=self.num_tasks,
                                           inner_dim=self.head_arch_params["inner_dim"],
                                           dropout=self.head_arch_params["dropout"],
                                           activation_fn=self.head_arch_params["activation_fn"])

        return model, graph_pred_linear


if __name__ == '__main__':
    from task.fine_tuning import parse_args
    args = parse_args()
    print(args)
    model = GeometryModelFactory(model_name="EGNN", head_arch="arch3", num_tasks=5, emb_dim=300, args=args)
    print(model)

