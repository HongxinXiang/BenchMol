


def add_geometry_model_params(parser):
    """https://github.com/chao1224/Geom3D/blob/e44b11d4959aeb757789a2bbe5080b4ccdb485a1/examples_3D/config.py
    """
    # for SchNet
    parser.add_argument("--SchNet_num_filters", type=int, default=128)
    parser.add_argument("--SchNet_num_interactions", type=int, default=6)
    parser.add_argument("--SchNet_num_gaussians", type=int, default=51)
    parser.add_argument("--SchNet_cutoff", type=float, default=10)
    parser.add_argument("--SchNet_readout", type=str, default="mean", choices=["mean", "add"])
    parser.add_argument("--SchNet_gamma", type=float, default=None)

    # for TFN
    parser.add_argument("--TFN_num_layers", type=int, default=7)
    parser.add_argument("--TFN_num_channels", type=int, default=32)
    parser.add_argument("--TFN_num_degrees", type=int, default=4)
    parser.add_argument("--TFN_num_nlayers", type=int, default=1)

    # for SE(3)-Transformer
    parser.add_argument("--SE3_Transformer_num_layers", type=int, default=7)
    parser.add_argument("--SE3_Transformer_num_channels", type=int, default=32)
    parser.add_argument("--SE3_Transformer_num_degrees", type=int, default=4)
    parser.add_argument("--SE3_Transformer_num_nlayers", type=int, default=1)
    parser.add_argument("--SE3_Transformer_div", type=int, default=2)
    parser.add_argument("--SE3_Transformer_n_heads", type=int, default=8)

    # for EGNN
    parser.add_argument("--EGNN_n_layers", type=int, default=7)
    parser.add_argument("--EGNN_attention", type=int, default=1)
    parser.add_argument("--EGNN_node_attr", type=int, default=0)
    parser.add_argument("--EGNN_positions_weight", type=float, default=1.0)
    parser.add_argument("--EGNN_charge_power", type=int, default=2)
    parser.add_argument("--EGNN_radius_cutoff", type=float, default=5.0)

    # for DimeNet++
    parser.add_argument("--DimeNetPlusPlus_num_blocks", type=int, default=4)
    parser.add_argument("--DimeNetPlusPlus_int_emb_size", type=int, default=64)
    parser.add_argument("--DimeNetPlusPlus_basis_emb_size", type=int, default=8)
    parser.add_argument("--DimeNetPlusPlus_out_emb_channels", type=int, default=128)
    parser.add_argument("--DimeNetPlusPlus_num_spherical", type=int, default=7)
    parser.add_argument("--DimeNetPlusPlus_num_radial", type=int, default=6)
    parser.add_argument("--DimeNetPlusPlus_cutoff", type=float, default=5.0)
    parser.add_argument("--DimeNetPlusPlus_envelope_exponent", type=int, default=5)
    parser.add_argument("--DimeNetPlusPlus_num_before_skip", type=int, default=1)
    parser.add_argument("--DimeNetPlusPlus_num_after_skip", type=int, default=2)
    parser.add_argument("--DimeNetPlusPlus_num_output_layers", type=int, default=3)
    parser.add_argument("--DimeNetPlusPlus_readout", type=str, default="add", choices=["mean", "add"])

    # for SphereNet
    parser.add_argument("--SphereNet_cutoff", type=float, default=5.0)
    parser.add_argument("--SphereNet_num_layers", type=int, default=4)
    parser.add_argument("--SphereNet_int_emb_size", type=int, default=64)
    parser.add_argument("--SphereNet_basis_emb_size_dist", type=int, default=8)
    parser.add_argument("--SphereNet_basis_emb_size_angle", type=int, default=8)
    parser.add_argument("--SphereNet_basis_emb_size_torsion", type=int, default=8)
    parser.add_argument("--SphereNet_out_emb_channels", type=int, default=256)
    parser.add_argument("--SphereNet_num_spherical", type=int, default=3)
    parser.add_argument("--SphereNet_num_radial", type=int, default=6)
    parser.add_argument("--SphereNet_envelope_exponent", type=int, default=5)
    parser.add_argument("--SphereNet_num_before_skip", type=int, default=1)
    parser.add_argument("--SphereNet_num_after_skip", type=int, default=2)
    parser.add_argument("--SphereNet_num_output_layers", type=int, default=3)

    # for SEGNN
    parser.add_argument("--SEGNN_radius", type=float, default=2)
    parser.add_argument("--SEGNN_N", type=int, default=7)
    parser.add_argument("--SEGNN_lmax_h", type=int, default=2)
    parser.add_argument("--SEGNN_lmax_pos", type=int, default=3)
    parser.add_argument("--SEGNN_norm", type=str, default="instance")
    parser.add_argument("--SEGNN_pool", type=str, default="avg")
    parser.add_argument("--SEGNN_edge_inference", type=int, default=0)

    # for PaiNN
    parser.add_argument("--PaiNN_radius_cutoff", type=float, default=5.0)
    parser.add_argument("--PaiNN_n_interactions", type=int, default=3)
    parser.add_argument("--PaiNN_n_rbf", type=int, default=20)
    parser.add_argument("--PaiNN_readout", type=str, default="add", choices=["mean", "add"])
    parser.add_argument("--PaiNN_gamma", type=float, default=None)

    # for GemNet
    parser.add_argument("--GemNet_num_spherical", type=int, default=7)
    parser.add_argument("--GemNet_num_radial", type=int, default=6)
    parser.add_argument("--GemNet_num_blocks", type=int, default=4)
    parser.add_argument("--GemNet_emb_size_trip", type=int, default=64)
    parser.add_argument("--GemNet_emb_size_quad", type=int, default=32)
    parser.add_argument("--GemNet_emb_size_rbf", type=int, default=16)
    parser.add_argument("--GemNet_emb_size_cbf", type=int, default=16)
    parser.add_argument("--GemNet_emb_size_sbf", type=int, default=32)
    parser.add_argument("--GemNet_emb_size_bil_trip", type=int, default=64)
    parser.add_argument("--GemNet_emb_size_bil_quad", type=int, default=32)
    parser.add_argument("--GemNet_num_before_skip", type=int, default=1)
    parser.add_argument("--GemNet_num_after_skip", type=int, default=1)
    parser.add_argument("--GemNet_num_concat", type=int, default=1)
    parser.add_argument("--GemNet_num_atom", type=int, default=2)
    parser.add_argument("--GemNet_cutoff", type=float, default=5.)
    parser.add_argument("--GemNet_int_cutoff", type=float, default=10.)
    parser.add_argument("--GemNet_triplets_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--GemNet_direct_forces", type=int, default=0, choices=[0, 1])
    parser.add_argument("--GemNet_envelope_exponent", type=int, default=5)
    parser.add_argument("--GemNet_extensive", type=int, default=1, choices=[0, 1])
    parser.add_argument("--GemNet_forces_coupled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--GemNet_output_init", type=str, default="HeOrthogonal")
    parser.add_argument("--GemNet_activation", type=str, default="swish")
    parser.add_argument("--GemNet_scale_file", type=str, default="scaling_factors.json")

    # for NequIP and Allegro
    parser.add_argument("--NequIP_radius_cutoff", type=float, default=4.)

    # For Equiformer
    parser.add_argument("--Equiformer_radius", type=float, default=5)
    parser.add_argument("--Equiformer_irreps_in", type=str, default="5x0e")
    parser.add_argument("--Equiformer_num_basis", type=int, default=128)
    parser.add_argument("--Equiformer_hyperparameter", type=int, default=0)

    # for ProNet
    parser.add_argument("--ProNet_level", type=str, default="aminoacid", choices=["aminoacid", "backbone", "allatom"])
    parser.add_argument("--ProNet_dropout", type=float, default=0.3)

    return parser