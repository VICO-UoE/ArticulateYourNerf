import argparse
import json

def config_parser(return_parser=False):
    parser = argparse.ArgumentParser()

    ################################## JSON config file ##################################

    parser.add_argument('--config', type=str, required=True, help="config file for runing")

    ################################## Dataset config ######################################
    
    parser.add_argument('--root_dir', type=str, help="root dir for images")
    parser.add_argument('--img_wh', type=int, default=[640, 480], nargs='+', help="image size")
    parser.add_argument('--motion_gt_json', type=str, help="json file that stores the motion gt")
    
    ################################## NGP config ######################################
    parser.add_argument('--aabb', type=int, default=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], nargs='+', help="config file for runing")
    parser.add_argument('--near_plane', type=float, default=0.0, help="near plane")
    parser.add_argument('--far_plane', type=float, default=1e10, help="far plane")
    parser.add_argument('--grid_resolution', type=int, default=128, help="grid resolution")
    parser.add_argument('--grid_nlvl', type=int, default=1, help="number of grid level")
    parser.add_argument('--render_step_size', type=float, default=5e-3, help="render step size")
    parser.add_argument('--cone_angle', type=float, default=0.0, help="cone_angle")
    parser.add_argument('--alpha_thre', type=float, default=0.0, help="alpha_thre")
    
    ################################## NGP config for prop ######################################
    parser.add_argument('--opaque_bkgd', type=bool, default=False, help="make background opaque")
    parser.add_argument('--sampling_type', type=str, default='uniform', help="sampling method")
    parser.add_argument('--num_samples', type=int, default=128, help="number of samples")
    parser.add_argument('--num_samples_per_prop', type=int, nargs='+', default=[128], help="number of samples for each estimator")
    parser.add_argument('--use_art_seg_estimator', type=bool, default=True, help="whether use custom prop estimator")
    
    parser.add_argument('--state', type=str, default='start', help="sampling method")
    ################################## trainig ######################################
    parser.add_argument('--max_steps', type=int, default=20000, help="max steps for training")
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size for training")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay")
    parser.add_argument('--resume_training', type=bool, default=False, help="whether to resume from pre-trained")
    parser.add_argument('--eval_step', type=int, default=10000, help="steps to run eval")
    parser.add_argument('--traverse_steps_limit', type=int, default=128, help="maximum samples per ray")
    parser.add_argument('--pre_trained_weights', type=str, default=None, help='pre trained weights to load for the model')
    parser.add_argument('--use_opa_entropy', type=bool, default=False, help="whether to use_opa_entropy loss for segmentation.")
    parser.add_argument('--accum_steps', type=int, default=50, help="whether to gradient accumulation during pose optimization with NeRF.")
    parser.add_argument('--pose_lr', type=float, default=1e-2, help="learning rate for pose estimation during joint estimation.")
    parser.add_argument('--use_num_frames', type=int, default=101, help="number of articulated frames used for training")
    parser.add_argument('--use_timestamp', type=bool, default=False, help="whether to cerate folder in exp with timestamp")
    parser.add_argument('--ignore_empty', type=bool, default=False, help="whether to ignore empty pixels during training")
    parser.add_argument('--gpu_id', type=int, default=0, help="choose GPU for training, default to GPU:0")
    parser.add_argument('--idx_list', type=int, default=None, nargs='+', help="idx for training frames")
    parser.add_argument('--voxel_res', type=int, default=128, help="voxel resolution for part initialization")
    parser.add_argument('--num_dy_parts', type=int, default=2, help="number of dynamic parts for motion estimation, only applicable in multipart object")
    parser.add_argument('--imp_sampling', type=bool, default=False, help="whether to use importance sampling during training")
    ################################## outputs ######################################
    parser.add_argument('--ckpt_step', type=int, default=10000, help="ckpt save steps")
    parser.add_argument('--output_dir', type=str, default='results', help="path to save outputs")
    parser.add_argument('--exp_name', type=str, default='debug', help="tag for exps")
    
    ################################## segmentation ######################################
    
    parser.add_argument('--seg_classes', type=int, default=2, help="ckpt save steps")
    parser.add_argument('--pretrained_config', type=str, default=None, help="config files for loading pretrained NeRF")
    parser.add_argument('--use_background', type=bool, default=False, help="whether to use background classes")
    parser.add_argument('--use_se3', type=bool, default=True, help="whether to se3 parameterization for pose estimation")
    parser.add_argument('--motion_type', type=str, default='r', help="motion type of the joint, options: r for revolute, p for prismatic")
    parser.add_argument('--eps', type=float, default=0.5, help="eps used for clustering in pose estimator")
    parser.add_argument('--use_init_seg', type=bool, default=True, help="eps used for clustering in pose estimator")
    parser.add_argument('--init_lr', type=float, default=1e-4, help="lr for init seg")
    parser.add_argument('--init_step', type=int, default=1000, help="steps for trainig init seg")
    parser.add_argument('--init_lr_decay', type=float, default=0.5, help="lr decay for init seg")
    parser.add_argument('--init_start_step', type=int, default=1000, help="when to start init_seg")
    parser.add_argument('--init_interval_steps', type=int, default=1000, help="when to start init_seg")
    parser.add_argument('--pose_accum_step', type=int, default=1000, help="when to start init_seg")
    parser.add_argument('--pose_accum_iter', type=int, default=4, help="when to start init_seg")
    parser.add_argument('--pose_scheduler_step', type=int, default=500, help="when to start init_seg")
    parser.add_argument('--min_points', type=int, default=100, help="min points for valid clusters")
    ################################## registration ######################################
    parser.add_argument('--voxel_fname', type=str, default='results/voxel_pcd.ply', help="ckpt save steps")
    parser.add_argument('--input_dim', type=int, default=3, help="input dimension to the voxel branch")
    ################################## ablation ######################################
    parser.add_argument('--skip_init', type=bool, default=False, help="whether to skip voxel point initialization")
    parser.add_argument('--end_to_end', type=bool, default=False, help="whether to use end-to-end estimation for segmentation and motion estimation")
    if return_parser:
        return parser
    else:
        args = parser.parse_args()

        return args

def get_opts(argv_string=None):

    parser = config_parser(return_parser=True)
    if argv_string is not None:
        args = parser.parse_args(argv_string)
    else:
        args = parser.parse_args()
    # Load and parse the JSON configuration file
    with open(args.config, "r") as config_file:
        config_data = json.load(config_file)
        
    # Update the args namespace with loaded JSON data
    for key, value in config_data.items():
        setattr(args, key, value)
        
    return args