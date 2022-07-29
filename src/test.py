from cfg_train import CFGTrainer
from cfg_config import cfg_args
import torch
import os
from utils import create_dir_if_not_exists, write_log_file
from utils import generate_epoch_pair
from model.DenseGraphMatching import MultiLevelGraphMatchNetwork

if __name__ == '__main__':
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_args.gpu_index)
    
    graph_init_dim = cfg_args.graph_init_dim

    model = MultiLevelGraphMatchNetwork(node_init_dims=graph_init_dim, arguments=cfg_args, device=d).to(d)

    model.load_state_dict(torch.load('ffmpeg_Min3_Max200.BestModel'))

    print(model)
