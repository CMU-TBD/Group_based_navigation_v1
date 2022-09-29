"""
This code is largely adopted from the sgan code developed by the Authors of SocialGAN
A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi.  Social GAN: Socially acceptable324trajectories with generative adversarial networks.  InProceedings of the IEEE Conference on325Computer Vision and Pattern Recognition (CVPR), pages 2255–2264, 2018

Author: Agrim Gupta
Link: https://github.com/agrimgupta92/sgan
"""


import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class SGANInference(object):

    def __init__(self, model_path):
        # To initialize it, a path to a pretrained model is needed
        # models are stored in sgan/models
        # for example: model_path = "models/sgan-models/eth_8_model.pt"
        #
        # model checkpoint names are like this
        # (dataset name)_(observation length)_model.pt
        #
        # dataset_name is the dataset that is the test set 
        # (the dataset that was not seen during the training of this model)
        #
        # obervation_length is the input length of the trajectory to this model

        path = model_path

        # number of samples to draw to get the final predicted trajectory
        self.num_samples = 20

        self.cuda = torch.device('cuda:0')
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        self.generator = self.get_generator(checkpoint)
        self.args = AttrDict(checkpoint['args'])
        return

    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.to(self.cuda)
        generator.eval()
        return generator

    def evaluate(self, obs_traj):
        # inputs:
        # depending on the observation length of your chosen model
        # the input obs_traj should be a numpy array of either size nx8x2 or nx12x2
        # where n is the number of people in the scene
        # obs_traj is simply the observed trajectories (sequence of coordinates)
        #
        # outputs:
        # outputs nx8x2 predicted trajectories (sequence of coordinates)

        num_people = obs_traj.shape[0]
        traj_length = obs_traj.shape[1]
        obs_traj_rel = obs_traj[:, 1:traj_length] - obs_traj[:, 0:traj_length - 1]
        obs_traj_rel = np.append(np.array([[[0,0]]] * num_people), obs_traj_rel, axis=1)
        obs_traj = np.transpose(obs_traj, (1, 0, 2))
        obs_traj_rel = np.transpose(obs_traj_rel, (1, 0, 2))
        seq_start_end = np.array([[0, num_people]])

        with torch.no_grad():
            obs_traj = torch.from_numpy(obs_traj).type(torch.float)
            obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
            seq_start_end = torch.from_numpy(seq_start_end).type(torch.int)
            obs_traj = obs_traj.to(self.cuda)
            obs_traj_rel = obs_traj_rel.to(self.cuda)
            seq_start_end = seq_start_end.to(self.cuda)

            pred_traj_avg = []
            for _ in range(self.num_samples):
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                tmp = pred_traj_fake.cpu().numpy()
                pred_traj_avg.append(np.transpose(tmp,(1,0,2)))

            pred_traj_avg = np.mean(np.asarray(pred_traj_avg), 0)

        return pred_traj_avg


