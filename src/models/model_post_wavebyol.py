import torch
import torch.nn as nn


class PostWaveBYOL(nn.Module):
    def __init__(self, encoder_model, projection_model, prediction_model):
        super(PostWaveBYOL, self).__init__()
        self.encoder_model = encoder_model
        self.projection_model = projection_model
        self.prediction_model = prediction_model
        self.target_encoder_model = None
        self.target_projection_model = None


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()