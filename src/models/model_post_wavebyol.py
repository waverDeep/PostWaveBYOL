import copy
import torch
import torch.nn as nn
import src.losses.loss as loss_function
import src.models.model as model
import matplotlib.pyplot as plt


class PostWaveBYOL(nn.Module):
    def __init__(self, encoder, projector, predictor, loss_function_name, ema_updator):
        super(PostWaveBYOL, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.target_encoder = None
        self.target_projector = None
        self.loss = loss_function.load_loss_function(loss_function_name)
        self.ema_updator = ema_updator
        self.get_target_network()

    def forward(self, x01, x02):
        online_encoding01 = self.encoder(x01)
        online_encoding02 = self.encoder(x02)
        online_projection01 = self.projector(online_encoding01)
        online_projection02 = self.projector(online_encoding02)
        online_prediction01 = self.predictor(online_projection01)
        online_prediction02 = self.predictor(online_projection02)

        with torch.no_grad():
            target_encoding01 = self.target_encoder(x01)
            target_encoding02 = self.target_encoder(x02)
            target_projection01 = self.target_projector(target_encoding01)
            target_projection02 = self.target_projector(target_encoding02)

        loss01 = self.loss(online_prediction01, target_projection02)
        loss02 = self.loss(online_prediction02, target_projection01)
        total_loss = loss01 + loss02
        total_loss = total_loss.mean()
        return total_loss

    def get_representation(self, x):
        encoding = self.encoder(x)
        return encoding

    def get_target_network(self):
        if self.target_encoder is None or self.target_projector is None:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_projector = copy.deepcopy(self.projector)
            model.set_requires_grad(self.target_encoder, requires=False)
            model.set_requires_grad(self.target_projector, requires=False)

    def update_target_weight(self):
        if self.target_encoder is None or self.target_projector is None:
            self.get_target_network()
        else:
            model.update_moving_average(self.ema_updator, self.target_encoder, self.encoder)
            model.update_moving_average(self.ema_updator, self.target_projector, self.projector)



class Encoder(nn.Module):
    def __init__(self, feature_extractor, feature_encoder):
        super(Encoder, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_encoder = feature_encoder

    def forward(self, x):
        feature_extraction = self.feature_extractor(x)
        reshape_feature_extraction = torch.transpose(feature_extraction, 1, 2)
        feature_encoding = self.feature_encoder(reshape_feature_extraction)
        reshape_feature_encoding = torch.transpose(feature_encoding, 1, 2)
        return reshape_feature_encoding


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 stride: list, filter_size: list, padding: list):
        super(FeatureExtractor, self).__init__()
        assert (
            len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.feature_extractor = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.feature_extractor.add_module(
                "feature_extraction_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class FeatureEncoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_layers: int):
        super(FeatureEncoder, self).__init__()
        self.feature_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.feature_encoder = nn.TransformerEncoder(self.feature_encoder_layer, num_layers=num_layers)

    def forward(self, x):
        out = self.feature_encoder(x)
        return out


class Projector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=out_dim),
        )

    def forward(self, x):
        out = self.projector(x)
        return out


class Predictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out = self.predictor(x)
        return out


def main():
    feature_extractor = FeatureExtractor(
        input_dim=1,
        hidden_dim=512,
        stride=[5, 4, 2, 2, 2, 2],
        filter_size=[10, 8, 4, 2, 2, 2],
        padding=[2, 2, 2, 2, 2, 1]
    )

    feature_encoder = FeatureEncoder(
        d_model=512,
        n_head=64,
        num_layers=6
    )

    encoder = Encoder(
        feature_extractor=feature_extractor,
        feature_encoder=feature_encoder
    )

    projector = Projector(
        input_dim=512,
        hidden_dim=4096,
        out_dim=4096
    )

    predictor = Predictor(
        input_dim=4096,
        hidden_dim=4096,
        out_dim=4096
    )

    ema_updator = model.EMA(0.99)

    post_waveBYOL = PostWaveBYOL(
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        loss_function_name="BYOLLoss",
        ema_updator=ema_updator
    )

    sample_data01 = torch.rand(1, 1, 20480)
    sample_data02 = torch.rand(1, 1, 20480)
    out_loss = post_waveBYOL(sample_data01, sample_data02)
    print(out_loss)

    representation = post_waveBYOL.get_representation(sample_data01)
    print(representation.size())

    out = representation.detach()
    out = out.squeeze(0)
    out = out.numpy()

    plt.pcolor(out)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()