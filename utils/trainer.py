import torch
from torch.optim import Adam

from locators.cnn import ConvNet
from utils.transformer import Transformer
from utils.dataloader import load_data
import Neurosmash


def criterion(y, t):
    """Computes loss as Euclidean distance from predicted to actual location.

    Args:
        y = [Tensor] predicted x and y pixel coordinates of an agent
        t = [Tensor] actual x and y pixel coordinates of an agent
    """
    distances = torch.sqrt((y[:, 0] - t[:, 0]) ** 2 + (y[:, 1] - t[:, 1]) ** 2)

    return distances.sum() / len(y)


def train_cnns(train_iter, validation_iter, num_epochs=100):
    """Trains and evaluates two convolutional neural networks that predict
       the locations of the red and blue agents, respectively.

    Args:
        train_iter      = [DataLoader] the training batch iterator
        validation_iter = [DataLoader] the validation batch iterator
        num_epochs      = [int] number of times all training data is sampled

    Returns [(nn.Module, nn.Module)]:
        The trained convolutional neural networks.
    """
    # initialize CNNs and optimizers
    cnns = ConvNet(device), ConvNet(device)
    optimizers = Adam(cnns[0].parameters()), Adam(cnns[1].parameters())
    indices = [0, 1], [2, 3]

    for cnn, optimizer, idx in zip(cnns, optimizers, indices):
        for i_epoch in range(num_epochs):
            # train CNN on training data
            train_losses = []
            for x, t in train_iter:
                y = cnn(x)
                loss = criterion(y, t[:, idx])
                train_losses.append(loss.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluate CNN on validation data
            validation_losses = []
            for x, t in validation_iter:
                y = cnn(x)
                loss = criterion(y, t[:, idx])
                validation_losses.append(loss.data)

            # print average losses per epoch
            print(f'Epoch {i_epoch + 1}:')
            avg_training_loss = torch.mean(torch.stack(train_losses))
            print(f'\tTraining loss: {avg_training_loss}')
            avg_validation_loss = torch.mean(torch.stack(validation_losses))
            print(f'\tValidation loss: {avg_validation_loss}')

    return cnns


if __name__ == '__main__':
    data_dir = '../data/'
    models_dir = '../models/'
    size = 64
    timescale = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    environment = Neurosmash.Environment(size=size, timescale=timescale)

    transformer = Transformer(
        size, bg_file_name=f'{data_dir}background_transposed_64.png'
    )
    data = load_data(transformer, train_split=1.0, test_split=0.0,
                     data_dir=data_dir, device=device)
    train_iter, validation_iter, test_iter = data

    cnn_red, cnn_blue = train_cnns(train_iter, train_iter)

    torch.save(cnn_red.state_dict(), f'{models_dir}cnn_red.pt')
    torch.save(cnn_blue.state_dict(), f'{models_dir}cnn_blue.pt')
