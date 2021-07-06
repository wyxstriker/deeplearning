import torch.nn
import os
from torch.utils import data
from torchvision import datasets, transforms
from options import HiDDenConfiguration, TrainingOptions
from average_meter import AverageMeter

import utils
from model.hidden import *
from noise_layers.noiser import Noiser


def test(model1: Hidden,
         model2: Hidden,
         device: torch.device,
         hidden_config: HiDDenConfiguration,
         train_options: TrainingOptions):
    test_transforms = transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    train_options.validation_folder = 'data/test/'
    test_images = datasets.ImageFolder(train_options.validation_folder, test_transforms)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    ori = AverageMeter()
    end = AverageMeter()
    for image, _ in test_loader:
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

        batch_size = image.shape[0]

        model2.encoder_decoder.eval()
        model1.discriminator.eval()
        with torch.no_grad():
            d_on_cover = model1.discriminator(image)
            ori.update(torch.sum(torch.sigmoid(d_on_cover)).item()/batch_size)
    print("ori: "+str(ori.avg))


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # this_run_folder = 'experiments/no-noise adam-eps-1e-4'
    # this_run_folder = 'runs/stog-noise 2021.06.25--13-26-14'
    this_run_folder = 'experiments/combined-noise'
    options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
    train_options, hidden_config, noise_config = utils.load_options(options_file)

    last_file = utils.sorted_nicely(os.listdir(os.path.join(this_run_folder, 'checkpoints')))[-1]
    checkpoint1 = os.path.join(os.path.join(this_run_folder, 'checkpoints'), last_file)
    checkpoint1 = torch.load(checkpoint1)

    last_file = utils.sorted_nicely(os.listdir(os.path.join(this_run_folder, 'checkpoints')))[-1]
    checkpoint2 = os.path.join(os.path.join(this_run_folder, 'checkpoints'), last_file)
    checkpoint2 = torch.load(checkpoint2)

    noiser = Noiser(noise_config, device)
    model1 = Hidden(hidden_config, device, noiser, None)
    model2 = Hidden(hidden_config, device, noiser, None)

    utils.model_from_checkpoint(model1, checkpoint1)
    utils.model_from_checkpoint(model2, checkpoint2)

    test(model1, model2, device, hidden_config, train_options)
