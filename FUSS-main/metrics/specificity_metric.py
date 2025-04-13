import os
import numpy as np
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

from utils.tensor_util import to_device, to_numpy
from utils.misc import plot_with_std
from utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_specificity(ssm_model, dataloader_train, logger, device, output_path):
    n_samples = 1000
    specificity_mean = []
    specificity_std = []
    logger.info(f'Calculating Specificity...')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        samples = ssm_model.generate_random_samples(n_samples=n_samples, n_modes=mode)
        samples = samples.reshape(n_samples, -1, 3).to(device)
        samples = samples - samples.mean(dim=1, keepdim=True)

        distances = np.zeros((n_samples, len(dataloader_train)))

        for index, data in enumerate(dataloader_train):
            data = to_device(data, device)
            target = (data['verts'].to(device) * data['face_area'].to(device))
            target = target.repeat(n_samples, 1, 1)

            loss, _ = chamfer_distance(target.float(), samples.float(), point_reduction=None, batch_reduction=None)
            distance = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))

            distance = to_numpy(distance)
            distances[:, index] = distance

        specificity_mean_value = distances.min(1).mean()
        specificity_std_value = distances.min(1).std()
        specificity_mean.append(specificity_mean_value)
        specificity_std.append(specificity_std_value)
        logger.info(f'Specificity for mode {mode} is {specificity_mean_value:.10f} +/- {specificity_std_value:.10f}')

    result_path = os.path.join(output_path, "specificity.png")
    specificity_mean = np.array(specificity_mean)
    specificity_std = np.array(specificity_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  specificity_mean, specificity_std,
                  "Specificity in mm", result_path)
    np.save(os.path.join(output_path, "specificity_mean.npy"), specificity_mean)
    np.save(os.path.join(output_path, "specificity_std.npy"), specificity_std)