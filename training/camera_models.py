import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import scipy.signal.windows
import logging
module_logger = logging.getLogger(__name__)

import constants

class SensorStatistics:
    def __init__(self):
        self.tracking = False
        self.reset()
        self.enable()

    def reset(self):
        self.min = None
        self.mean = None
        self.max = None
        self.N = 0

    @torch.no_grad()
    def update(self, x):
        if self.enabled:
            assert x.ndim == 2

            x_min = x.min(0)[0]
            x_mean = x.mean(0)
            x_max = x.max(0)[0]
            if self.N == 0:
                self.min = x_min.clone()
                self.mean = x_mean.clone()
                self.max = x_max.clone()
            else:
                self.min = torch.minimum(self.min, x_min)
                self.mean = (self.mean * self.N + x_mean * x.shape[0]) / \
                    (self.N + x.shape[0])
                self.max = torch.maximum(self.max, x_max)
            self.N += x.shape[0]

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __str__(self):
        return "Sensor Statistics \n(min : %s, \n mean: %s, \n max : %s)" % \
            (self.min.cpu() if self.min is not None else 0,
             self.mean.cpu() if self.mean is not None else 0,
             self.max.cpu() if self.max is not None else 0)


class Baseline(nn.Module):
    """
    A traditional camera followed by a task network.
    Only the task network is trainable.

    The input to the baseline camera is an appropriately downsampled image.
    This image is fed directly into the task network. As such, the dataset
    should contain downsampled images.
    """
    def __init__(self, output_size, img_size, hidden_layer_size,
                 hidden_layer_count):
        super().__init__()

        self.img_size = img_size

        # Hard-coded sensor noise values to emulate an 8-bit camera
        self.register_buffer("sensor_n_bits",
                             torch.tensor(8, dtype=torch.int64))
        self.register_buffer("read_noise_std",
                             torch.tensor(1/255, dtype=torch.float32))

        task_network_input = img_size[0] * img_size[1]
        self.task_network = TaskNetwork(\
            task_network_input, output_size, hidden_layer_size,
            hidden_layer_count)
        self.sensor_stats = SensorStatistics()
        self.sensor_stats.disable()

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        # Read noise
        x = x + torch.randn_like(x) * self.read_noise_std

        # Quantization noise
        x = x + torch.rand_like(x) * (1 / 2**self.sensor_n_bits)

        x = torch.clamp(x, 0, torch.inf)

        self.sensor_stats.update(x)

        task_network_output = self.task_network(x)

        return task_network_output

    @torch.no_grad()
    def visualize_mask(self, *args):
        """
        Returns None since the baseline camera does not store any masks
        """
        return None


class Mincam(nn.Module):
    def __init__(self,
                 output_size,
                 img_size,
                 mincam_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 realistic_sensor=False,
                 sensor_gain=1,
                 sensor_n_bits=12,
                 sensor_saturation_val=1,
                 read_noise_std=1/255,
                 mask_init_method="random",
                 simulate_pd_area_blur=False,
                 mask_blur_kernel_sigma=0,
                 simulate_directivity=False,
                 mask_min_value=0,
                 mask_max_value=1,
                 model_vert_fov=70,
                 model_horiz_fov=70):
        super().__init__()

        self.output_size = output_size
        self.mincam_size = mincam_size
        self.img_size = img_size

        task_network_input = self.mincam_size
        self.task_network = TaskNetwork(
            task_network_input, output_size, hidden_layer_size,
            hidden_layer_count)

        # Field of view of the model masks, unrelated to the prototype
        self.model_vert_fov = model_vert_fov
        self.model_horiz_fov = model_horiz_fov

        self.simulate_pd_area_blur = simulate_pd_area_blur
        self.mask_blur_kernel_sigma = mask_blur_kernel_sigma
        self._create_mask_blur_kernel()

        self.realistic_sensor = realistic_sensor
        self.simulate_directivity = simulate_directivity

        self.register_buffer("sensor_gain",
                             torch.tensor(sensor_gain, dtype=torch.float32))
        self.register_buffer("sensor_n_bits",
                             torch.tensor(sensor_n_bits, dtype=torch.int64))
        self.register_buffer("sensor_saturation_val",
                             torch.tensor(sensor_saturation_val,
                                          dtype=torch.float32))
        self.register_buffer("read_noise_std",
                             torch.tensor(read_noise_std, dtype=torch.float32))
        self.register_buffer("mask_min_value",
                             torch.tensor(mask_min_value, dtype=torch.float32))
        self.register_buffer("mask_max_value",
                             torch.tensor(mask_max_value, dtype=torch.float32))

        if simulate_directivity:
            self.register_buffer("diode_directivity",
                                 self._load_diode_directivity())
        else:
            self.directivity_fn = None

        self._mincam_init_masks(mask_init_method)

        self.sensor_stats = SensorStatistics()
        self.sensor_stats.disable()

    def _create_mask_blur_kernel(self):
        # Assuming image size for blur kernel size
        assert self.img_size == (128, 128)

        mask_kernel = None

        if self.simulate_pd_area_blur:
            """
            # Compute photodiode area as a function of mask height
            f = 8e-3 # mask height
            pd_size_m = .88e-3 # photodiode side lengrth
            mask_width_m = 2*f*np.tan(np.deg2rad(self.model_horiz_fov/2))
            mask_height_m = 2*f*np.tan(np.deg2rad(self.model_vert_fov/2))
            pd_kernel_h = self.img_size[0] * pd_size_m / mask_height_m
            pd_kernel_w = self.img_size[1] * pd_size_m / mask_width_m
            """
            # Best case when the 128x128 px image matches the pixel's field of 
            # view
            pd_kernel_h = 7
            pd_kernel_w = 7

            # Box filter from photodiode's active area
            mask_kernel = np.ones((int(np.round(pd_kernel_h)), 
                                   int(np.round(pd_kernel_w))), 
                                  dtype=np.float32)
            mask_kernel /= mask_kernel.sum()

        if self.mask_blur_kernel_sigma is not None and \
            self.mask_blur_kernel_sigma > 0:
            # Gaussian smoothing for robustness to misalignment
            M = self.mask_blur_kernel_sigma * 4 + 1
            k_gaussian = 1 / np.sqrt(2 * np.pi * self.mask_blur_kernel_sigma**2) * \
                np.exp(
                    - np.arange(-np.floor(M/2), np.floor(M/2)+1)**2 / \
                        (2 * self.mask_blur_kernel_sigma**2))
            k_gaussian /= k_gaussian.sum()
            k_gaussian = k_gaussian[:,None] * k_gaussian[None,:]

            if mask_kernel is not None:
                mask_kernel = scipy.signal.convolve2d(
                    mask_kernel, k_gaussian, mode='full')
            else:
                mask_kernel = k_gaussian

        if mask_kernel is not None:
            mask_kernel = torch.from_numpy(mask_kernel).to(torch.float32)[None,None,:,:]

        self.register_buffer("mask_blur_kernel", mask_kernel)

    def _load_diode_directivity(self):
        p = scipy.io.loadmat(
            str(constants.DIODE_DIRECTIVITY_PATH))["p"].ravel()

        r_start = np.tan(np.deg2rad(self.model_vert_fov) / 2)
        c_start = np.tan(np.deg2rad(self.model_horiz_fov) / 2)
        r, c = np.meshgrid(
            np.linspace(-r_start, r_start, self.img_size[0]),
            np.linspace(-c_start, c_start, self.img_size[1]),
            indexing="ij")

        radius = np.sqrt(r**2 + c**2)
        theta = np.rad2deg(np.arctan(radius))

        v = np.polyval(p, theta)
        v /= v.max()

        return torch.from_numpy(v).to(torch.get_default_dtype())

    @torch.no_grad()
    def visualize_mask(self, num_masks=4):
        """
        Return the first N masks for visualization
        """
        if num_masks == -1:
            num_masks = self.masks.shape[0]

        masks_vis = self._mask_param_transform(
            self.masks[:num_masks,:]).reshape(
                num_masks, *self.img_size).detach()

        masks_vis = torch.tile(masks_vis[:,None,:,:], (1, 3, 1, 1))

        return masks_vis # Nx3xHxW

    def _mask_param_transform(self, x: torch.Tensor):
        y = torch.sigmoid(x) * \
            (self.mask_max_value - self.mask_min_value) + self.mask_min_value

        # Convolve with mask blur kernel
        if self.mask_blur_kernel is not None:
            y = y.reshape(y.shape[0], 1, *self.img_size) # Nx1xHxW
            y = F.conv2d(y, self.mask_blur_kernel, padding="same")
            y = y.reshape(y.shape[0], -1)

        return y

    def _mask_param_inv_transform(self, y: torch.Tensor):
        x = torch.logit((y - self.mask_min_value) / \
            (self.mask_max_value - self.mask_min_value))

        return x

    def _mincam_init_masks(self, mask_init_method):
        """
        Initialize the masks using uniform noise
        """
        mask_size = (self.mincam_size, self.img_size[0] * self.img_size[1])

        if mask_init_method == "random":
            # Initialize M(x,y) ~ Uniform(mid_point - half_range,
            #                             mid_point + half_range)
            mid_point = 0.1
            half_range = mid_point / 5
            masks = self._mask_param_inv_transform(
                torch.rand(mask_size) * 2 * half_range + mid_point - half_range
            )

        else:
            module_logger.error("Unsupported mask initialization method: %s" %
                                mask_init_method)
            sys.exit(1)

        self.masks = nn.Parameter(masks)

    def _forward_mask(self, x):
        M = self._mask_param_transform(self.masks)
        x = F.linear(x.reshape(x.shape[0], -1), M)
        return x

    def _apply_sensor_model(self, x):
        # Scale to the sensor's dynamic range
        x = x * self.sensor_gain

        if self.realistic_sensor:
            # Saturation
            x = torch.where(x < self.sensor_saturation_val,
                            x,
                            0.01 * (x - self.sensor_saturation_val) +
                            self.sensor_saturation_val)

            # Read Noise
            x = x + torch.randn_like(x) * self.read_noise_std

            # Quantization noise
            x = x + torch.rand_like(x) * \
                (self.sensor_saturation_val / 2**self.sensor_n_bits)

        return x


    def _sensor_fn(self, x):
        # Apply sensor directivity (vignetting)
        if self.simulate_directivity:
            x = x * self.diode_directivity[None,None,:,:]

        # Projection with masks
        x = self._forward_mask(x)

        # Sensor model
        x = self._apply_sensor_model(x)

        return x

    def forward(self, imgs):
        x = self._sensor_fn(imgs)
        self.sensor_stats.update(x)
        task_network_output = self.task_network(x)

        return task_network_output

    def extra_repr(self) -> str:
        return "Masks: %dx%dpx x%d\nSensor gain: %.5f\nSensor Bits: %d\nRead Noise Standard Deviation: %.4f\nRealistic Sensor Enable: %r" % \
            (*self.img_size, self.mincam_size, self.sensor_gain,
             self.sensor_n_bits, self.read_noise_std, self.realistic_sensor)


class TaskNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer_size,
                 hidden_layer_count):
        super().__init__()

        relu_neg_slope = 0.1

        layers = []

        for i in range(1,hidden_layer_count+1):
            if i == 1:
                layers.append(("linear%d" % i,
                               nn.Linear(num_inputs, hidden_layer_size)))
            else:
                layers.append(("linear%d" % i,
                               nn.Linear(hidden_layer_size, hidden_layer_size)))
            layers.append(("relu%d" % i, nn.LeakyReLU(relu_neg_slope)))

        layers.append(("linear%d" % (hidden_layer_count+1),
                       nn.Linear(hidden_layer_size, num_outputs)))

        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

