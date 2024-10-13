import numpy as np
import cv2
from skimage.transform import resize, rotate
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from abc import ABC
import scipy.signal


class Noise(ABC):
    def __init__(self, name = None):
        self.name = name
        
    def add_noise(self, image):
        raise NotImplementedError


class ConstantNoise(Noise):
    def __init__(self):
        super().__init__("constant")
    
    def add_noise(self, image):
        return image
   
 
class UniformNoise(Noise):
    def __init__(self, min = 0.9, max = 1):
        super().__init__("uniform")
        self.min = min
        self.max = max
    
    def add_noise(self, image):
        return image * np.random.uniform(self.min, self.max, size=image.shape)


class SpatialNoise(Noise):
    def __init__(self, correlation_scale = 10, range = 0.7):
        super().__init__("spatial" + str(correlation_scale))
        self.correlation_scale = correlation_scale
        self.range = range
    
    def add_noise(self, image):
        img = image.copy()
        for i in range(img.shape[2]):
            img[:, :, i] = self.add_channel_noise(img[:, :, i])
        return img
        
    
    def add_channel_noise(self, image):
        correlation_scale = self.correlation_scale
        x = np.arange(-correlation_scale, correlation_scale)
        y = 2 * np.arange(-correlation_scale, correlation_scale)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X*X + Y*Y)
        filter_kernel = np.exp(-dist**2/(2*correlation_scale))

        n = 100
        noise = np.random.randn(n, n)
        noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        noise -= noise.min()
        noise /= noise.max()
        noise *= self.range
        noise += 1 - self.range
        return image * noise


class ImageTemplate():
    def __init__(self, template = None, rot_inc = 1, repeats = 10, noise = ConstantNoise(), verbose = True):
        self.template, self.template_name = self.get_template(template)
        self.rot_inc = range(0, 360, int(rot_inc))
        self.repeats = repeats
        self.noise = noise
        self.verbose = verbose
        
    def get_template(self, template):
        match template:
            case "circle":
                nucl_mask = np.zeros((10001, 10001))
                cv2.circle(nucl_mask, (4300, 5000), 1000, 1, -1)
                cell_mask = np.zeros((10001, 10001))
                cv2.circle(cell_mask, (5000, 5000), 2150, 1, -1)
                return np.stack([np.zeros_like(cell_mask), cell_mask, nucl_mask], axis=-1), template
            case "ellipse":
                nucl_mask = np.zeros((10001, 10001))
                cv2.circle(nucl_mask, (4000, 5000), 1000, 1, -1)
                cell_mask = np.zeros((10001, 10001))
                cv2.ellipse(cell_mask, (5000, 5000), (3000, 1500), 0, 0, 360, 1, -1)
                return np.stack([np.zeros_like(cell_mask), cell_mask, nucl_mask], axis=-1), template
            case "ellipse-gradient":
                nucl_mask = np.zeros((10001, 10001))
                cv2.circle(nucl_mask, (4000, 5000), 1000, 1, -1)
                cell_mask = np.zeros((10001, 10001))
                cv2.ellipse(cell_mask, (5000, 5000), (3000, 1500), 0, 0, 360, 1, -1)
                for i in range(10001):
                    cell_mask[:, i] *= (0.8*i/10001) + 0.2
                return np.stack([np.zeros_like(cell_mask), cell_mask, nucl_mask], axis=-1), template
            case _:
                raise ValueError(f"Template {template} not found")
            
    def gen_images(self, path = None):
        if path is None:
            raise ValueError("Path not specified")
        if not os.path.exists(path):
            os.makedirs(path)
        pbar = tqdm(self.rot_inc, disable = not self.verbose)
        for angle in pbar:
            rotated = rotate(self.template, angle)
            scaled = resize(rotated, (100, 100))
            for i in range(self.repeats):
                noise_img = self.noise.add_noise(scaled)
                plt.imsave(f"{path}/{self.template_name}_{self.noise.name}noise_rotated{angle}_repeat{i+1}.png", noise_img)
  
  
class ImageTemplateSet():
    def __init__(self, templates = None, rot_inc = 1, repeats = 10, noise = ConstantNoise(), seed = None, verbose = True):
        if templates == "all":
            templates = ["circle", "ellipse", "ellipse-gradient"]
        else:
            templates = [templates]
        self.noise = noise
        self.seed = seed
        self.verbose = verbose
        self.templates = [ImageTemplate(template, rot_inc, repeats, noise, verbose) for template in templates]
        
    def gen_images(self, path = None):
        if self.seed is not None:
            np.random.seed(self.seed)
        if path is None:
            raise ValueError("Path not specified")
        if not os.path.exists(path):
            os.makedirs(path)
        for template in self.templates:
            if self.verbose:
                print(f"\nGenerating images for {template.template_name} template")
            template.gen_images(path)
        if self.verbose:
            print(f"\nAll images for {self.noise.name} noise generated and stored in {path}\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog = "image_generator.py",
        description = 'Generate example images for demonstrating non-biological dependent structure in morphological measurements'
    )
    parser.add_argument(
        "-d", "--directory",
        default = "images",
        help = "The directory to save the generated images to. Default: images",
        type = str
    )
    parser.add_argument(
        "-t", "--templates",
        choices = ["all", "circle", "ellipse", "ellipse-gradient"],
        default = "all",
        help = "The template to use for generating images, specifying all will generate images for all of the templates. Default: all",
        type = str
    )
    parser.add_argument(
        "-i", "--rot_inc",
        default = 360,
        help = "The number of degrees to rotate the template by for each image. Default: 360",
        type = int
    )
    parser.add_argument(
        "-n", "--noise",
        choices = ["all", "constant", "uniform", "spatial", "spatial10", "spatial20", "spatial30", "spatial40", "spatial50"],
        default = "all",
        help = "The type of noise to add to the images. Default: all",
        type = str
    )
    parser.add_argument(
        "-r", "--repeats",
        default = 10,
        help = "The number of times to repeat each rotation. Default: 10",
        type = int
    )
    parser.add_argument(
        "-s", "--seed",
        default = None,
        help = "The seed to use for the random number generator. Default: None",
        type = int
    )
    parser.add_argument(
        "-v", "--verbose",
        action = "store_true"
    )
    args = parser.parse_args()
    
    noise = "all"
    if noise == "all":
        noise = ["constant", "uniform", "spatial10", "spatial20", "spatial30", "spatial40", "spatial50"]
    elif noise == "spatial":
        noise = ["spatial10", "spatial20", "spatial30", "spatial40", "spatial50"]
    else:
        noise = [noise]
        
    for n in noise:
        match n:
            case "constant":
                noise_obj = ConstantNoise()
            case "uniform":
                noise_obj = UniformNoise()
            case "spatial10":
                noise_obj = SpatialNoise(correlation_scale = 10)
            case "spatial20":
                noise_obj = SpatialNoise(correlation_scale = 20)
            case "spatial30":
                noise_obj = SpatialNoise(correlation_scale = 30)
            case "spatial40":
                noise_obj = SpatialNoise(correlation_scale = 40)
            case "spatial50":
                noise_obj = SpatialNoise(correlation_scale = 50)
            case _:
                raise ValueError(f"Noise {n} not found")
        if noise_obj.name == "constant":
            repeats = 1
        else:
            repeats = args.repeats
        set = ImageTemplateSet(
            templates = args.templates,
            rot_inc = args.rot_inc,
            repeats = repeats,
            noise = noise_obj,
            verbose = args.verbose
        )
        set.gen_images(args.directory)
