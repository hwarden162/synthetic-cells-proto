from tqdm import tqdm
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
from abc import ABC
import numpy as np
import cv2
import os
import scipy
from argparse import ArgumentParser


# ========================= Image Templates =========================

class ImageTemplate():
    
    def get_template(self) -> np.ndarray:
        raise NotImplementedError


class CircleTemplate(ImageTemplate):
    
    def get_template(self) -> np.ndarray:
        nucl_mask = np.zeros((1001, 1001))
        cv2.circle(nucl_mask, (430, 500), 100, 1, -1)
        cell_mask = np.zeros((1001, 1001))
        cv2.circle(cell_mask, (500, 500), 215, 1, -1)
        return np.stack([np.zeros_like(cell_mask), cell_mask, nucl_mask], axis=-1)


class EllipseTemplate(ImageTemplate):
        
    def get_template(self) -> np.ndarray:
        nucl_mask = np.zeros((1001, 1001))
        cv2.circle(nucl_mask, (400, 500), 100, 1, -1)
        cell_mask = np.zeros((1001, 1001))
        cv2.ellipse(cell_mask, (500, 500), (300, 150), 0, 0, 360, 1, -1)
        return np.stack([np.zeros_like(cell_mask), cell_mask, nucl_mask], axis=-1)


# ========================= Stain Structures =========================

class StainStructure(ABC):
    
    def get_stain_structure(self) -> np.ndarray:
        raise NotImplementedError


class ConstantStructure(StainStructure):
    
    def apply_stain_structure(self, image) -> np.ndarray:
        return image


class GradientStructure(StainStructure):
    
    def apply_stain_structure(self, image) -> np.ndarray:
        img = image.copy()
        ncol = img.shape[1]
        for i in range(ncol):
            img[:, i, 1] = (img[:, i, 1] * (i / ncol)) * 0.4 + 0.6
        return img


class RadialStructure(StainStructure):
    
    def __init__(self, global_divisor = 50, x_scaler = 1, y_scaler = 1) -> None:
        super().__init__()
        self.global_divisor = global_divisor
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
    
    def apply_stain_structure(self, image) -> np.ndarray:
        img = image.copy()
        x = self.x_scaler * np.arange(-500, 501, 1)
        y = self.y_scaler * np.arange(-500, 501, 1)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X*X + Y*Y)/self.global_divisor
        periodoc_dist = np.cos(dist * np.pi)
        periodoc_dist -= periodoc_dist.min()
        periodoc_dist /= periodoc_dist.max()
        periodoc_dist *= 0.4
        periodoc_dist += 0.6
        struct = np.stack([np.ones_like(periodoc_dist), periodoc_dist, np.ones_like(periodoc_dist)], axis=-1)
        return img * struct

    
class SpatialStructure(StainStructure):
    
    def __init__(self, correlation_scale = 30, x_scale = 1, y_scale = 1) -> None:
        super().__init__()
        self.correlation_scale = correlation_scale
        self.x_scale = x_scale
        self.y_scale = y_scale
    
    def apply_stain_structure(self, image) -> np.ndarray:
        img = image.copy()
        correlation_scale = self.correlation_scale
        x = self.x_scale * np.arange(-correlation_scale, correlation_scale + 1)
        y = self.y_scale * np.arange(-correlation_scale, correlation_scale + 1)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X*X + Y*Y)
        filter_kernel = np.exp(-dist**2/(2*correlation_scale))

        n = 100
        noise1 = np.random.randn(n, n)
        noise1 = scipy.signal.fftconvolve(noise1, filter_kernel, mode='same')
        noise1 -= noise1.min()
        noise1 /= noise1.max()
        noise1 *= 0.4
        noise1 += 0.6
        noise = np.stack([np.ones_like(noise1), noise1, np.ones_like(noise1)], axis=-1)
        noise = resize(noise, (1001, 1001, 3))
        return img * noise


# ========================= Noises =========================

class Noise(ABC):
        
    def apply_noise(self) -> np.ndarray:
        raise NotImplementedError


class ConstantNoise(Noise):
    
    def apply_noise(self, image) -> np.ndarray:
        return image


class UniformNoise(Noise):
        
    def apply_noise(self, image) -> np.ndarray:
        return image * np.random.uniform(0.9, 1, image.shape)


# ========================= Image Generator =========================

class ImageGenerator():
    
    def __init__(self, templates = "all", stain_structs = "all", major_reps = 3, rot_inc = 30, trans_inc = 10, trans_lim = 10, noises = "all", minor_reps = 3, seed = None) -> None:
        if templates == "all":
            self.templates = ["circle", "ellipse"]
        else:
            self.templates = [templates]
        if stain_structs == "all":
            self.stain_structs = ["constant", "gradient", "radial01", "radial02", "radial10", "radial11", "radial12", "radial20", "radial21", "radial22", "spatial11" , "spatial12", "spatial13", "spatial21", "spatial22", "spatial23", "spatial31", "spatial32", "spatial33"]
        elif stain_structs == "radial":
            self.stain_structs = ["radial01", "radial02", "radial10", "radial11", "radial12", "radial20", "radial21", "radial22"]
        elif stain_structs == "spatial":
            self.stain_structs = ["spatial11" , "spatial12", "spatial13", "spatial21", "spatial22", "spatial23", "spatial31", "spatial32", "spatial33"]
        else:
            self.stain_structs = [stain_structs]
        self.major_reps = major_reps
        self.rot_incs = range(0, 360, int(rot_inc))
        self.trans_incs = range(-trans_lim, trans_lim + 1, int(trans_inc))
        self.trans_lim = trans_lim
        if noises == "all":
            self.noises = ["constant", "uniform"]
        else:
            self.noises = [noises]
        self.minor_reps = minor_reps
        self.seed = seed
        
    def print_params(self) -> None:
        print("Image Generator Parameters:")
        print(f"\tTemplates: {self.templates}")
        print(f"\tStain Structures: {self.stain_structs}")
        print(f"\tMajor Repetitions: {self.major_reps}")
        print(f"\tRotation Increments: {self.rot_incs}")
        print(f"\tTranslational Increments: {self.trans_incs}")
        print(f"\tNoises: {self.noises}")
        print(f"\tMinor Repetitions: {self.minor_reps}")
        if self.seed is None:
            print(f"\tSeed: None")
        else:
            print(f"\tSeed: {self.seed}")
        
    def get_template(self, name) -> ImageTemplate():
        match name:
            case "circle": return CircleTemplate()
            case "ellipse": return EllipseTemplate()
    
    def get_stain_struct(self, name) -> StainStructure():
        match name:
            case "constant": return ConstantStructure()
            case "gradient": return GradientStructure()
            case "radial01": return RadialStructure(x_scaler = 0, y_scaler = 1)
            case "radial02": return RadialStructure(x_scaler = 0, y_scaler = 2)
            case "radial10": return RadialStructure(x_scaler = 1, y_scaler = 0)
            case "radial11": return RadialStructure(x_scaler = 1, y_scaler = 1)
            case "radial12": return RadialStructure(x_scaler = 1, y_scaler = 2)
            case "radial20": return RadialStructure(x_scaler = 2, y_scaler = 0)
            case "radial21": return RadialStructure(x_scaler = 2, y_scaler = 1)
            case "radial22": return RadialStructure(x_scaler = 2, y_scaler = 2)
            case "spatial11": return SpatialStructure(x_scale = 1, y_scale = 1)
            case "spatial12": return SpatialStructure(x_scale = 1, y_scale = 2)
            case "spatial13": return SpatialStructure(x_scale = 1, y_scale = 3)
            case "spatial21": return SpatialStructure(x_scale = 2, y_scale = 1)
            case "spatial22": return SpatialStructure(x_scale = 2, y_scale = 2)
            case "spatial23": return SpatialStructure(x_scale = 2, y_scale = 3)
            case "spatial31": return SpatialStructure(x_scale = 3, y_scale = 1)
            case "spatial32": return SpatialStructure(x_scale = 3, y_scale = 2)
            case "spatial33": return SpatialStructure(x_scale = 3, y_scale = 3)
    
    def get_noise(self, name) -> Noise():
        match name:
            case "constant": return ConstantNoise()
            case "uniform": return UniformNoise()
    
    def generate_file_name(self, template, stain_struct, major_rep, rot_inc, trans_inc_x, trans_inc_y, noise, minor_rep) -> str:
        if trans_inc_x >= 0:
            xpos = "pos"
        else:
            xpos = "neg"
            trans_inc_x *= -1
        if trans_inc_y >= 0:
            ypos = "pos"
        else:
            ypos = "neg"
            trans_inc_y *= -1
        return f"templ-{template}_struct-{stain_struct}_majrep-{major_rep + 1}_rotation-{rot_inc}_transx-{xpos}{str(trans_inc_x)}_transy-{ypos}{str(trans_inc_y)}_noise-{noise}_minrep-{minor_rep + 1}"
    
    def generate_images(self, path = None, verbose = True) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)
        if path is None:
            raise ValueError("Path not specified")
        if not os.path.exists(path):
            os.makedirs(path)
        sc_path = os.path.join(path, "single_cell")
        if not os.path.exists(sc_path):
            os.makedirs(sc_path)
        print("\nGenerating Images...\n")
        self.print_params()
        print("\nGenerating Single Cell Images:")
        pbar = tqdm(self.templates, disable = not verbose)
        for template in pbar:
            img = self.get_template(template).get_template()
            for stain_struct in self.stain_structs:
                if stain_struct == "constant":
                    maj_reps = [0]
                elif stain_struct == "gradient":
                    maj_reps = [0]
                elif stain_struct[0:6] == "radial":
                    maj_reps = [0]
                else:
                    maj_reps = range(self.major_reps)
                for major_rep in maj_reps:
                    img_struct = self.get_stain_struct(stain_struct).apply_stain_structure(img)
                    for rot_inc in self.rot_incs:
                        if rot_inc == 0:
                            img_rotated = img_struct
                        else:
                            img_rotated = rotate(img_struct, rot_inc)
                        img_scaled = resize(img_rotated, (100, 100))
                        for trans_inc_x in self.trans_incs:
                            for trans_inc_y in self.trans_incs:
                                shift_mat = np.array([[1, 0, trans_inc_x], [0, 1, trans_inc_y]], dtype=np.float32)
                                img_trans = cv2.warpAffine(img_scaled, shift_mat, (100, 100))
                                for noise in self.noises:
                                    if noise == "constant":
                                        minor_reps = [0]
                                    else:
                                        minor_reps = range(self.minor_reps)
                                    for minor_rep in minor_reps:
                                        img_noise = self.get_noise(noise).apply_noise(img_trans)
                                        img_name = self.generate_file_name(
                                            template, stain_struct, 
                                            major_rep, rot_inc, 
                                            trans_inc_x, trans_inc_y, 
                                            noise, minor_rep
                                        )
                                        plt.imsave(f"{sc_path}/{img_name}.png", img_noise)
        print("\nDone.\n")





if __name__ == "__main__":
    parser = ArgumentParser(
        description = "Generate synthetic single cell images"
    )
    parser.add_argument(
        "-p", "--path",
        default = None,
        help = "Directory to save generated images",
        type = str
    )
    parser.add_argument(
        "-temp", "--templates",
        choices = ["all", "circle", "ellipse"],
        default = "all",
        help = "Template shape to use to generate images",
        type = str
    )
    parser.add_argument(
        "-struct", "--stain_structs",
        choices = ["all", "radial", "spatial", "constant", "gradient", "radial01", "radial02", "radial10", "radial11", "radial12", "radial20", "radial21", "radial22", "spatial11" , "spatial12", "spatial13", "spatial21", "spatial22", "spatial23", "spatial31", "spatial32", "spatial33"],
        default = "all",
        help = "Structure of staining pattern in the generated images",
        type = str
    )
    parser.add_argument(
        "-majrep", "--major_reps",
        default = 3,
        help = "Number of repetitions of the template and stain structure",
        type = int
    )
    parser.add_argument(
        "-rotinc", "--rot_incs",
        default = 30,
        help = "Rotation increment in degrees in which the images are rotated",
        type = int
    )
    parser.add_argument(
        "-transinc", "--trans_inc",
        default = 10,
        help = "Translation increment in pixels in which the images are translated",
        type = int
    )
    parser.add_argument(
        "-translim", "--trans_lim",
        default = 10,
        help = "Maximum translation in pixels in which the images are translated",
        type = int
    )
    parser.add_argument(
        "-noise", "--noises",
        choices = ["all", "constant", "uniform"],
        default = "all",
        help = "Type of noise to add to the images",
        type = str
    )
    parser.add_argument(
        "-minrep", "--minor_reps",
        default = 3,
        help = "Number of repetitions of the noise added to the genreated images",
        type = int
    )
    parser.add_argument(
        "-seed", "--seed",
        default = None,
        help = "Seed for the random number generator",
        type = int
    )
    args = parser.parse_args()
    ig = ImageGenerator(
        templates = args.templates, 
        stain_structs = args.stain_structs, 
        major_reps = args.major_reps,
        rot_inc = args.rot_incs,
        trans_inc = args.trans_inc,
        trans_lim = args.trans_lim,
        noises = args.noises,
        minor_reps = args.minor_reps,
        seed = args.seed
    )
    ig.generate_images(path = args.path)
