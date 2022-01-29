import numpy as np
import torch
import torch.nn as nn
import math


# def invert_ycoord_to_idx(y_cos, y_sin, period):
#     ratio_cos = math.acos(y_cos) / np.pi
#     ratio_sin = math.asin(y_sin) / np.pi
#     if ratio_cos > 0 and ratio_sin > 0:

class CoordHandler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.coord_num_dir = self.config.train_params.coord_num_dir

        ss_unfold_size = self.config.train_params.ss_n_layers * self.config.train_params.ss_unfold_radius
        self.ts_spatial_size = config.train_params.ts_input_size
        self.ss_spatial_size = config.train_params.ts_input_size + 2 * ss_unfold_size

        self.const_grid_size_x = \
            self.ss_spatial_size + self.config.train_params.coord_vert_sample_size
        self.const_grid_size_y = \
            int(round(self.ss_spatial_size / self.config.train_params.coord_hori_occupy_ratio))
        if self.coord_num_dir == 4:
            self.const_grid_size_x = self.const_grid_size_y

        self.const_grid = self._creat_coord_grid(
            height=self.const_grid_size_x, 
            width=self.const_grid_size_y,
        ).contiguous().cuda()

        # Uniform perturbation that creates continuous coordinates
        if config.train_params.coord_continuous:
            if self.coord_num_dir == 1:
                self.coord_perturb_range = (
                    abs(self.const_grid[0, 0, 0].item() - self.const_grid[0, 1, 0].item()) / 2,
                )
            elif self.coord_num_dir == 2:
                self.coord_perturb_range = (
                    abs(self.const_grid[0, 0, 0].item() - self.const_grid[0, 0, 1].item()) / 2,
                    abs(self.const_grid[1, 0, 0].item() - self.const_grid[1, 0, 1].item()) / 2,
                )
            elif self.coord_num_dir == 4:
                self.coord_perturb_range = (
                    abs(self.const_grid[0, 0, 0].item() - self.const_grid[0, 1, 0].item()) / 2,
                    abs(self.const_grid[1, 0, 0].item() - self.const_grid[1, 1, 0].item()) / 2,
                    abs(self.const_grid[2, 0, 0].item() - self.const_grid[2, 0, 1].item()) / 2,
                    abs(self.const_grid[3, 0, 0].item() - self.const_grid[3, 0, 1].item()) / 2,
                )
            elif self.coord_num_dir in {3, 21}:
                self.coord_perturb_range = (
                    abs(self.const_grid[0, 0, 0].item() - self.const_grid[0, 1, 0].item()) / 2,
                    abs(self.const_grid[1, 0, 0].item() - self.const_grid[1, 0, 1].item()) / 2,
                    abs(self.const_grid[2, 0, 0].item() - self.const_grid[2, 0, 1].item()) / 2,
                )
            else:
                raise ValueError()
            for v in self.coord_perturb_range:
                assert v > 0, " [Sanity check failed] perturb should always >0 while enabled, but got {}".format(self.coord_perturb_range)
        else:
            print(" [!] Discrete coords should be depricated!")
            self.coord_perturb_range = (0,) * self.coord_num_dir

    def sample_xy_st_index(self, batch_size):
        if self.config.train_params.coord_num_dir == 4:
            x_rnd_range = self.const_grid_size_x
        else:
            x_rnd_range = self.config.train_params.coord_vert_sample_size #self.const_grid_size_x - x_size

        if x_rnd_range == 0:
            x_st = np.zeros(batch_size).astype(np.uint8)
        else:
            x_st = np.random.randint(0, x_rnd_range, batch_size)
        y_st = np.random.randint(0, self.const_grid_size_y, batch_size)
        return x_st, y_st

    def add_rnd_perturb(self, mesh_indices):
        # mesh_indices: [B, C, H, W]
        B, C, H, W = mesh_indices.shape
        for dim,perturb_range in enumerate(self.coord_perturb_range):
            perturb_v = (torch.rand(B) * perturb_range * 2) - perturb_range
            perturb_v = perturb_v.unsqueeze(-1).unsqueeze(-1)
            perturb_v = perturb_v.to(mesh_indices.device)
            mesh_indices[:, dim, :, :] += perturb_v
        return mesh_indices

    def _creat_coord_grid(self, height, width, coord_init=None,
                          auto_calibrate_pano_coord=False):

        if coord_init is None:
            coord_init = (0, 0) # Workaround

        x = torch.arange(height).type(torch.float32) + coord_init[0] 
        y = torch.arange(width).type(torch.float32)  + coord_init[1]

        # To [0, 1], but may exceed this range at testing
        x = x / (self.ts_spatial_size+self.config.train_params.coord_vert_sample_size-1)
        if not auto_calibrate_pano_coord:
            y = y / (self.const_grid_size_y-1)
        else:
            # Disregard the coordinate frequency on the horizontal axis, enforced to [-pi, +pi] 
            # at the start and the end of the y-sequence, and ss_unfold_size is auto extrapolated.
            y = y / width
            
        # Re-center the x coords
        exceeding_part = (x[-1] - 1)
        x = x - exceeding_part / 2

        x = x * 2 - 1 # [-1, 1]
        y = y * 2 - 1 # [-1, 1]

        # X-axis: 
        # 1. to [-N, N], which N is `coord_y_cut_pt`
        x = x * self.config.train_params.coord_vert_cut_pt #* x_extra_scale
        x_tiled = x.view(-1, 1).repeat(1, width)

        if self.coord_num_dir == 1:
            meshed = x_tiled.unsqueeze_(0) # [1, H, W]
        elif self.config.train_params.coord_num_dir == 2:
            y_tiled = y.view(1, -1).repeat(height, 1)
            meshed = torch.cat([
                y_tiled.unsqueeze(0), # apply cos later
                y_tiled.unsqueeze(0), # apply sin later
            ], 0) # [2, H, W]
        elif self.config.train_params.coord_num_dir == 3:
            y_tiled = y.view(1, -1).repeat(height, 1)
            meshed = torch.cat([
                x_tiled.unsqueeze_(0), 
                y_tiled.unsqueeze(0), # apply cos later
                y_tiled.unsqueeze(0), # apply sin later
            ], 0) # [3, H, W]
        elif self.config.train_params.coord_num_dir == 4:
            x_tiled = y.view(-1, 1).repeat(1, width) # Reuse y
            y_tiled = y.view(1, -1).repeat(width, 1) # make into shape (w, w)
            meshed = torch.cat([
                x_tiled.unsqueeze(0), # apply cos later
                x_tiled.unsqueeze(0), # apply sin later
                y_tiled.unsqueeze(0), # apply cos later
                y_tiled.unsqueeze(0), # apply sin later
            ], 0) # [4, H, W]
        elif self.config.train_params.coord_num_dir == 21:
            y_tiled = y.view(1, -1).repeat(height, 1)
            meshed = torch.cat(
                [x_tiled.unsqueeze_(0)] + 
                [y_tiled.unsqueeze(0) for _ in range(20)], # apply cos/sin later
            0) # [3, H, W]
        else:
            raise NotImplementedError("Unkown coord dimension {}".format(
                self.config.train_params.coord_num_dir))
        return meshed

    def invert_coord_to_idx(self, coord, is_x_dir=False, is_y_dir=False):
        assert False, "Not well-tested, use with care!"
        assert is_x_dir or is_y_dir
        if is_x_dir:
            back_proj = math.atanh(coord.clamp(-1, 1))
            denum = (self.ts_spatial_size-1)
        elif is_y_dir:
            assert False, "This functions is malfunctioned, it only considers one quadrant."
            try:
                back_proj = math.asin(coord) / math.pi
                denum = (self.ts_spatial_size-1)
            except Exception as e:
                print("Coord:", coord)
                raise e

        back_proj = (back_proj + 1) / 2 # to [0, 1]
        back_proj = back_proj * denum
        return round(back_proj)

    def convert_idx_to_input_coords(self, mesh):
        assert mesh.ndim == 4, "Did not consider other cases, got mesh shape {}".format(mesh.shape)
        if self.config.train_params.coord_num_dir == 1: 
            mesh[:, 0, :] = torch.tanh(mesh[:, 0, :])
        elif self.config.train_params.coord_num_dir == 2:
            mesh[:, 0, :] = torch.cos(mesh[:, 0, :] * np.pi)
            mesh[:, 1, :] = torch.sin(mesh[:, 1, :] * np.pi)
        elif self.config.train_params.coord_num_dir == 3:
            mesh[:, 0, :] = torch.tanh(mesh[:, 0, :])
            mesh[:, 1, :] = torch.cos(mesh[:, 1, :] * np.pi)
            mesh[:, 2, :] = torch.sin(mesh[:, 2, :] * np.pi)
        elif self.config.train_params.coord_num_dir == 4:
            mesh[:, 0, :] = torch.cos(mesh[:, 0, :] * np.pi)
            mesh[:, 1, :] = torch.sin(mesh[:, 1, :] * np.pi)
            mesh[:, 2, :] = torch.cos(mesh[:, 2, :] * np.pi)
            mesh[:, 3, :] = torch.cos(mesh[:, 3, :] * np.pi)
        elif self.config.train_params.coord_num_dir == 21:
            mesh[:, 0, :] = torch.tanh(mesh[:, 0, :])
            for i in range(10):
                mesh[:, i*2+1, :] = torch.cos(mesh[:, i*2+1, :] * np.pi * 2**i)
                mesh[:, i*2+2, :] = torch.sin(mesh[:, i*2+2, :] * np.pi * 2**i)
        return mesh

    def _safe_select(self, meta_grid, x_st, y_st, x_size, y_size):
        grids = []
        for xx, yy in zip(x_st, y_st):
            if self.coord_num_dir == 4:
                # Need wrapping
                if yy > self.const_grid_size_y - y_size:
                    if xx > self.const_grid_size_x - x_size:
                        remainings_x = xx + x_size - self.const_grid_size_x
                        remainings_y = yy + y_size - self.const_grid_size_y
                        grid = torch.cat([
                            torch.cat([
                                meta_grid[:, xx:, yy:],
                                meta_grid[:, :remainings_x, yy:],
                            ], 1),
                            torch.cat([
                                meta_grid[:, xx:, :remainings_y],
                                meta_grid[:, :remainings_x, :remainings_y],
                            ], 1)
                        ], 2)
                    else:
                        remainings = yy + y_size - self.const_grid_size_y
                        grid = torch.cat([
                            meta_grid[:, xx:xx+x_size, yy:],
                            meta_grid[:, xx:xx+x_size, :remainings],
                        ], 2)
                else:
                    if xx > self.const_grid_size_x - x_size:
                        remainings = xx + x_size - self.const_grid_size_x
                        grid = torch.cat([
                            meta_grid[:, xx:, yy:yy+y_size],
                            meta_grid[:, :remainings, yy:yy+y_size],
                        ], 1)
                    else:
                        grid = meta_grid[:, xx:xx+x_size, yy:yy+y_size]
            else:
                # Need wrapping
                if yy > self.const_grid_size_y - y_size:
                    remainings = yy + y_size - self.const_grid_size_y
                    grid = torch.cat([
                        meta_grid[:, xx:xx+x_size, yy:],
                        meta_grid[:, xx:xx+x_size, :remainings],
                    ], 2)
                # Naive selection
                else:
                    grid = meta_grid[:, xx:xx+x_size, yy:yy+y_size]

            grids.append(grid)

        grids = torch.stack(grids).contiguous().clone().detach()
        return grids

    def create_coord_sequence(self, abs_disp):
        x_max = np.abs(abs_disp[:, 0]).max()
        y_max = np.abs(abs_disp[:, 1]).max()
        meta_grid = self._creat_coord_grid(
            height=x_max*2+1, 
            width=y_max*2+1)
        center = (x_max, y_max) # center of a 2x larger grid
        ret = []
        for (x,y) in abs_disp:
            coord = meta_grid[:, center[0]+x, center[1]+y]
            ret.append(coord)
        return torch.stack(ret)

    def sample_coord_grid(self, spatial_latent, coord_init=None, is_training=True, is_fid_eval=False,
                          override_coords=None, return_ac_coords=False, 
                          auto_calibrate_pano_coord=False, 
                          specific_shape=None, device=None, batch_size=None):
        if specific_shape is None:
            # Note:
            # spatial_latent shape can be slightly different from different configs, use runtime size is better
            bs, _, x_size, y_size = spatial_latent.shape # (B, C, H, W)
            device = spatial_latent.device
        else:
            assert device is not None, "Device must be specified."
            assert batch_size is not None, "Batch size must be specified."
            if isinstance(specific_shape, tuple) or isinstance(specific_shape, list):
                x_size, y_size = specific_shape
                assert len(specific_shape) == 2, "Got {}".format(specific_shape)
            else:
                x_size = specific_shape
                y_size = specific_shape
            bs = batch_size

        needs_extrap = (x_size > self.ss_spatial_size) or (y_size > self.ss_spatial_size)
        fid_use_training = is_fid_eval and (not needs_extrap)

        ac_coords = None
        if is_training or fid_use_training:
            assert coord_init is None, "Not considered"
            assert auto_calibrate_pano_coord is False, \
                "This argument is specifically designed for panorama generation at testing"
            if override_coords is None:
                if needs_extrap: 
                    # Select random disp first, then extrapolate coordinates base on the disp
                    x_disp, y_disp = self.sample_xy_st_index(bs)
                    # In some special cases (very few) that we need to sample coords larger than
                    # training coords grid, we have to create new at runtime...
                    grid_indices = torch.stack([self._creat_coord_grid(
                        height=x_size, width=y_size, coord_init=(xx, yy))
                            for xx, yy in zip(x_disp, y_disp)])
                    grid_indices = self.add_rnd_perturb(grid_indices)
                    coords = self.convert_idx_to_input_coords(grid_indices)
                    x_st, y_st = x_disp, y_disp
                else:
                    x_st, y_st = self.sample_xy_st_index(bs)
                    grid_indices = self._safe_select(self.const_grid, x_st, y_st, x_size, y_size)
                    grid_indices = self.add_rnd_perturb(grid_indices)
                    coords = self.convert_idx_to_input_coords(grid_indices)
                coords = coords.to(device)

                if return_ac_coords:
                    # x_denom = self.const_grid_size_x - x_size - 1
                    x_denom = self.config.train_params.coord_vert_sample_size - 1
                    norm_x_st = (x_st / x_denom) * 2 - 1 # [-1, 1]
                    norm_y_st = (y_st / (self.const_grid_size_y-1)) * 2  - 1 # [-1, 1]
                    ac_coords_x = norm_x_st # np.tanh(norm_x_st) # meaningless to do this projection
                    if self.config.train_params.coord_num_dir == 1:
                        ac_coords = torch.from_numpy(ac_coords_x).unsqueeze(1).float().to(device)
                    elif self.config.train_params.coord_num_dir == 2:
                        ac_coords_a = np.cos(norm_y_st * np.pi)
                        ac_coords_b = np.sin(norm_y_st * np.pi)
                        ac_coords = np.stack([ac_coords_a, ac_coords_b], 1)
                        ac_coords = torch.from_numpy(ac_coords).float().to(device)
                    elif self.config.train_params.coord_num_dir == 4:
                        norm_x_st = (x_st / (self.const_grid_size_y-1)) * 2  - 1 # [-1, 1]
                        ac_coords = np.stack([
                            np.cos(norm_x_st * np.pi),
                            np.sin(norm_x_st * np.pi),
                            np.cos(norm_y_st * np.pi),
                            np.sin(norm_y_st * np.pi),
                        ], 1)
                        ac_coords = torch.from_numpy(ac_coords).float().to(device)
                    elif self.config.train_params.coord_num_dir in {3, 21}:
                        ac_coords_a = np.cos(norm_y_st * np.pi)
                        ac_coords_b = np.sin(norm_y_st * np.pi)
                        ac_coords = np.stack([ac_coords_x, ac_coords_a, ac_coords_b], 1)
                        ac_coords = torch.from_numpy(ac_coords).float().to(device)
                    else:
                        raise ValueError("Unknown coord_num_dir {}".format(self.config.train_params.coord_num_dir))
                    if self.training:
                        assert ac_coords.min()>-1.1, "Got unexpected ac_coords min {} < -1".format(ac_coords.min())
                        assert ac_coords.max()<1.1, "Got unexpected ac_coords max {} > 1".format(ac_coords.max())
            else:
                # Scale invariant loss requires this "feature" ...
                # raise ValueError("Unexpected sending override_coords during training!")
                coords = override_coords
                ac_coords = None
        else:
            # # Testing
            # Assumes the center of z_spatial and coord grid are aligned, then accordingly extrapolate the coord
            # Note: supports extrapolate on x-axis.
            if override_coords is None:
                grid_indices = self._creat_coord_grid(
                    height=x_size, width=y_size, coord_init=coord_init, 
                    auto_calibrate_pano_coord=auto_calibrate_pano_coord)
                grid_indices = grid_indices.unsqueeze(0).repeat(bs, 1, 1, 1)
                # grid_indices = self.add_rnd_perturb(grid_indices) # Probably don't want this, which may break consistency at testing
                coords = self.convert_idx_to_input_coords(grid_indices)
                coords = coords.to(device)
            else:
                coords = override_coords
            ac_coords = None

        if return_ac_coords:
            return coords, ac_coords
        else:
            return coords

    def update_coords_by_mean(self, coords, new_mean):
        # Coords may not be a valid one after inversion update, 
        # here we calibrates a new coords that matches the mean of the dirty region.

        for i in range(coords.shape[0]):
            new_init = (
                self.invert_coord_to_idx(new_mean[i, 0], is_x_dir=True), 
                self.invert_coord_to_idx(new_mean[i, 1], is_y_dir=True),
            )
            new_coords = self.sample_coord_grid(
                coords[i:i+1], 
                coord_init=new_init, 
                is_training=False)
            coords[i:i+1].data = new_coords[i:i+1].data

       


