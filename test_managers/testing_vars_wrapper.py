import torch
import numbers
import numpy as np
import pickle as pkl

class TestingVars():
    def __init__(self, **kwargs):
        self.meta_img = kwargs.get("meta_img")
        self.global_latent = kwargs.get("global_latent")
        self.local_latent = kwargs.get("local_latent")
        self.meta_coords = kwargs.get("meta_coords")
        self.styles = kwargs.get("styles")
        self.noises = kwargs.get("noises")
        self.device = kwargs.get("device")
        self.clone_keys = list(kwargs.keys())

        self.loaded_inv_records = kwargs.get("loaded_inv_records")
        self.loaded_inv_placements = kwargs.get("loaded_inv_placements")
        self.clone_keys += [
            "loaded_inv_records",
            "loaded_inv_placements",
        ]

    def load(self, path):
        ckpt = pkl.load(open(path, "rb"))
        for k in self.clone_keys:
            cur_var = getattr(self, k)
            if hasattr(cur_var, "device"):
                setattr(self, k, getattr(ckpt, k).to(cur_var.device))
            elif isinstance(cur_var, list) and hasattr(cur_var[0], "device"):
                try:
                    setattr(self, k, [v.to(cur_var[0].device) for v in getattr(ckpt, k)])
                except Exception as e:
                    import pdb; pdb.set_trace()
            elif cur_var is None:
                setattr(self, k, None)
            else:
                setattr(self, k, getattr(ckpt, k))

    def update_global_latent(self, latent_sampler, g_ema, mixing=False, update_index=None):
        assert mixing == False
        if isinstance(self.global_latent, list):
            new_global_latents = []
            for i,single_g in enumerate(self.global_latent):
                if update_index is not None and i != update_index:
                    global_latent = self.global_latent[i]
                else:
                    global_latent = latent_sampler.sample_global_latent(
                        single_g.shape[0],
                        mixing=mixing,
                        device=self.device)
                new_global_latents.append(global_latent)
            self.global_latent = new_global_latents

            # Fused generation extracts styles early
            new_styles = []
            for i,single_g in enumerate(self.global_latent):
                if update_index is not None and i != update_index:
                    style = self.styles[i]
                else:
                    # style = g_ema.texture_synthesizer.get_style(single_g)
                    style = g_ema(
                        call_internal_method="get_style", 
                        internal_method_kwargs={"global_latent": single_g})
                new_styles.append(style)
            self.styles = new_styles
        else:
            self.global_latent = \
                latent_sampler.sample_global_latent(
                    self.global_latent.shape[0],
                    mixing=mixing,
                    device=self.device)

    def update_local_latent(self, latent_sampler, selection_map, ref_selection_map=None):
        specific_shape = (
            self.local_latent.shape[2],
            self.local_latent.shape[3],
        )
        new_local_latent = latent_sampler.sample_local_latent(
            self.local_latent.shape[0], 
            device=self.local_latent.device,
            specific_shape=specific_shape,
            exclude_padding=True)

        if ref_selection_map is not None:
            bs, ch, _, _ = self.local_latent.shape
            ref_selection_map = ref_selection_map.repeat(1, ch, 1, 1).bool()
            ref_region = self.local_latent[ref_selection_map]
            if len(ref_region) > 0: # Non empty
                ref_region = ref_region.reshape(bs, ch, 1, -1)
                ref_mean = ref_region.mean(3, keepdim=True)
                ref_std  = ref_region.std(3, keepdim=True)
                new_local_latent = new_local_latent * ref_std + ref_mean
                print(" [*] Reference region stats: mean {:.4f}; std {:.4f}".format(
                    ref_mean.mean().item(), ref_std.mean().item()))

        self.local_latent = self.local_latent * (1-selection_map) + new_local_latent * selection_map

    def update_noises(self, selection_maps):
        for i in range(len(self.noises)):
            prev_noise = self.noises[i]
            new_noise = torch.randn_like(prev_noise)
            selection_map = selection_maps[i]
            self.noises[i] = prev_noise * (1-selection_map) + new_noise * selection_map

    def clone_var(self, var):
        if isinstance(var, list):
            return [self.clone_var(v) for v in var]
        elif isinstance(var, tuple):
            return tuple([self.clone_var(v) for v in var])
        elif isinstance(var, dict):
            return {k: self.clone_var(v) for k,v in var.items()}
        elif isinstance(var, torch.Tensor):
            return var.clone()
        elif isinstance(var, numbers.Number):
            return var
        elif isinstance(var, str):
            return var
        elif var is None:
            return None
        else:
            raise TypeError("Unknown dtype {}".format(type(var)))

    def clone(self):
        clone_kwargs = {k: getattr(self, k) for k in self.clone_keys}
        return TestingVars(**self.clone_var(clone_kwargs))

    def _single_to_batch(self, var, batch_size):
        if isinstance(var, list):
            return [self._single_to_batch(el, batch_size) for el in var]
        elif isinstance(var, dict):
            return {k: self._single_to_batch(v, batch_size) for k,v in var.items()}
        elif isinstance(var, torch.Tensor):
            trail_shape = (1, ) * var.ndim
            return var.unsqueeze(0).repeat(batch_size, *trail_shape)
        else:
            raise TypeError("Unknown type {}".format(type(var)))

    def _apply_device(self, var, device):
        if isinstance(var, list):
            return [self._apply_device(el, device) for el in var]
        elif isinstance(var, dict):
            return {k: self._apply_device(v, device) for k,v in var.items()}
        elif isinstance(var, torch.Tensor):
            return var.to(device)
        else:
            raise TypeError("Unknown type {}".format(type(var)))


    def _assert_coords_by_pin_loc(self, cur_meta_coords, ckpt_coords, pin_loc, 
                                  g_ema_module=None, inplace_force_fixing=False):
        device = cur_meta_coords.device
        n, _, h, w = ckpt_coords.shape
        pin_st_x = pin_loc[0] - h // 2
        pin_st_y = pin_loc[1] - w // 2
        aligned_coords = cur_meta_coords[:, :, pin_st_x:pin_st_x+h, pin_st_y:pin_st_y+w]
        # print(" [*] In the future, the coords are is placing at [NE] ({}, {}) with shape {}.".format(
        #     pin_st_x, pin_st_y, ckpt_coords.shape))
        max_diffs = (aligned_coords - ckpt_coords.to(device)).abs()
        max_diffs_per_dim = [max_diffs[:, i].max().item() for i in range(max_diffs.shape[1])]

        calibrate_vert = False
        calibrate_hori = False
        for dim, max_diff in enumerate(max_diffs_per_dim):
            if inplace_force_fixing:
                # [NOTE]
                # This should only happen only if either:
                # 1. The inversion coordinates are inconsistent across configs, or
                # 2. Coordinates are learned and updated during inversion.
                if max_diff > 1e-4:
                    if dim == 0:
                        calibrate_vert = True
                    else:
                        calibrate_hori = True
            else:
                assert max_diff < 1e-4, \
                    "Expect saved coords correctly aligned with the target placement." +\
                    " but got max_diff {} on dim {}".format(max_diff, dim)

    def _assign_by_pin_loc(self, container, patch, pin_loc):
        _, _, patch_h, patch_w = patch.shape
        pin_st_x = pin_loc[0] - patch_h // 2
        pin_st_y = pin_loc[1] - patch_w // 2
        container[:, :, pin_st_x:pin_st_x+patch_h, pin_st_y:pin_st_y+patch_w] = patch.to(container.device)

    def maybe_reset_to_inv_records(self, g_ema_module):
        if self.loaded_inv_records is not None:
            self.replace_by_records(g_ema_module, self.loaded_inv_records, self.loaded_inv_placements, keep_none_modified=True)

    def replace_by_records(self, g_ema_module, inv_records, inv_placements, assert_no_style=False, keep_none_modified=False):

        self.loaded_inv_records = inv_records
        self.loaded_inv_placements = inv_placements

        n_latent = g_ema_module.texture_synthesizer.n_latent
        if hasattr(self, "styles") and (self.styles is not None):
            # Inversion stores wplus styles, always use w_plus styles here
            # [NOTE] Assumes style mixing is already disabled outside
            self.wplus_styles = [
                style[:, 0:1, :].repeat(1, n_latent, 1) for style in self.styles]

        assert_global_idx = []
        for nth_center, (path,loc) in enumerate(zip(inv_records,inv_placements)):
            if path is None:
                # This happens when the number of `inv_records` is less than the number of `style_centers`
                break

            inv_record_obj = pkl.load(open(path, "rb"))["latents"]
            # keys in records:
            # {
            #    "eval": {...}, # Irrelavent metrics
            #    "latents": {
            #       "ss_local_latents": torch.Tensor,
            #       "ss_global_latents": torch.Tensor,
            #       "ts_styles": [torch.Tensor, ...],
            #       "ts_noises": [torch.Tensor, ...],
            #    },
            # }
            batch_size = self.local_latent.shape[0]
            inv_record_obj = self._single_to_batch(inv_record_obj, batch_size)
            inv_record_obj = self._apply_device(inv_record_obj, self.device)

            # [Hubert] Forcingly disable mixing.
            # There are potential risks in still having this bug somewhere else, but have to time to fundamentally fix this
            inv_record_obj["ss_global_latents"][0, 1] = inv_record_obj["ss_global_latents"][0, 0]

            # [Global & styles]
            if hasattr(g_ema_module.config.task, "style_centers"):
                # If fused generation, place the global latent at where closest to the style center
                closest_idx = np.argmin([
                    np.abs(c[0]-loc[0]) + np.abs(c[1]-loc[1]) 
                        for c in g_ema_module.config.task.style_centers])
                assert closest_idx not in assert_global_idx, \
                    "Found two global latents assigned to the same" + \
                    " global latent slot {} (assert list {}).".format(closest_idx, assert_global_idx)
                assert_global_idx.append(closest_idx)
                self.global_latent[closest_idx] = inv_record_obj["ss_global_latents"]
                inv_img_center_loc_ratio = \
                    g_ema_module.config.task.style_centers[closest_idx]
                print(" [*] Replace global latent at {}".format(closest_idx))
            else:
                assert len(inv_records) == 1, \
                    "Pure infinite generation does not support multiple inversion placement."
                self.global_latent = inv_record_obj["ss_global_latents"]
                inv_img_center_loc_ratio = loc

            if hasattr(self, "styles") and (self.styles is not None):
                self.wplus_styles[closest_idx] = inv_record_obj["ts_styles"]
            elif "ts_styles" in inv_record_obj:
                # Single style generation, like infinite generation
                assert len(inv_records) == 1
                self.wplus_styles = inv_record_obj["ts_styles"]

            # Cast the location in the image to place
            config = g_ema_module.config
            _, _, H, W = self.meta_img.shape
            meta_pad_h = (H - config.task.height) // 2
            meta_pad_w = (W - config.task.width) // 2
            inv_img_center_loc_pix = [
                round(inv_img_center_loc_ratio[0] * config.task.height + meta_pad_h),
                round(inv_img_center_loc_ratio[1] * config.task.width  + meta_pad_w)]

            # Reverse the location to all features in SS and TS
            inv_img_size_h = g_ema_module.calc_out_spatial_size(
                inv_record_obj["ss_local_latents"].shape[2], include_ss=True)
            inv_img_size_w = g_ema_module.calc_out_spatial_size(
                inv_record_obj["ss_local_latents"].shape[3], include_ss=True)
            inv_img_st_loc_x = inv_img_center_loc_pix[0] - (inv_img_size_h // 2)
            inv_img_st_loc_y = inv_img_center_loc_pix[1] - (inv_img_size_w // 2)

            assert inv_img_st_loc_x >= 0 and inv_img_st_loc_y >= 0, \
                "Top-left corner of intended image exceeds image boundary. Got ({}, {}).".format(
                    inv_img_st_loc_x, inv_img_st_loc_y)
            assert inv_img_st_loc_x+inv_img_size_h <= H and inv_img_st_loc_y+inv_img_size_w <= W, \
                "Bottom-right corner of intended image exceeds image boundary. Got ({}, {}) > ({}, {}).".format(
                    inv_img_st_loc_x+inv_img_size_h, inv_img_st_loc_y+inv_img_size_w, H, W)

            replacement_mask = torch.zeros(1, 1, H, W, device=self.device)
            replacement_mask[
                :, 
                :, 
                inv_img_st_loc_x:inv_img_st_loc_x+inv_img_size_h, 
                inv_img_st_loc_y:inv_img_st_loc_y+inv_img_size_w] = 1
            _, ts_noise_replacement_masks, pin_loc_list_ss, pin_loc_list_ts = \
                g_ema_module.calibrate_spatial_shape(
                    replacement_mask, direction="backward", padding_mode="replicate", 
                    verbose=False, pin_loc=inv_img_center_loc_pix)

            # The firse element in TS list is corresponding to z_spatial.
            # Remove it, and add output dimension back.
            # ts_noise_replacement_masks = ts_noise_replacement_masks[1:] + [replacement_mask]
            pin_loc_list_ts = pin_loc_list_ts[1:] + [inv_img_center_loc_pix]
            pin_loc_z_local = pin_loc_list_ss[0]

            # Safety check if the placement is correctly set
            # import pdb; pdb.set_trace()
            self._assert_coords_by_pin_loc(
                self.meta_coords, inv_record_obj["coords"], pin_loc_z_local, 
                g_ema_module=g_ema_module)

            # Assign z_local
            self._assign_by_pin_loc(self.local_latent, inv_record_obj["ss_local_latents"], pin_loc_z_local)

            # Start assigning TS noises
            assert len(pin_loc_list_ts) == len(self.noises), \
                "Expected casted number of layers exactly the same as the number of noises, " +\
                    "but got {} != {}".format(len(pin_loc_list_ts), len(self.noises))
            for i in range(len(pin_loc_list_ts)):
                # # [NOTE] Replacement with mask may have numerical instability concer, use pin loc instead.
                # cur_bin_mask = (ts_noise_replacement_masks[i] > 0.1) # Exclude numerical issue and binarize
                # cur_bin_mask = self._single_to_batch(cur_bin_mask, batch_size)
                cur_pin_loc = pin_loc_list_ts[i]
                self._assign_by_pin_loc(self.noises[i], inv_record_obj["ts_noises"][i], cur_pin_loc)

