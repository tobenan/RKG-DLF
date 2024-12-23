# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    """GradCAM class helps create visualization results.

    Visualization results are blended by heatmaps and input images.
    This class is modified from
    https://github.com/facebookresearch/SlowFast/blob/master/slowfast/visualization/gradcam_utils.py # noqa
    For more information about GradCAM, please visit:
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(self, model, target_layer_name, colormap='jet'):####viridis
        """Create GradCAM class with recognizer, target layername & colormap.

        Args:
            model (nn.Module): the recognizer model to be used.
            target_layer_name (str): name of convolutional layer to
                be used to get gradients and feature maps from for creating
                localization maps.
            colormap (Optional[str]): matplotlib colormap used to create
                heatmap. Default: 'viridis'. For more information, please visit
                https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
            supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
            'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 
            'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
            'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
            'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
            'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 
            'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 
            'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r',
            'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 
            'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
            'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
            'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 
            'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 
            'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 
            'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r',
            'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r',
            'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
            'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
        """
        from ..models.recognizers import Recognizer2D, Recognizer3D, RecoginizerTwoStream,RecoginizerTwoStream_fuse_noguide,RecoginizerTwoStream_nofuse
        if isinstance(model, Recognizer2D):
            self.is_recognizer2d = True
        elif isinstance(model, Recognizer3D):
            self.is_recognizer2d = False
        elif isinstance(model, RecoginizerTwoStream):
            self.is_recognizer2d = True
        elif isinstance(model, RecoginizerTwoStream_fuse_noguide):
            self.is_recognizer2d = True
        elif isinstance(model, RecoginizerTwoStream_nofuse):
            self.is_recognizer2d = True
        else:
            raise ValueError(
                'GradCAM utils only support Recognizer2D & Recognizer3D.')

        self.model = model
        self.model.eval()
        self.target_gradients = None
        self.target_activations = None

        import matplotlib.pyplot as plt
        self.colormap = plt.get_cmap(colormap)
        
        self.data_mean = torch.tensor(model.cfg.img_norm_cfg['mean'])#36.5539, 61.8800
        self.data_std = torch.tensor(model.cfg.img_norm_cfg['std'])
        if 'backbone_CEUS' in target_layer_name:
            self.data_mean=self.data_mean[0:3]
            self.data_std=self.data_std[0:3]
        else:
            self.data_mean=self.data_mean[3:]
            self.data_std=self.data_std[3:]
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        """Register forward and backward hook to a layer, given layer_name, to
        obtain gradients and activations.

        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.target_gradients = grad_output[0].detach()

        def get_activations(module, input, output):
            self.target_activations = output.clone().detach()

        layer_ls = layer_name.split('/')
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _calculate_localization_map(self, inputs, use_labels, delta=1e-20):
        """Calculate localization map for all inputs with Grad-CAM.

        Args:
            inputs (dict): model inputs, generated by test pipeline,
                at least including two keys, ``imgs`` and ``label``.
            use_labels (bool): Whether to use given labels to generate
                localization map. Labels are in ``inputs['label']``.
            delta (float): used in localization map normalization,
                must be small enough. Please make sure
                `localization_map_max - localization_map_min >> delta`
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (localization_map, preds)
                localization_map (torch.Tensor): the localization map for
                    input imgs.
                preds (torch.Tensor): Model predictions for `inputs` with
                    shape (batch_size, num_classes).
        """
        #inputs['imgs'] = inputs['imgs'].clone()
        inputs['img_b'] = inputs['img_b'].clone()

        # model forward & backward
        preds = self.model(gradcam=True, **inputs)
        if use_labels:
            labels = inputs['label']
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)
        else:
            score = torch.max(preds, dim=-1)[0]
        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()

        if self.is_recognizer2d:
            # [batch_size, num_segments, 3, H, W]
            #b, t, _, h, w = inputs['imgs'].size()
            b, t, _, h, w = inputs['img_b'].size()#imgs 
        else:
            # [batch_size, num_crops*num_clips, 3, clip_len, H, W]
            b1, b2, _, t, h, w = inputs['img_b'].size()
            b = b1 * b2

        gradients = self.target_gradients
        activations = self.target_activations #[32,512,7,7]
        if self.is_recognizer2d:
            # [B*Tg, C', H', W']
            b_tg, c, _, _ = gradients.size()
            tg = b_tg // b
        else:
            # source shape: [B, C', Tg, H', W']
            _, c, tg, _, _ = gradients.size()
            # target shape: [B, Tg, C', H', W']
            gradients = gradients.permute(0, 2, 1, 3, 4)
            activations = activations.permute(0, 2, 1, 3, 4)

        # calculate & resize to [B, 1, T, H, W]
        weights = torch.mean(gradients.view(b, tg, c, -1), dim=3)
        weights = weights.view(b, tg, c, 1, 1)
        activations = activations.view([b, tg, c] +
                                       list(activations.size()[-2:]))
        localization_map = torch.sum(
            weights * activations, dim=2, keepdim=True)
        localization_map = F.relu(localization_map)
        localization_map = localization_map.permute(0, 2, 1, 3, 4)
        localization_map = F.interpolate(
            localization_map,
            size=(t, h, w),
            mode='trilinear',
            align_corners=False)

        # Normalize the localization map.
        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(b, -1), dim=-1, keepdim=True)[0],
            torch.max(localization_map.view(b, -1), dim=-1, keepdim=True)[0])
        localization_map_min = torch.reshape(
            localization_map_min, shape=(b, 1, 1, 1, 1))
        localization_map_max = torch.reshape(
            localization_map_max, shape=(b, 1, 1, 1, 1))
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + delta)
        localization_map = localization_map.data

        return localization_map.squeeze(dim=1), preds

    def _alpha_blending(self, localization_map, input_imgs, alpha):
        """Blend heatmaps and model input images and get visulization results.

        Args:
            localization_map (torch.Tensor): localization map for all inputs,
                generated with Grad-CAM
            input_imgs (torch.Tensor): model inputs, normed images.(0-1)
            alpha (float): transparency level of the heatmap,
                in the range [0, 1].
        Returns:
            torch.Tensor: blending results for localization map and input
                images, with shape [B, T, H, W, 3] and pixel values in
                RGB order within range [0, 1].
        """
        # localization_map shape [B, T, H, W]
        localization_map = localization_map.cpu()

        # heatmap shape [B, T, H, W, 3] in RGB order
        heatmap = self.colormap(localization_map.detach().numpy())
        heatmap = heatmap[:, :, :, :, :3]
        # import numpy as np
        # heat_map = localization_map.detach().numpy().astype(np.uint8)
        # heatmap = []
        # for i in range(32):
        #     heatmap.append(cv2.applyColorMap(heat_map[0,i], cv2.COLORMAP_JET))
        #     heatmap[i] = cv2.cvtColor(heatmap[i], cv2.COLOR_BGR2RGB)
        # heatmap = np.expand_dims(np.array(heatmap),axis=0)
        # heatmap = heatmap/255.
        # heatmap = cv2.applyColorMap(localization_map.detach().numpy(), cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        heatmap = torch.from_numpy(heatmap)
        #heatmap = torch.where(heatmap<0.10,0.15,heatmap)
        # Permute input imgs to [B, T, H, W, 3], like heatmap
        if self.is_recognizer2d:
            # Recognizer2D input (B, T, C, H, W)
            curr_inp = input_imgs.permute(0, 1, 3, 4, 2)
        else:
            # Recognizer3D input (B', num_clips*num_crops, C, T, H, W)
            # B = B' * num_clips * num_crops
            curr_inp = input_imgs.view([-1] + list(input_imgs.size()[2:]))
            curr_inp = curr_inp.permute(0, 2, 3, 4, 1)

        # renormalize input imgs to [0, 1]
        curr_inp = curr_inp.cpu()
        curr_inp *= self.data_std
        curr_inp += self.data_mean
        curr_inp /= 255.

        # alpha blending
        blended_imgs = alpha * heatmap + (1 - alpha) * curr_inp

        return blended_imgs

    def __call__(self, inputs, use_labels=False, alpha=0.5):#0.5 use label
        """Visualize the localization maps on their corresponding inputs as
        heatmap, using Grad-CAM.

        Generate visualization results for **ALL CROPS**.
        For example, for I3D model, if `clip_len=32, num_clips=10` and
        use `ThreeCrop` in test pipeline, then for every model inputs,
        there are 960(32*10*3) images generated.

        Args:
            inputs (dict): model inputs, generated by test pipeline,
                at least including two keys, ``imgs`` and ``label``.
            use_labels (bool): Whether to use given labels to generate
                localization map. Labels are in ``inputs['label']``.
            alpha (float): transparency level of the heatmap,
                in the range [0, 1].
        Returns:
            blended_imgs (torch.Tensor): Visualization results, blended by
                localization maps and model inputs.
            preds (torch.Tensor): Model predictions for inputs.
        """

        # localization_map shape [B, T, H, W]
        # preds shape [batch_size, num_classes]
        localization_map, preds = self._calculate_localization_map(
            inputs, use_labels=use_labels)
        # blended_imgs shape [B, T, H, W, 3]
        blended_imgs = self._alpha_blending(localization_map, inputs['img_b'],# imgs  img_ceus
                                            alpha)

        # blended_imgs shape [B, T, H, W, 3]
        # preds shape [batch_size, num_classes]
        # Recognizer2D: B = batch_size, T = num_segments
        # Recognizer3D: B = batch_size * num_crops * num_clips, T = clip_len
        return blended_imgs, preds
