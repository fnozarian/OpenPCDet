import torch
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from scipy.stats import norm, truncnorm, beta


def _lbl(tensor, mask=None):
    return _lbl(tensor)[_lbl(mask)] if mask is not None else tensor.chunk(2)[0].squeeze(0)


def _ulb(tensor, mask=None):
    return _ulb(tensor)[_ulb(mask)] if mask is not None else tensor.chunk(2)[1].squeeze(0)


def _get_cls_dist(labels):
    cls_counts = labels.int().bincount(minlength=4)[1:]
    return cls_counts / cls_counts.sum()


class AdaptiveThresholding(Metric):
    full_state_update: bool = False

    def __init__(self, **configs):
        super().__init__(**configs)

        self.reset_state_interval = configs.get('RESET_STATE_INTERVAL', 32)
        self.thresh_method = configs.get('THRESH_METHOD', 'AdaMatch')
        self.enable_plots = configs.get('ENABLE_PLOTS', False)
        self.fixed_thresh = configs.get('FIXED_THRESH', 0.9)
        self.momentum = configs.get('MOMENTUM', 0.9)
        self.temperature = configs.get('TEMPERATURE', 1)  # TODO: Both temperatures should be tuned
        self.temperature_sa = configs.get('TEMPERATURE_SA', 1)
        self.ulb_ratio = configs.get('ULB_RATIO', 0.5)
        self.joint_dist_align = configs.get('JOINT_DIST_ALIGN', True)
        self.enable_ulb_cls_dist_loss = configs.get('ENABLE_ULB_CLS_DIST_LOSS', False)
        self.states_name = ['sem_scores_wa', 'conf_scores_wa', 'joint', 'pls_wa', 'gts_wa',
                            'gt_labels_wa', 'scores_rect']
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.thresh = torch.ones(len(self.class_names)) / len(self.class_names) # set by the thresh method
        self.iteration_count = 0
        self.min_overlaps = np.array([0.7, 0.5, 0.5])
        self.bs = configs.get('BS', 16)
        self.avg_num_fgs_per_sample = 4.5  # TODO: get this from the labeled dataset

        # States are of shape (N, M, P) where N is # samples, M is # RoIs and P = 3 is the Car, Ped, Cyc
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.mean_p_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_classwise = {s_name: None for s_name in self.states_name}
        self.std_p_max_model_classwise = {s_name: None for s_name in self.states_name}
        self.labels_hist = {s_name: None for s_name in self.states_name}

        self.mean_p_model_shadow = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_shadow = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_classwise_shadow = {s_name: None for s_name in self.states_name}
        self.std_p_max_model_classwise_shadow = {s_name: None for s_name in self.states_name}
        self.labels_hist_shadow = {s_name: None for s_name in self.states_name}

        self.ratio = {'AdaMatch': None}
        self.max_scores = dict()
        self.labels = dict()
        # Two fixed targets dists
        self.mean_p_model['uniform'] = torch.ones(len(self.class_names)) / len(self.class_names)
        self.mean_p_model['gt'] = torch.tensor([0.82, 0.13, 0.05])

    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)

    def _arrange_tensor(self, tensor):
        splits = torch.split(tensor, int(self.ulb_ratio * self.bs), dim=0)
        lbl = torch.cat(splits[::2], dim=0)
        ulb = torch.cat(splits[1::2], dim=0)
        mstate = torch.cat([lbl, ulb], dim=0)
        return mstate.view(-1, mstate.shape[-1])

    def _accumulate_metrics(self):
        self.bs = len(self.sem_scores_wa[0])  # TODO: Refactor
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if not len(mstate): continue
            assert all(m.shape[0] == mstate[0].shape[0] for m in mstate), "Shapes along axis 0 do not match."
            if isinstance(mstate, list):
                mstate = torch.cat(mstate, dim=0)
            accumulated_metrics[mname] = self._arrange_tensor(mstate)
        return accumulated_metrics

    def _get_p_max_stats(self, max_scores, labels, mask, weights=None, split=None, num_classes=3, type='micro'):
        if split is None:
            split = ['lbl', 'ulb']
        if isinstance(split, str) and split in ['lbl', 'ulb']:
            _split = _lbl if split == 'lbl' else _ulb
            max_scores = _split(max_scores, mask)
            labels = _split(labels, mask)
            p_max_model = labels.new_zeros(num_classes, dtype=max_scores.dtype).scatter_add_(0, labels, max_scores)

            if weights is not None:
                weights = _split(weights, mask)
                labels_hist = torch.bincount(labels, weights=weights, minlength=num_classes)
            else:
                labels_hist = torch.bincount(labels, minlength=num_classes)
            mean_p_max_classwise = p_max_model / (labels_hist + 1e-6)
            std_p_max_classwise = torch.cat([max_scores[labels == i].std().view(1) for i in range(num_classes)], dim=0)
            if type == 'micro':
                p_max_model = p_max_model.sum() / labels_hist.sum()
            elif type == 'macro':
                p_max_model = mean_p_max_classwise.mean()

            labels_hist = labels_hist / labels_hist.sum()
            return p_max_model, labels_hist, mean_p_max_classwise, std_p_max_classwise

        elif isinstance(split, list) and len(split) == 2:
            p_s0, h_s0, pc_s0, std0 = self._get_p_max_stats(max_scores, labels, mask, num_classes=num_classes,
                                                            weights=weights, split=split[0], type=type)
            p_s1, h_s1, pc_s1, std1 = self._get_p_max_stats(max_scores, labels, mask, num_classes=num_classes,
                                                            weights=weights, split=split[1], type=type)
            return (torch.vstack([p_s0, p_s1]).squeeze(), torch.vstack([h_s0, h_s1]),
                    torch.vstack([pc_s0, pc_s1]), torch.vstack([std0, std1]))
        else:
            raise ValueError(f"Invalid split type: {split}")

    def get_max_iou(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6):
        num_anchors = anchors.shape[0]
        num_gts = gt_boxes.shape[0]

        ious = torch.zeros((num_anchors,), dtype=torch.float, device=anchors.device)
        labels = torch.ones((num_anchors,), dtype=torch.int64, device=anchors.device) * -1
        gt_to_anchor_max = torch.zeros((num_gts,), dtype=torch.float, device=anchors.device)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
            gt_to_anchor_max = anchor_by_gt_overlap.max(dim=0)[0]
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            ious[:len(anchor_to_gt_max)] = anchor_to_gt_max

        return ious, labels, gt_to_anchor_max

    @staticmethod
    def pad_tensor(tensor_in, max_len=50):
        assert tensor_in.dim() == 3, "Input tensor should be of shape (N, M, C)"
        diff_ = max_len - tensor_in.shape[1]
        if diff_ > 0:
            tensor_in = torch.cat(
                [tensor_in, torch.zeros((tensor_in.shape[0], diff_, tensor_in.shape[-1]), device=tensor_in.device)],
                dim=1)
        return tensor_in

    def compute(self):
        results = {}

        if len(self.sem_scores_wa) < self.reset_state_interval:
            return

        self.iteration_count += 1
        acc_metrics = self._accumulate_metrics()
        if self.enable_plots:
            fig, axis = plt.subplots(4, figsize=(4, 10), sharex=True, tight_layout=True, facecolor='w', edgecolor='k')
        padding_mask = torch.logical_not(torch.all(acc_metrics['sem_scores_wa'] == 0, dim=-1))
        # sem_mask = torch.logical_not(acc_metrics['sem_scores_wa'].sigmoid().max(dim=-1)[0] <= 0.1)
        # padding_mask = padding_mask & sem_mask
        for sidx, sname in enumerate(['sem_scores_wa', 'conf_scores_wa', 'joint', 'scores_rect']):

            if sname == 'conf_scores_wa':
                scores = acc_metrics['conf_scores_wa']
                sem_scores = acc_metrics['sem_scores_wa']
                _, labels = torch.max(sem_scores, dim=-1)
                max_scores, _ = torch.max(scores, dim=-1)

            if sname == 'sem_scores_wa':
                scores = acc_metrics['sem_scores_wa']
                scores = torch.softmax(scores / self.temperature, dim=-1)
                # scores = torch.sigmoid(scores)
                # scores /= scores.sum(dim=-1, keepdim=True)
                max_scores, labels = torch.max(scores, dim=-1)

            if sname == 'joint' or sname == 'scores_rect':
                sem_logits = acc_metrics['sem_scores_wa']
                sem_scores = torch.softmax(sem_logits / self.temperature, dim=-1)
                conf_scores = acc_metrics['conf_scores_wa'].repeat(1, 3)

                if sname == 'scores_rect':
                    _, labels = torch.max(sem_scores, dim=-1)
                    rect = self.ratio['AdaMatch'] if self.ratio['AdaMatch'] is not None else conf_scores.new_ones(3)
                    conf_scores_new = conf_scores * rect
                    # conf_scores_new = conf_scores_new / _ulb(conf_scores_new, padding_mask).sum()
                    # conf_scores_new *= _ulb(conf_scores, padding_mask).sum()
                    # lbl_conf_scores_new = _lbl(conf_scores) / _lbl(conf_scores, padding_mask).sum()
                    conf_scores = torch.cat([_lbl(conf_scores), _ulb(conf_scores_new)], dim=0)

                scores = sem_scores * conf_scores
                # scores_norm = scores / torch.clamp(scores.sum(dim=-1, keepdim=True), min=1e-6)  # Cancels out conf_scores
                # conf_scores = scores_norm / torch.clamp(scores, min=1e-6)
                max_scores, labels = torch.max(scores, dim=-1)
                max_scores = torch.clamp(max_scores, max=1)

            # TODO: Experiment with pre-filtering the scores
            pre_filtering_mask = max_scores > 0.1
            padding_mask = padding_mask & pre_filtering_mask

            # self.max_scores[sname] = max_scores
            # self.labels[sname] = labels

            mean_p_max, labels_hist, mean_p_max_classwise, std_p_max_classwise = self._get_p_max_stats(max_scores, labels, padding_mask)
            if sname == 'joint' or sname == 'scores_rect':
                mean_p_model_lbl = _lbl(scores, padding_mask).sum(dim=0) / _lbl(conf_scores, padding_mask).sum(dim=0)
                mean_p_model_ulb = _ulb(scores, padding_mask).sum(dim=0) / _ulb(conf_scores, padding_mask).sum(dim=0)
            else:
                mean_p_model_lbl = _lbl(scores, padding_mask).mean(dim=0)
                mean_p_model_ulb = _ulb(scores, padding_mask).mean(dim=0)
            mean_p_model = torch.vstack([mean_p_model_lbl, mean_p_model_ulb])

            self._update_ema('mean_p_max_model', mean_p_max, sname)
            self._update_ema('mean_p_max_model_classwise', mean_p_max_classwise, sname)
            self._update_ema('std_p_max_model_classwise', std_p_max_classwise, sname)
            self._update_ema('labels_hist', labels_hist, sname)
            self._update_ema('mean_p_model', mean_p_model, sname)

            self.log_results(results, sname=sname)

            if self.enable_plots:
                fit_params = self.draw_dist_plots(max_scores, labels, padding_mask, sname, axis, sidx)
                results[f'dist_plots'] = fig
                if fit_params is not None:
                    for j in range(fit_params.shape[1]):
                        results[f'fit_param_{j}'] = self._arr2dict(fit_params[:, j])

        # self.draw_joint_dist_plots(acc_metrics['sem_scores_wa'], acc_metrics['conf_scores_wa'], prev_pad_mask,
        #                            self.pls_wa, self.gts_wa)

        results['labels_hist_lbl/gts_wa'] = self._arr2dict(_get_cls_dist(_lbl(acc_metrics['gt_labels_wa'].view(-1))))
        results['labels_hist_ulb/gts_wa'] = self._arr2dict(_get_cls_dist(_ulb(acc_metrics['gt_labels_wa'].view(-1))))

        if self.thresh_method == 'AdaMatch':
            assert self.joint_dist_align == True, "AdaMatch requires joint scores currently."
            mean_p_model_ulb = _ulb(self.mean_p_model['joint'])
            self.ratio['AdaMatch'] = self.mean_p_model['gt'].to(mean_p_model_ulb.device) / mean_p_model_ulb
            self.thresh = _ulb(self.mean_p_max_model_classwise['scores_rect']) * self.fixed_thresh
            results[f'ratio/AdaMatch'] = self._arr2dict(self.ratio['AdaMatch'])
            results[f'threshold/AdaMatch'] = self._arr2dict(self.thresh)

        elif self.thresh_method == 'FreeMatch':
            results[f'threshold/FreeMatch'] = self._arr2dict(self._get_threshold())

        elif self.thresh_method == 'LabelMatch':
            assert self.joint_dist_align == True, "LabelMatch requires joint scores currently."
            ulb_bs = int(self.bs * self.ulb_ratio)
            exp_num_fgs = self.mean_p_model['gt'].to(
                scores.device) * self.avg_num_fgs_per_sample * ulb_bs * self.reset_state_interval
            exp_num_fgs = exp_num_fgs.ceil().int()
            max_scores_ulb = _ulb(self.max_scores['joint'], padding_mask)
            labels_ulb = _ulb(self.labels['joint'], padding_mask)
            num_scores = max(exp_num_fgs.max() + 1, len(max_scores_ulb))
            classwise_max_scores = torch.zeros((3, num_scores), device=scores.device)
            for cls in range(3):
                mask = labels_ulb == cls
                classwise_max_scores[cls, :mask.sum()] = max_scores_ulb[mask]
            classwise_max_scores = classwise_max_scores.sort(dim=-1, descending=True)[0]
            self.thresh = classwise_max_scores[torch.arange(3).long(), exp_num_fgs.long()]
            results[f'threshold/LabelMatch'] = self._arr2dict(self.thresh)
        self.reset()
        return results

    def log_results(self, results, sname):
        results[f'mean_p_max_model_lbl/{sname}'] = _lbl(self.mean_p_max_model[sname])
        results[f'mean_p_max_model_ulb/{sname}'] = _ulb(self.mean_p_max_model[sname])
        # results[f'std_p_max_model_lbl/{sname}'] = self._arr2dict(_lbl(self.std_p_max_model_classwise[sname]), ignore_zeros=True)
        results[f'std_p_max_model_ulb/{sname}'] = self._arr2dict(_ulb(self.std_p_max_model_classwise[sname]), ignore_zeros=True)
        results[f'labels_hist_lbl/{sname}'] = self._arr2dict(_lbl(self.labels_hist[sname]))
        results[f'labels_hist_ulb/{sname}'] = self._arr2dict(_ulb(self.labels_hist[sname]))
        results[f'mean_p_model_lbl/{sname}'] = self._arr2dict(_lbl(self.mean_p_model[sname]))
        results[f'mean_p_model_ulb/{sname}'] = self._arr2dict(_ulb(self.mean_p_model[sname]))
        results[f'mean_p_max_model_classwise_lbl/{sname}'] = self._arr2dict(
            _lbl(self.mean_p_max_model_classwise[sname]), ignore_zeros=True)
        results[f'mean_p_max_model_classwise_ulb/{sname}'] = self._arr2dict(
            _ulb(self.mean_p_max_model_classwise[sname]), ignore_zeros=True)
        # unbiased_p_model = self.mean_p_model[sname] / self.labels_hist[sname]
        # unbiased_p_model = unbiased_p_model / unbiased_p_model.sum(dim=-1, keepdim=True)
        # results[f'unbiased_p_model_lbl/{sname}'] = self._arr2dict(_lbl(unbiased_p_model))
        # results[f'unbiased_p_model_ulb/{sname}'] = self._arr2dict(_ulb(unbiased_p_model))

    # Use for debugging ;)
    def _get_true_ious_labels(self, pls: [torch.Tensor], gts: [torch.Tensor]):
        batch_gts = torch.cat(gts)
        batch_pls = torch.cat(pls)
        assigned_labels = torch.ones((batch_pls.shape[0], batch_pls.shape[1], 1), dtype=torch.int64,
                                     device=batch_pls.device) * -1
        ious = torch.ones((batch_pls.shape[0], batch_pls.shape[1], 1), dtype=torch.float, device=batch_pls.device)
        for batch_idx in range(len(batch_gts)):
            gts = batch_gts[batch_idx]
            pls = batch_pls[batch_idx]
            mask_gt = torch.logical_not(torch.all(gts == 0, dim=-1))
            mask_pl = torch.logical_not(torch.all(pls == 0, dim=-1))

            valid_gts = gts[mask_gt]
            valid_pls = pls[mask_pl]

            if len(valid_gts) > 0 and len(valid_pls) > 0:
                valid_gts_labels = valid_gts[:, -1].long() - 1
                valid_pls_labels = valid_pls[:, -1].long() - 1
                matched_threshold = torch.tensor(self.min_overlaps, dtype=torch.float, device=valid_pls_labels.device)[
                    valid_pls_labels]
                valid_pls_iou_wrt_gt, assigned_label, gt_to_pls_max_iou = self.get_max_iou(valid_pls[:, 0:7],
                                                                                           valid_gts[:, 0:7],
                                                                                           valid_gts_labels,
                                                                                           matched_threshold=matched_threshold)
                assigned_labels[batch_idx, mask_pl, 0] = assigned_label
                ious[batch_idx, mask_pl, 0] = valid_pls_iou_wrt_gt
        assigned_labels = self._arrange_tensor(assigned_labels)
        return ious, assigned_labels

    def draw_joint_dist_plots(self, sem_scores_wa, conf_scores_wa, mask, pls, gts):
        results = {}

        _, assigned_labels = self._get_true_ious_labels(pls, gts)

        for split in ['lbl', 'ulb']:
            fn = _lbl if split == 'lbl' else _ulb
            sem_scores = fn(sem_scores_wa, mask)
            sem_scores = torch.softmax(sem_scores / self.temperature, dim=-1)
            # sem_scores = fn(acc_metrics['scores_rect'], mask)  # TODO: Decide which scores to use
            conf_scores = fn(conf_scores_wa, mask)
            labels = fn(assigned_labels, mask)
            max_sem, cls = torch.max(sem_scores, dim=-1)
            nbins = 100
            x_lim = (0.5, 1.0)
            y_lim = (0.5, 1.0)
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            for cls_ind, cls_name in enumerate(self.class_names):
                mask = cls == cls_ind
                x, y = max_sem[mask].cpu().numpy(), conf_scores[mask].view(-1).cpu().numpy()
                l = labels[mask].view(-1).cpu().numpy()
                k = kde.gaussian_kde([x, y])
                xi, yi = np.mgrid[x_lim[0]:x_lim[1]:nbins * 1j, y_lim[0]:y_lim[1]:nbins * 1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                ax[cls_ind].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                colors = np.where(l != -1, 'b', 'r')
                ax[cls_ind].scatter(x, y, s=4, c=colors, alpha=0.5, edgecolor='none')
                ax[cls_ind].set_xlim(x_lim)
                ax[cls_ind].set_ylim(y_lim)
                ax[cls_ind].set_title(f"Class: {cls_name}")
                ax[cls_ind].set_xlabel('Max Sem Scores')
                ax[cls_ind].set_ylabel('Conf Scores')
                plt.tight_layout()
            results[f'joint_dist_plots_{split}'] = fig.get_figure()

        return results

    def draw_dist_plots(self, max_scores, labels, mask, sname, axis, sidx, bins=20):
        ulb_mean_p_max = _ulb(self.mean_p_max_model_classwise[sname])
        ulb_std_p_max = _ulb(self.std_p_max_model_classwise[sname])
        ulb_max_scores, ulb_labels = _ulb(max_scores, mask), _ulb(labels, mask)

        colors = plt.cm.tab10.colors
        fit_params = []
        ax = axis[sidx]
        for i in range(3):
            scores = ulb_max_scores[ulb_labels == i].cpu().numpy()
            ax.hist(scores, bins=bins, alpha=0.6, color=colors[i], label=self.class_names[i], density=True)
            ax.axvline(x=self.thresh[i].item(), linestyle='--', color=colors[i], linewidth=2)

            # if sname == 'joint' or sname == 'scores_rect':
                # alpha, beta_val, _, _ = beta.fit(scores, loc=0, scale=1)
                # x = np.linspace(0, 1, 100)
                # p_beta = beta.pdf(x, alpha, beta_val, loc=0, scale=1)
                # ax.plot(x, p_beta, linewidth=1, color=colors[i])
                # fit_params.append(np.array([alpha, beta_val]).reshape(1, -1))

            x = np.linspace(0, 1, 100)
            mu = ulb_mean_p_max[i].item()
            std = ulb_std_p_max[i].item()
            p_norm = norm.pdf(x, mu, std)
            ax.plot(x, p_norm, linewidth=1, color=colors[i])

            # Calculate the parameters for the truncated distribution
            # a = (lower_bound - mu) / std
            # b = (upper_bound - mu) / std

            # truncated_mu, truncated_std = truncnorm.stats(a, b, moments='mv')

            # Plot the truncated Gaussian distribution
            # x_trunc = np.linspace(lower_bound, upper_bound, 100)
            # p_trunc = truncnorm.pdf(x_trunc, a, b, loc=mu, scale=std)
            # plt.plot(x_trunc, p_trunc, '--', linewidth=1, color=colors[i])

        fit_params = np.concatenate(fit_params) if len(fit_params) > 0 else None
        ax.set_xlabel(f'Max {sname} Scores')
        ax.set_xlim(0, 1)
        if sidx == 0:
            ax.set_ylabel('Probability Density')
            ax.legend()
        # bs = len(self.sem_scores_wa[0])
        # info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    Batch Size: {bs}"
        # plt.suptitle(info, fontsize='small')
        # plt.tight_layout()
        # plt.show()
        # return plt.gcf(), fit_params
        return fit_params

    def rectify_scores(self, scores_ulb):
        if self.ratio['AdaMatch'] is not None:
            scores_ulb = scores_ulb * self.ratio['AdaMatch']
            scores_ulb /= scores_ulb.sum(dim=-1, keepdims=True) + 1e-6
        return scores_ulb

    def _update_ema(self, p_name, probs, tag, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        prob_shadow = getattr(self, p_name + '_shadow')
        if prob_shadow[tag] is None:
            prob_shadow[tag] = torch.zeros_like(probs)
        prob_shadow[tag] = momentum * prob_shadow[tag] + (1 - momentum) * probs
        prob = getattr(self, p_name)
        prob[tag] = prob_shadow[tag] / (1 - momentum ** self.iteration_count)

    def get_mask(self, conf_scores, sem_logits):
        assert self.thresh_method in ['AdaMatch', 'FreeMatch', 'DebiasedPL', 'LabelMatch'], \
            f'{self.thresh_method} not in list [AdaMatch, FreeMatch, SoftMatch, DebiasedPL]'

        if self.thresh_method == 'AdaMatch':
            sem_scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            # scores = torch.sigmoid(sem_logits)
            # scores /= scores.sum(dim=-1, keepdim=True)
            _, labels = torch.max(sem_scores, dim=-1)
            rect = self.ratio['AdaMatch'] if self.ratio['AdaMatch'] is not None else conf_scores.new_ones(3)
            conf_weights = torch.gather(rect, 0, labels.long())
            weights = torch.clamp(conf_scores * conf_weights, max=1)
            rect_scores = sem_scores * conf_scores.unsqueeze(-1) * rect
            rect_scores /= rect_scores.sum(dim=-1, keepdim=True)
            max_scores, _ = torch.max(rect_scores, dim=-1)

            # conf_scores_new = conf_scores_new / conf_scores_new.sum()
            # conf_scores_new *= conf_scores.sum()
            # rect_scores = sem_scores * conf_scores_new.unsqueeze(-1)
            # rect_scores = rect_scores / rect_scores.sum(dim=-1, keepdim=True)
            # max_scores, labels = torch.max(rect_scores, dim=-1)
            # thresh = self._get_threshold('scores_rect')
            thresh = self.thresh.to(max_scores.device).view(1, 3).repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return weights > thresh, rect_scores, weights

        elif self.thresh_method == 'FreeMatch':
            # scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            scores = torch.sigmoid(sem_logits)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold()
            thresh = thresh.repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores

        elif self.thresh_method == 'DebiasedPL':
            # rect_logits = sem_logits / self.temperature - torch.log(_ulb(self.mean_p_model['sem_scores_wa']))
            rect_logits = sem_logits - torch.log(_ulb(self.mean_p_model['sem_scores_wa']))
            # scores = torch.softmax(rect_logits, dim=-1)
            scores = torch.sigmoid(rect_logits)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold()
            thresh = thresh.repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores

        elif self.thresh_method == 'LabelMatch':
            # scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            scores = torch.sigmoid(sem_logits)
            # scores /= scores.sum(dim=-1, keepdim=True)
            if self.joint_dist_align:
                scores = scores * conf_scores.unsqueeze(-1)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold()
            thresh = thresh.view(1, 3).repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores

    def _get_threshold(self, tag='sem_scores_wa'):
        if self.thresh_method == 'AdaMatch':
            return _ulb(self.mean_p_max_model_classwise[tag]) * self.fixed_thresh

        elif self.thresh_method == 'FreeMatch':
            normalized_p_model = torch.div(_ulb(self.mean_p_model[tag]), _ulb(self.mean_p_model[tag]).max())
            return normalized_p_model * _ulb(self.mean_p_max_model[tag])

        elif self.thresh_method == 'DebiasedPL':
            thresh = _ulb(self.mean_p_max_model[tag]) * self.fixed_thresh
            return thresh.repeat(3)

        elif self.thresh_method == 'LabelMatch':
            return self.thresh

    def _arr2dict(self, array, ignore_zeros=False):
        if array.shape[-1] == 1:
            return array.item()
        elif array.shape[-1] == len(self.class_names):
            return {cls: array[cind] for cind, cls in enumerate(self.class_names) if
                    array[cind] > 0 or not ignore_zeros}
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")
