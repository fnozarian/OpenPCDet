import torch
import torch.functional as F
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import seaborn as sns
from sklearn.mixture import GaussianMixture
from pcdet.ops.iou3d_nms import iou3d_nms_utils
palettes = dict(zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4)))
import pandas as pd
import warnings

# TODO: can we delay the teacher/student weight updates until all the reset_state_interval number of samples are obtained?
# TODO: Test the effect of the reset_state_interval on the performance
# TODO: Set the FG of lbl data based on GTs

def _lbl(tensor, mask=None):
    return _lbl(tensor)[_lbl(mask)] if mask is not None else tensor.chunk(2)[0].squeeze(0)

def _ulb(tensor, mask=None):
    return _ulb(tensor)[_ulb(mask)] if mask is not None else tensor.chunk(2)[1].squeeze(0)

def _get_cls_dist(labels):
    cls_counts = labels.int().bincount(minlength=4)[1:]
    return cls_counts / cls_counts.sum()

class AdaMatch(Metric):
    """
        Adamatch based relative Thresholding
        mean conf. of the top-1 prediction on the weakly aug source data multiplied by a user provided threshold

        Adamatch based Dist. Alignment
        Rectify the target unlabeled pseudo-labels by multiplying them by the ratio of the expected
        value of the weakly aug source labels E[YcapSL;w] to the expected
        value of the target labels E[YcapTU;w], obtaining the final pseudo-labels YtildaTU;w

        REF: UPS FRAMEWORK DA
        probs_x_ulb_w = accumulated_metrics['pl_scores_wa_unlab'].view(-1)
        probs_x_lb_s = accumulated_metrics['pl_scores_wa_lab'].view(-1)
        self.p_model = self.momentum  * self.p_model + (1 - self.momentum) * torch.mean(probs_x_ulb_w)
        self.p_target = self.momentum  * self.p_target + (1 - self.momentum) * torch.mean(probs_x_lb_s)
        probs_x_ulb_aligned = probs_x_ulb_w * (self.p_target + 1e-6) / (self.p_model + 1e-6)
    """
    full_state_update: bool = False

    def __init__(self, **configs):
        super().__init__(**configs)

        self.reset_state_interval = configs.get('RESET_STATE_INTERVAL', 32)
        self.thresh_method = configs.get('THRESH_METHOD', 'AdaMatch')
        self.target_to_align = configs.get('TARGET_TO_ALIGN', 'gt_labels_pre_gt_wa')
        self.prior_sem_fg_thresh = configs.get('SEM_FG_THRESH', 0.5)
        self.enable_plots = configs.get('ENABLE_PLOTS', False)
        self.fixed_thresh = configs.get('FIXED_THRESH', 0.9)
        self.momentum = configs.get('MOMENTUM', 0.9)
        self.temperature = configs.get('TEMPERATURE', 1)  # TODO: Both temperatures should be tuned
        self.temperature_sa = configs.get('TEMPERATURE_SA', 1)
        self.ulb_ratio = configs.get('ULB_RATIO', 0.5)
        self.joint_dist_align = configs.get('JOINT_DIST_ALIGN', False)
        self.enable_ulb_cls_dist_loss = configs.get('ENABLE_ULB_CLS_DIST_LOSS', False)
        self.states_name = ['sem_scores_wa', 'sem_scores_wa_rect', 'conf_scores_wa', 'pls_wa', 'gts_wa', 'gt_labels_wa']
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.iteration_count = 0
        self.min_overlaps = np.array([0.7, 0.5, 0.5])

        # States are of shape (N, M, P) where N is # samples, M is # RoIs and P = 4 is the Car, Ped, Cyc, FG scores
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        # mean_p_model aka p_target (_lbl(mean_p_model)) and p_model (_ulb(mean_p_model))
        self.mean_p_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_classwise = {s_name: None for s_name in self.states_name}
        self.labels_hist = {s_name: None for s_name in self.states_name}

        self.mean_p_model_shadow = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_shadow = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model_classwise_shadow = {s_name: None for s_name in self.states_name}
        self.labels_hist_shadow = {s_name: None for s_name in self.states_name}

        self.ratio = {'AdaMatch': None}

        # Two fixed targets dists
        self.mean_p_model['uniform'] = torch.ones(len(self.class_names)) / len(self.class_names)
        self.mean_p_model['gt'] = torch.tensor([0.82, 0.13, 0.05])

    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)
    def _arrange_tesnor(self, tensor, bs):
        splits = torch.split(tensor, int(self.ulb_ratio * bs), dim=0)
        lbl = torch.cat(splits[::2], dim=0)
        ulb = torch.cat(splits[1::2], dim=0)
        mstate = torch.cat([lbl, ulb], dim=0)
        return mstate.view(-1, mstate.shape[-1])

    def _accumulate_metrics(self):
        bs = len(self.sem_scores_wa[0])  # TODO: Refactor
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if not len(mstate): continue
            assert all(m.shape[0] == mstate[0].shape[0] for m in mstate), "Shapes along axis 0 do not match."
            if isinstance(mstate, list):
                mstate = torch.cat(mstate, dim=0)
            accumulated_metrics[mname] = self._arrange_tesnor(mstate, bs)
        return accumulated_metrics
    def _get_mean_p_max_model_and_label_hist(self, max_scores, labels, mask, hist_minlength=3, split=None, type='micro'):
        if split is None:
            split = ['lbl', 'ulb']
        if isinstance(split, str) and split in ['lbl', 'ulb']:
            _split = _lbl if split == 'lbl' else _ulb
            max_scores = _split(max_scores, mask)
            labels = _split(labels, mask)
            labels_hist = torch.bincount(labels, minlength=hist_minlength)
            p_max_model = labels.new_zeros(hist_minlength, dtype=max_scores.dtype).scatter_add_(0, labels, max_scores)
            mean_p_max_classwise = p_max_model / (labels_hist + 1e-6)

            if type == 'micro':
                p_max_model = p_max_model.sum() / labels_hist.sum()
            elif type == 'macro':
                p_max_model = mean_p_max_classwise.mean()

            labels_hist = labels_hist / labels_hist.sum()
            return p_max_model, labels_hist, mean_p_max_classwise
        elif isinstance(split, list) and len(split) == 2:
            p_s0, h_s0, pc_s0 = self._get_mean_p_max_model_and_label_hist(max_scores, labels, mask, hist_minlength=hist_minlength, split=split[0], type=type)
            p_s1, h_s1, pc_s1 = self._get_mean_p_max_model_and_label_hist(max_scores, labels, mask, hist_minlength=hist_minlength, split=split[1], type=type)
            return torch.vstack([p_s0, p_s1]).squeeze(), torch.vstack([h_s0, h_s1]), torch.vstack([pc_s0, pc_s1])
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
        bs = self.pls_wa[0].shape[0]  # TODO: Refactor
        batch_gts = torch.cat(self.gts_wa)
        batch_pls = torch.cat(self.pls_wa)
        assigned_labels = torch.ones((batch_pls.shape[0], batch_pls.shape[1], 1), dtype=torch.int64, device=batch_pls.device) * -1
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
                matched_threshold = torch.tensor(self.min_overlaps, dtype=torch.float, device=valid_pls_labels.device)[valid_pls_labels]
                valid_pls_iou_wrt_gt, assigned_label, gt_to_pls_max_iou = self.get_max_iou(valid_pls[:, 0:7],
                                                                                       valid_gts[:, 0:7],
                                                                                       valid_gts_labels,
                                                                                       matched_threshold=matched_threshold)
                assigned_labels[batch_idx, :len(assigned_label), 0] = assigned_label
        assigned_labels = self._arrange_tesnor(assigned_labels, bs)

        prev_pad_mask = None
        for sname in ['sem_scores_wa', 'sem_scores_wa_rect', 'conf_scores_wa']:
            scores = acc_metrics[sname]
            padding_mask = torch.logical_not(torch.all(scores == 0, dim=-1))
            if prev_pad_mask is not None:
                assert torch.equal(_ulb(prev_pad_mask), _ulb(padding_mask)), "Padding masks do not match."
            prev_pad_mask = padding_mask

            if sname == 'sem_scores_wa':
                scores = torch.softmax(scores / (self.temperature_sa if '_sa' in sname else self.temperature), dim=-1)
            max_scores, labels = torch.max(scores, dim=-1)

            hist_minlength = scores.shape[-1]
            mean_p_max_model, labels_hist, mean_p_max_model_classwise = self._get_mean_p_max_model_and_label_hist(max_scores, labels, padding_mask, hist_minlength)
            mean_p_model_lbl = _lbl(scores, padding_mask).mean(dim=0)
            mean_p_model_ulb = _ulb(scores, padding_mask).mean(dim=0)
            mean_p_model = torch.vstack([mean_p_model_lbl, mean_p_model_ulb])

            self._update_ema('mean_p_max_model', mean_p_max_model, sname)
            self._update_ema('mean_p_max_model_classwise', mean_p_max_model_classwise, sname)
            self._update_ema('labels_hist', labels_hist, sname)
            self._update_ema('mean_p_model', mean_p_model, sname)

            self.log_results(results, sname=sname)

            if self.enable_plots and self.iteration_count % 10 == 0:
                fig = self.draw_dist_plots(max_scores, labels, padding_mask, sname)
                results[f'dist_plots_{sname}'] = fig

        if self.iteration_count % 50 == 0:
            for split in ['lbl', 'ulb']:
                fn = _lbl if split == 'lbl' else _ulb
                # sem_scores = fn(acc_metrics['sem_scores_wa'], prev_pad_mask)
                # sem_scores = torch.softmax(sem_scores / self.temperature, dim=-1)
                sem_scores = fn(acc_metrics['sem_scores_wa_rect'], prev_pad_mask)
                conf_scores = fn(acc_metrics['conf_scores_wa'], prev_pad_mask)
                labels = fn(assigned_labels, prev_pad_mask)
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


        sem_scores_wa_ulb = _ulb(self.mean_p_model['sem_scores_wa'])
        self.ratio['AdaMatch'] =  self.mean_p_model['gt'].to(sem_scores_wa_ulb.device) / sem_scores_wa_ulb
        # self._update_ema('ratio', ratio, 'AdaMatch')
        results[f'ratio/gt_over_ulb_sem_scores_wa'] = self._arr2dict(self.ratio['AdaMatch'])
        results['labels_hist_lbl/gts_wa'] = self._arr2dict(_get_cls_dist(_lbl(acc_metrics['gt_labels_wa'].view(-1))))
        results['labels_hist_ulb/gts_wa'] = self._arr2dict(_get_cls_dist(_ulb(acc_metrics['gt_labels_wa'].view(-1))))
        if self.thresh_method == 'AdaMatch':
            results[f'threshold/AdaMatch'] = self._arr2dict(self._get_threshold('sem_scores_wa_rect'))
        elif self.thresh_method == 'FreeMatch':
            results[f'threshold/FreeMatch'] = self._arr2dict(self._get_threshold())

        self.reset()
        return results

    def log_results(self, results, sname):
        results[f'mean_p_max_model_lbl/{sname}'] = _lbl(self.mean_p_max_model[sname])
        results[f'mean_p_max_model_ulb/{sname}'] = _ulb(self.mean_p_max_model[sname])
        results[f'labels_hist_lbl/{sname}'] = self._arr2dict(_lbl(self.labels_hist[sname]))
        results[f'labels_hist_ulb/{sname}'] = self._arr2dict(_ulb(self.labels_hist[sname]))
        results[f'mean_p_model_lbl/{sname}'] = self._arr2dict(_lbl(self.mean_p_model[sname]))
        results[f'mean_p_model_ulb/{sname}'] = self._arr2dict(_ulb(self.mean_p_model[sname]))
        results[f'mean_p_max_model_classwise_lbl/{sname}'] = self._arr2dict(_lbl(self.mean_p_max_model_classwise[sname]), ignore_zeros=True)
        results[f'mean_p_max_model_classwise_ulb/{sname}'] = self._arr2dict(_ulb(self.mean_p_max_model_classwise[sname]), ignore_zeros=True)
        # unbiased_p_model = self.mean_p_model[sname] / self.labels_hist[sname]
        # unbiased_p_model = unbiased_p_model / unbiased_p_model.sum(dim=-1, keepdim=True)
        # results[f'unbiased_p_model_lbl/{sname}'] = self._arr2dict(_lbl(unbiased_p_model))
        # results[f'unbiased_p_model_ulb/{sname}'] = self._arr2dict(_ulb(unbiased_p_model))


    def draw_dist_plots(self, max_scores, labels, fg_mask, tag, meta_info=''):

        max_scores_lbl = _lbl(max_scores, fg_mask)
        labels_lbl = _lbl(labels, fg_mask)

        max_scores_ulb = _ulb(max_scores, fg_mask)
        labels_ulb = _ulb(labels, fg_mask)

        BS = len(self.sem_scores_wa[0])
        WS = self.reset_state_interval * BS
        info = (f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    " +
                f"BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}    M: {meta_info}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        plt.suptitle(info, fontsize='small')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fg_scores_labels_lbl_df = pd.DataFrame(
                torch.cat([max_scores_lbl.view(-1, 1), labels_lbl.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            fg_scores_labels_ulb_df = pd.DataFrame(
                torch.cat([max_scores_ulb.view(-1, 1), labels_ulb.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            sns.histplot(data=fg_scores_labels_lbl_df, ax=axes[0], x='scores', binrange=(0.5, 1.0), bins=20, kde_kws={'bw_adjust': 1.5, 'cut':1}, hue='labels', kde=True).set(
                title=f"max-scores dist of WA LBL input {tag} ({len(labels_lbl)})")
            sns.histplot(data=fg_scores_labels_ulb_df, ax=axes[1], x='scores', binrange=(0.5, 1.0), bins=20, kde_kws={'bw_adjust': 1.5, 'cut':1}, hue='labels', kde=True).set(
                title=f"max-scores dist of WA ULB input {tag} ({len(labels_ulb)})")
            plt.tight_layout()
        # plt.show()

        return fig.get_figure()

    def rectify_sem_scores(self, sem_scores_ulb):
        sem_scores_ulb = sem_scores_ulb * self.ratio['AdaMatch']
        sem_scores_ulb /= sem_scores_ulb.sum(dim=-1, keepdims=True)
        return sem_scores_ulb

    def _update_ema(self, p_name, probs, tag, momentum=None):
        momentum = self.momentum if momentum is None else momentum
        prob_shadow = getattr(self, p_name + '_shadow')
        if prob_shadow[tag] is None:
            prob_shadow[tag] = torch.zeros_like(probs)
        prob_shadow[tag] = momentum * prob_shadow[tag] + (1 - momentum) * probs
        prob = getattr(self, p_name)
        prob[tag] = prob_shadow[tag] / (1 - momentum ** self.iteration_count)

    def get_mask(self, conf_scores, sem_logits):
        assert self.thresh_method in ['AdaMatch', 'FreeMatch', 'DebiasedPL'],\
            f'{self.thresh_method} not in list [AdaMatch, FreeMatch, SoftMatch, DebiasedPL]'

        if self.iteration_count == 0:
            scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            if self.joint_dist_align:
                scores = scores * conf_scores.unsqueeze(-1)
            max_scores, labels = torch.max(scores, dim=-1)
            return max_scores > torch.ones_like(max_scores) / len(self.class_names), scores

        if self.thresh_method == 'AdaMatch':
            scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            if self.joint_dist_align:
                scores = scores * conf_scores.unsqueeze(-1)
            scores = self.rectify_sem_scores(scores)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold('sem_scores_wa_rect')
            thresh = thresh.view(1, 3).repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores

        elif self.thresh_method == 'FreeMatch':
            scores = torch.softmax(sem_logits / self.temperature, dim=-1)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold()
            thresh = thresh.repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores # these scores will be discarded

        elif self.thresh_method == 'DebiasedPL':
            # TODO can we use the same logic for conf score rectification?
            # I guess the rect_logits can be equivalently implemented as:
            # rect_logits = logits - lambda * torch.log(_ulb(self.mean_p_model['sem_scores_wa']))
            # where lambda is a hyperparameter proportional to the temperature.
            rect_logits = sem_logits / self.temperature - torch.log(_ulb(self.mean_p_model['sem_scores_wa']))
            scores = torch.softmax(rect_logits, dim=-1)
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold()
            thresh = thresh.repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh, scores  # these scores will be discarded

    def _get_threshold(self, tag='sem_scores_wa'):
        if self.thresh_method == 'AdaMatch':
            return _ulb(self.mean_p_max_model_classwise[tag]) * self.fixed_thresh

        elif self.thresh_method == 'FreeMatch':
            normalized_p_model = torch.div(_ulb(self.mean_p_model[tag]), _ulb(self.mean_p_model[tag]).max())
            return normalized_p_model * _ulb(self.mean_p_max_model[tag])

        elif self.thresh_method == 'DebiasedPL':
            thresh = _ulb(self.mean_p_max_model[tag]) * self.fixed_thresh
            return thresh.repeat(3)

    def _arr2dict(self, array, ignore_zeros=False):
        if array.shape[-1] == 1:
            return array.item()
        elif array.shape[-1] == len(self.class_names):
            return {cls: array[cind] for cind, cls in enumerate(self.class_names) if array[cind] > 0 or not ignore_zeros}
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")