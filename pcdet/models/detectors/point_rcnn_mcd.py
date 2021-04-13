from .point_rcnn import PointRCNN
import torch

class PointRCNNMCD(PointRCNN):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        is_source = torch.all(batch_dict['is_source'])
        if self.training:
            if is_source:
                loss, tb_dict, disp_dict = self.get_training_loss()
            else:
                loss, tb_dict, disp_dict = self.get_training_loss_target()

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_target(self):
        disp_dict = {}
        disc_point, tb_dict = self.point_head.get_discrepancy_loss()
        loss = disc_point
        return loss, tb_dict, disp_dict