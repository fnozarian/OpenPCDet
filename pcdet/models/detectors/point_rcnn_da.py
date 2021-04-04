from .point_rcnn import PointRCNN


class PointRCNNDA(PointRCNN):

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss_da, tb_dict = self.da_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn + loss_da
        return loss, tb_dict, disp_dict
