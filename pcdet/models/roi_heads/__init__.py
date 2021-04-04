from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pointrcnn_head_da import PointRCNNHeadDA
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'PointRCNNHeadDA': PointRCNNHeadDA
}
