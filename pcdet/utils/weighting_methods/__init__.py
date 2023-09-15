from .freematch import FreeMatchThreshold
from .adamatch import AdaMatchThreshold
from .consistant_teacher import AdaptiveThresholdGMM
from .softmatch import SoftMatchThreshold

__all__ = {
    'FreeMatchThreshold': FreeMatchThreshold,
    'AdaMatchThreshold': AdaMatchThreshold,
    'AdaptiveThresholdGMM': AdaptiveThresholdGMM,
    # 'SoftMatchThreshold': SoftMatchThreshold # not finalised yet
    }


def build_thresholding_method(tag, dataset, config):
    model = __all__[config.NAME](tag=tag, dataset=dataset, config=config)

    return model