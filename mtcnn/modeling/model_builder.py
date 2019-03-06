from mtcnn.modeling.pnet import PNet
from mtcnn.modeling.rnet import RNet

net_type = {'pnet': PNet, 'rnet': RNet}


def build_model(type):
    return net_type[type]()
