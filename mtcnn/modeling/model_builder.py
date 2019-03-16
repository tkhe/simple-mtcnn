from mtcnn.modeling.pnet import PNet

net_type = {'pnet': PNet}


def build_model(type):
    return net_type[type]()