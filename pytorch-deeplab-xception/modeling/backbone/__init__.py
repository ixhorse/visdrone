from modeling.backbone import resnet, xception, drn, mobilenet, mobilenetv3, shufflenetv2

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'mobilenetv3':
        return mobilenetv3.MobileNetV3_Small()
    elif backbone == 'shufflenet':
        return shufflenetv2.ShuffleNetV2()
    else:
        raise NotImplementedError
