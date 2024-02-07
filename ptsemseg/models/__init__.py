from ptsemseg.models.agent import MIMOcom

def get_model(model_dict, n_classes, version=None):
    name = model_dict["model"]["arch"] #MIMOcom

    model = _get_model_instance(name) #获取模型实例
    in_channels = 3
    if name == "MIMOcom": #实例化模型
        model = model(n_classes=n_classes, #11
                      in_channels=in_channels, #3 RGB
                      attention=model_dict["model"]['attention'], #'general'
                      has_query=model_dict["model"]['query'], #True
                      sparse=model_dict["model"]['sparse'], #False
                      agent_num=model_dict["model"]['agent_num'], #6
                      shared_img_encoder=model_dict["model"]["shared_img_encoder"], #'unified'
                      image_size=model_dict["data"]["img_rows"], #512
                      query_size=model_dict["model"]["query_size"], #32
                      key_size=model_dict["model"]["key_size"], #1024
                      enc_backbone=model_dict["model"]['enc_backbone'], #resnet_encoder
                      dec_backbone=model_dict["model"]['dec_backbone'] #simple_decoder
                      )
    else:
        model = model(n_classes=n_classes, in_channels=in_channels,
                      enc_backbone=model_dict["model"]['enc_backbone'],
                      dec_backbone=model_dict["model"]['dec_backbone'])

    return model


def _get_model_instance(name):
    try:
        return {
            'MIMOcom': MIMOcom ###
        }[name]
    except:
        raise ("Model {} not available".format(name))
