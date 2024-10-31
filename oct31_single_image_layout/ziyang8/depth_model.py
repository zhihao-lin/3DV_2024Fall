
import depth_pro
# from unidepth.models import UniDepthV2

def build_depth_pro(device):
    model, _ = depth_pro.create_model_and_transforms()
    model.to(device)
    model.eval()
    return model

# def build_unidepth(device):
#     model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
#     model.to(device)
#     model.eval()
#     return model


MODEL_BUILD = {
    # 'unidepth': build_unidepth,
    'depth_pro': build_depth_pro
}