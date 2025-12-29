import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import sys
sys.path.append('../dataset')
from data_context_aware import get_loader as get_loader

from PIL import Image
import torch


synonym ={"cut":["chop"], "cut_with":["chop_with"], "hit":["strike"], "hold":["grasp"], "peel":["skin"], "sit_on":["rest_on"]}

def find_key_by_value(d=synonym, target=None):
    for k, v_list in d.items():
        if target in v_list:
            return k
    return None

def visualize_mask_with_white_bg_cv2(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    white_alpha: float = 0.2,
    mask_color=(255, 0, 0),  # RGB in 0~255
):
    """
    image_rgb: (H, W, 3) RGB uint8
    mask: (H, W) or (H, W, 1), 값은 {0,1} 또는 {0,255} 등 (0보다 크면 마스크)
    white_alpha: 원본 이미지에 흰색 섞는 정도 (0~1)
    mask_color: 마스크 칠할 색 (R,G,B) in 0~255
    return: (H, W, 3) RGB uint8
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be (H,W,3), got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"image_rgb must be uint8, got {image_rgb.dtype}")

    m = mask
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[:, :, 0]
    if m.ndim != 2:
        raise ValueError(f"mask must be (H,W) or (H,W,1), got {mask.shape}")

    # binary mask: True where masked
    m_bin = (m > 0)

    # 1) 흰 배경 섞기 (원본만)
    # bg = (1-a)*img + a*255
    bg = cv2.addWeighted(image_rgb, 1.0 - white_alpha,
                         np.full_like(image_rgb, 255), white_alpha, 0)

    # 2) 마스크 영역을 지정 색으로 완전 덮기
    out = bg.copy()
    out[m_bin] = np.array(mask_color, dtype=np.uint8)  # RGB

    return out
    
import cv2
import numpy as np

def add_top_bottom_text(panel_bgr, top_text, bottom_text,
                        top_pad=26, bottom_pad=26,
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.55, thickness=1):
    """
    panel_bgr: (H, W, 3) uint8 BGR
    위/아래 패딩을 추가하고, 그 패딩 영역에 top/bottom 텍스트를 그려서 반환
    """
    assert panel_bgr.ndim == 3 and panel_bgr.shape[2] == 3
    panel_bgr = np.ascontiguousarray(panel_bgr)

    # 위/아래 여백 추가 (검정 배경)
    out = cv2.copyMakeBorder(panel_bgr, top_pad, bottom_pad, 0, 0,
                             borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    H, W = out.shape[:2]

    def draw_centered(text, y_center):
        if text is None:
            return
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(0, (W - tw) // 2)
        y = int(y_center + th // 2)
        # 흰색 + 가독성 위해 검정 외곽선(Shadow) 살짝
        cv2.putText(out, text, (x, y), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    # 위쪽 텍스트는 top_pad 중앙, 아래쪽은 bottom_pad 중앙
    draw_centered(top_text, top_pad // 2)
    draw_centered(bottom_text, H - bottom_pad // 2)

    return out


# unseen_list = sorted(glob("./unseen*"))
unseen_list = ['./unseen_wsag', './unseen_ours', './unseen_ours_wo_part', './unseen_ours_wo_active', './unseen_ours_wo_passive']
save_dir = "./concat_unseen"
os.makedirs(save_dir,exist_ok=True)

# for i in range(159):
#     print(i+1)

     
eval_data_loader = get_loader(
    gt_dir="../../../06_Affordance-R1/Affordance-R1/vis_results_refine_anno_v2/sampled/AGD20K_refine/",
    batch_size=1,
    img_size=224, # follow LOCATE, Cross-View-AG, eval at 224*224
    split_file="Unseen",
    data_dir="../../../AGD20K",
    shuffle=False,
    train=False,
    exo_obj_file=None, 
    ego_obj_file=None, 
    no_pad_gt=True
)
gt_root = "../../../06_Affordance-R1/Affordance-R1/vis_results_refine_anno_v2/sampled/AGD20K_refine/Seen/testset/egocentric"
rgb_root ="../../../AGD20K/Seen/testset/egocentric"
for ii, batch_data in tqdm(enumerate(eval_data_loader)):
    # print(batch_data)
    aff = find_key_by_value(target=batch_data['verbs'][0])
    rgb_img= cv2.imread(batch_data['input_paths'][0])
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_mask= cv2.imread(f"{gt_root}/{aff}/{batch_data['nouns'][0]}/{batch_data['input_paths'][0].split('/')[-1]}")
    gt_img = visualize_mask_with_white_bg_cv2(
            rgb_img,
            cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY),
            white_alpha = 0.2,
            mask_color=(255.0, 0.0, 0.0)
        )
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    # GT 이미지 크기
    gt_h, gt_w = rgb_img.shape[:2]
    results = gt_img.astype(np.uint8)

    for result_dir in unseen_list:
        
        file_p = glob(f"{result_dir}/img/AGD{str(ii+1)}_*.png")
        if len(file_p) != 1:
            print(file_p)
            import pdb;pdb.set_trace()
        img = cv2.imread(file_p[0])
                
        # 이미지 크기 (height, width, channel)
        h, w, _ = img.shape
        # 오른쪽 절반 슬라이싱
        left_half = img[:, :w//2].astype(np.uint8)
        
        # ---- text 추가 ----
        top_text = os.path.basename(result_dir)  # 또는 원하는 이름 매핑
        kld = float(file_p[0].split("/")[-1].split("_")[2])
        bottom_text = f"KLD: {kld:.4f}" if kld is not None else "KLD: N/A"

        

        # left_half resize (width는 그대로 쓰기 위해 w//2 비율 유지)
        left_half = cv2.resize(
            left_half,
            (gt_w, gt_h),   # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        left_half = add_top_bottom_text(left_half, top_text, bottom_text)
        # ----------------------

        # results도 패널 높이를 맞춰줘야 함(위/아래 패딩 때문에)
        if results.shape[0] != left_half.shape[0]:
            # gt(왼쪽)도 같은 패딩을 줘서 높이를 맞추는 게 가장 깔끔
            results = add_top_bottom_text(results, "GT", f"{aff}-{batch_data['nouns'][0]}")  # 또는 top/bottom 공란
        results = cv2.hconcat([results, left_half])

    cv2.imwrite(f"{save_dir}/AGD{str(ii+1)}.png", results)
