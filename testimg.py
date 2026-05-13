import os
import numpy as np
from PIL import Image
from math import log10

# ===================== 配置路径（你可以直接修改这里）=====================
GT_IMG_PATH = "/media/penglab/datab/wuhongji/nerf_synthetic/lego/test/r_0.png"
RENDER_IMG_PATH = "/media/penglab/datab/wuhongji/gaussian_models/nerf_synthetic/lego/ours_30000/renders/00000.png"
POINT_CLOUD_PATH = "/media/penglab/datab/wuhongji/gaussian_models/nerf_synthetic/lego/point_cloud/iteration_30000/point_cloud.ply"
RENDER_FOLDER = "/media/penglab/datab/wuhongji/gaussian_models/nerf_synthetic/lego/ours_30000/renders"
TRAIN_LOG_PATH = "/home/wuhongji/projects/gaussian-splatting/train_all.log"

# ===================== 1. 检查 point_cloud.ply =====================
print("=" * 60)
print("【1】检查 point_cloud.ply 是否存在")
print("=" * 60)
if os.path.exists(POINT_CLOUD_PATH):
    size = os.path.getsize(POINT_CLOUD_PATH) / 1024 / 1024
    print(f"✅ 存在 | 大小：{size:.2f} MB | 路径：{POINT_CLOUD_PATH}")
else:
    print(f"❌ 不存在 | 路径：{POINT_CLOUD_PATH}")

# ===================== 2. 检查渲染文件夹前5个文件 =====================
print("\n" + "=" * 60)
print("【2】检查渲染输出文件夹（前5个文件）")
print("=" * 60)
if os.path.exists(RENDER_FOLDER):
    files = sorted(os.listdir(RENDER_FOLDER))[:5]
    for f in files:
        print(f"📂 {f}")
else:
    print(f"❌ 渲染文件夹不存在：{RENDER_FOLDER}")

# ===================== 3. 检查渲染图片数值范围 =====================
print("\n" + "=" * 60)
print("【3】渲染图片数值范围（shape / min / max / mean）")
print("=" * 60)
try:
    img = np.array(Image.open(RENDER_IMG_PATH).convert("RGB")).astype(np.float64)
    print(f"✅ RENDER | shape: {img.shape}")
    print(f"min: {img.min():.2f} | max: {img.max():.2f} | mean: {img.mean():.2f}")
except Exception as e:
    print(f"❌ 无法读取渲染图：{e}")

# ===================== 4. 计算 GT vs RENDER MSE / PSNR =====================
print("\n" + "=" * 60)
print("【4】GT 与 渲染图 MSE / PSNR 评估")
print("=" * 60)
if os.path.exists(GT_IMG_PATH) and os.path.exists(RENDER_IMG_PATH):
    try:
        a = np.array(Image.open(GT_IMG_PATH).convert("RGB")).astype(np.float64) / 255.0
        b = np.array(Image.open(RENDER_IMG_PATH).convert("RGB")).astype(np.float64) / 255.0
        mse = ((a - b) ** 2).mean()
        psnr = -10 * log10(mse) if mse > 0 else float("inf")
        print(f"✅ MSE:  {mse:.6f}")
        print(f"✅ PSNR: {psnr:.2f} dB")
    except Exception as e:
        print(f"❌ 计算失败：{e}")
else:
    print("❌ 缺失图片：")
    print(f"   GT: {'存在' if os.path.exists(GT_IMG_PATH) else '缺失'}")
    print(f"   渲染图: {'存在' if os.path.exists(RENDER_IMG_PATH) else '缺失'}")

# ===================== 5. 读取训练日志 Loss & Evaluating =====================
print("\n" + "=" * 60)
print("【5】训练日志关键信息（Loss / Evaluating）")
print("=" * 60)
if os.path.exists(TRAIN_LOG_PATH):
    try:
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        count = 0
        for idx, line in enumerate(lines):
            line = line.strip()
            if "Loss=" in line or "Evaluating" in line:
                print(f"第{idx+1}行 | {line}")
                count += 1
                if count >= 200:
                    break
        if count == 0:
            print("📝 日志中未找到 Loss 或 Evaluating 信息")
    except Exception as e:
        print(f"❌ 读取日志失败：{e}")
else:
    print(f"❌ 日志文件不存在：{TRAIN_LOG_PATH}")

print("\n✅ 所有检查完成！")