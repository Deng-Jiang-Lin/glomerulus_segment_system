import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openslide
from PIL import Image
from tqdm import tqdm


def save_patch(patch_img: Image.Image, out_path: str, fmt: str = 'png'):
    patch_img.save(out_path, fmt)


def extract_patches_openslide(
        slide_path: str,
        out_dir: str,
        patch_size: int = 512,
        stride: int = 512,
        level: int = 0,
        downsample: int = 1,
        fmt: str = 'png',
        workers: int = 8,
        max_patches: int = None):
    os.makedirs(out_dir, exist_ok=True)
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.level_dimensions[level]
    read_resize = downsample != 1
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            coords.append((x, y))
            if max_patches and len(coords) >= max_patches:
                break
        if max_patches and len(coords) >= max_patches:
            break

    def worker(i_xy):
        x, y = i_xy
        img = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        if read_resize:
            new_size = (patch_size // downsample, patch_size // downsample)
            img = img.resize(new_size, Image.BILINEAR)
        fname = f"patch_x{x}_y{y}_l{level}.{fmt}"
        out_path = os.path.join(out_dir, fname)
        save_patch(img, out_path, fmt)
        return out_path, x, y

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, c): c for c in coords}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting patches"):
            try:
                results.append(f.result())
            except Exception as e:
                print("Patch save error:", e)

    csv_path = os.path.join(out_dir, "patch_index.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'x', 'y'])
        writer.writerows(results)

    slide.close()
    print(f"Done. Total patches extracted: {len(results)}")


def load_configs(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 格式错误: {str(e)}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"读取配置文件时出错: {str(e)}")
