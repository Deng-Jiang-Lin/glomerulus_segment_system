import asyncio
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import aiofiles
import openslide
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.model import SegModel
from utils import load_configs

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载配置
configs = load_configs("configs.json")

# 验证必需配置
REQUIRED_CONFIGS = ["upload_path", "results_path", "pretrained_path", 
                    "templates_path", "static_path"]
for key in REQUIRED_CONFIGS:
    if key not in configs:
        raise ValueError(f"Missing required config: {key}")

app = FastAPI(
    title="Glomerulus Segmentation System",
    description="肾小球分割系统",
    version="1.0.0"
)

# 静态文件挂载
for key in REQUIRED_CONFIGS:
    path = Path(configs[key])
    if path.exists():
        app.mount(f"/{path.name}", StaticFiles(directory=path), name=path.name)
    else:
        logger.warning(f"Path does not exist: {path}")

# 模板配置
templates_dir = Path(configs["templates_path"])
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    logger.error(f"Templates directory not found: {templates_dir}")
    raise RuntimeError("Templates directory not found")

# 模型配置
MODEL_DIR = Path(configs["pretrained_path"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量
model: Optional[SegModel] = None
current_model_name: str = configs.get("model")
patch_cache: Dict[str, List[Path]] = {}  # 缓存patch路径


@app.on_event("startup")
async def init_model():
    """初始化模型"""
    global model
    model_path = r"D:\glomerulus_segment_system\pretrained\UNetpp_best_model.pth"
    model = SegModel("UNetpp_best_model", in_chns=3, cls=2)
    logger.info(f"Model initialized successfully")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """登录页面"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request):
    """处理登录请求"""
    try:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        
        # 简单的用户验证（实际项目中应该连接数据库）
        valid_users = {
            "admin": "password123",
            "user": "123456",
            "test": "test123"
        }
        
        if username in valid_users and valid_users[username] == password:
            return JSONResponse({
                "ok": True,
                "user": username,
                "message": "登录成功"
            })
        else:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "用户名或密码错误"
                },
                status_code=401
            )
            
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return JSONResponse(
            {
                "ok": False,
                "error": "登录服务异常"
            },
            status_code=500
        )


async def _generate_patches_from_slide(
    slide_path: Path, 
    base_name: str,
    patch_size: int = 2048,
    resized_size: int = 512,
    max_workers: int = 8
) -> Tuple[Path, Path]:
    """
    从WSI生成patch和下采样预览图
    
    Returns:
        tuple: (patch_dir_path, preview_image_path)
    """
    out_dir = Path(configs["upload_path"]) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with openslide.OpenSlide(str(slide_path)) as slide:
            w, h = slide.dimensions
            
            # 计算网格
            stride = patch_size
            cols = math.ceil(w / patch_size)
            rows = math.ceil(h / patch_size)
            
            # 生成坐标
            coords = [
                (i, j, x, y) 
                for j, y in enumerate(range(0, h, stride))
                for i, x in enumerate(range(0, w, stride))
            ]
            
            # 保存patch的函数
            def save_patch(i: int, j: int, x: int, y: int, idx: int):
                try:
                    with openslide.OpenSlide(str(slide_path)) as s:
                        region = s.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                    
                    region = region.resize((resized_size, resized_size))
                    patch_path = out_dir / f"patch_{idx:04d}.png"
                    region.save(str(patch_path))
                    return i, j, region, patch_path
                except Exception as e:
                    logger.error(f"Error saving patch {idx}: {e}")
                    raise
            
            # 并行处理patches
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(save_patch, i, j, x, y, idx): idx
                    for idx, (i, j, x, y) in enumerate(coords)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process patch {idx}: {e}")
            
            # 合并预览图
            if results:
                merged = Image.new("RGB", (cols * resized_size, rows * resized_size))
                for i, j, region, _ in results:
                    merged.paste(region, (i * resized_size, j * resized_size))
                
                # 调整预览图大小
                merged = merged.resize((1024, 1024))
                merged_path = Path(configs["upload_path"]) / f"{base_name}_preview.png"
                merged.save(str(merged_path))
                
                # 缓存patch路径
                patch_paths = sorted(out_dir.glob("patch_*.png"))
                patch_cache[base_name] = patch_paths
                
                return out_dir, merged_path
            else:
                raise RuntimeError("No patches were generated")
                
    except Exception as e:
        logger.error(f"Error processing slide: {e}")
        raise


@app.post("/upload-trigger")
async def upload_trigger():
    """
    触发后端处理，生成下采样预览
    
    TODO: 从配置或数据库获取实际文件路径
    """
    try:
        # 这里应该从请求参数或配置获取实际路径
        base_name = "12-299_wsi"
        local_slide_path = Path(r"D:\glomerulus_segment_system\temp\11-356_wsi.tiff")
        
        if not local_slide_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide not found: {local_slide_path}")
        
        # 生成patches和预览图
        patch_dir, preview_path = await _generate_patches_from_slide(
            local_slide_path, base_name
        )
        
        # 计算文件大小
        size_mb = preview_path.stat().st_size / (1024 * 1024)
        
        return JSONResponse({
            "filename": str(preview_path),
            "contentType": "image/png",
            "sizeMb": round(size_mb, 2),
            "patchCount": len(patch_cache.get(base_name, [])),
            "message": "Processing completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Upload trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_patches_from_cache_or_dir(base_name: str) -> List[Path]:
    """从缓存或目录获取patches"""
    if base_name in patch_cache:
        return patch_cache[base_name]
    
    patch_dir = Path(configs["upload_path"]) / base_name
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")
    
    patches = sorted(patch_dir.glob("patch_*.png"), 
                     key=lambda x: int(x.stem.split("_")[1]))
    
    if patches:
        patch_cache[base_name] = patches
    
    return patches


@app.post("/segment")
async def segment(request: Request):
    """执行分割"""
    try:
        # 从请求中获取参数或使用默认值
        data = await request.json() if request.headers.get("content-type") == "application/json" else {}
        base_name = data.get("base_name", "12-299_wsi")
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        print(f"Using model: {current_model_name}")
        # 获取patches
        patches = _get_patches_from_cache_or_dir(base_name)
        if not patches:
            raise HTTPException(status_code=404, detail="No patches found")
        
        # 获取patch尺寸
        with Image.open(patches[0]) as img:
            patch_w, patch_h = img.size
        print(f"Patch size: {patch_w}x{patch_h}")
        # 计算合并尺寸
        num_patches = len(patches)
        cols = int(math.sqrt(num_patches))
        rows = math.ceil(num_patches / cols)
        
        merged_mask = Image.new("L", (cols * patch_w, rows * patch_h))
        
        # 分批处理避免内存溢出
        batch_size = 16
        for batch_start in range(0, num_patches, batch_size):
            batch_end = min(batch_start + batch_size, num_patches)
            batch_patches = patches[batch_start:batch_end]
            
            for idx, patch_path in enumerate(batch_patches):
                absolute_idx = batch_start + idx
                
                try:
                    # 执行分割
                    print( f"Processing patch {absolute_idx + 1}/{num_patches}: {patch_path}" )
                    patch_mask = model.predict(str(patch_path))
                    patch_mask_img = Image.fromarray(patch_mask)
                    
                    # 计算位置
                    i = absolute_idx % cols
                    j = absolute_idx // cols
                    
                    # 粘贴到合并图像
                    merged_mask.paste(patch_mask_img, (i * patch_w, j * patch_h))
                    
                except Exception as e:
                    logger.error(f"Error processing patch {patch_path}: {e}")
                    # 使用空白mask作为占位符
                    blank_mask = Image.new("L", (patch_w, patch_h))
                    i = absolute_idx % cols
                    j = absolute_idx // cols
                    merged_mask.paste(blank_mask, (i * patch_w, j * patch_h))
        
        # 保存结果
        results_dir = Path(configs["results_path"])
        results_dir.mkdir(exist_ok=True)
        
        merged_path = results_dir / f"{base_name}_segmentation_mask.png"
        merged_mask.save(str(merged_path))
        
        # 生成缩略图用于预览
        thumbnail = merged_mask.resize((512, 512))
        thumbnail_path = results_dir / f"{base_name}_segmentation_thumbnail.png"
        thumbnail.save(str(thumbnail_path))
        
        return JSONResponse({
            "seg_url": str(merged_path),
            "thumbnail_url": str(thumbnail_path),
            "mask_size": merged_mask.size,
            "patch_count": num_patches,
            "message": "Segmentation completed"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")


@app.get("/models")
async def list_models():
    """获取可用模型列表"""
    try:
        import datetime
        
        model_files = list(MODEL_DIR.glob("*.pt"))
        models_info = []
        
        for model_file in model_files:
            try:
                # 获取文件信息
                stat = model_file.stat()
                size_mb = stat.st_size / (1024 * 1024)  # 转换为MB
                
                # 格式化时间
                upload_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                
                # 创建模型信息字典
                model_info = {
                    "name": model_file.name,
                    "id": hash(model_file.name) % 10000,  # 简单的ID生成
                    "version": "v1.0",  # 默认版本
                    "size_mb": round(size_mb, 2),
                    "upload_time": upload_time,
                    "type": "U-Net" if "unet" in model_file.name.lower() else "通用",
                    "path": str(model_file)
                }
                models_info.append(model_info)
            except Exception as e:
                # 如果获取某个模型信息失败，记录错误但继续处理其他模型
                logger.warning(f"Failed to get info for model {model_file.name}: {e}")
                models_info.append(model_file.name)  # fallback到简单字符串
        
        return JSONResponse({
            "models": models_info,
            "selected": current_model_name,
            "model_dir": str(MODEL_DIR),
            "count": len(models_info)
        })
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """上传模型文件"""
    if not (file.filename.endswith(".pt") or file.filename.endswith(".pth")):
        raise HTTPException(status_code=400, detail="Only .pt and .pth files are supported")
    
    try:
        # 安全检查：限制文件大小（例如500MB）
        max_size = 500 * 1024 * 1024  # 500MB
        file_size = 0
        
        save_path = MODEL_DIR / file.filename
        
        # 检查文件是否已存在
        if save_path.exists():
            raise HTTPException(status_code=400, detail="Model file already exists")
        
        # 异步写入文件
        async with aiofiles.open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > max_size:
                    await out.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="File too large (max 500MB)")
                await out.write(chunk)
        
        logger.info(f"Model uploaded successfully: {file.filename}, size: {file_size/1024/1024:.2f}MB")
        
        return JSONResponse({
            "success": True,
            "msg": "模型上传成功",
            "name": file.filename,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "path": str(save_path)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select_model")
async def select_model(request: Request):
    """选择当前使用的模型"""
    global model, current_model_name
    
    try:
        data = await request.json()
        model_name = data.get("name")
        
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # 尝试加载模型
        try:
            new_model = SegModel(model_path=str(model_path))
            
            # 如果成功加载，更新全局变量
            model = new_model
            current_model_name = model_name
            
            # 清理缓存，因为模型已更改
            patch_cache.clear()
            
            logger.info(f"Model switched to: {model_name}")
            
            return JSONResponse({
                "success": True,
                "selected": model_name,
                "message": f"Successfully switched to model: {model_name}"
            })
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_model")
async def delete_model(request: Request):
    """删除指定的模型文件"""
    global model, current_model_name
    
    try:
        data = await request.json()
        model_name = data.get("name")
        
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # 检查是否是当前使用的模型
        if model_name == current_model_name:
            # 如果删除的是当前模型，需要重置
            model = None
            current_model_name = None
            patch_cache.clear()
            logger.info(f"Current model {model_name} was deleted, model cleared")
        
        # 删除文件
        model_path.unlink()
        logger.info(f"Model deleted successfully: {model_name}")
        
        return JSONResponse({
            "success": True,
            "deleted": model_name,
            "message": f"Successfully deleted model: {model_name}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "current_model": current_model_name,
        "cache_size": len(patch_cache)
    })


@app.post("/clear_cache")
async def clear_cache():
    """清理缓存"""
    try:
        cleared_count = len(patch_cache)
        patch_cache.clear()
        
        return JSONResponse({
            "message": "Cache cleared successfully",
            "cleared_entries": cleared_count
        })
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=configs.get("host", "127.0.0.1"),
        port=configs.get("port", 8000),
        reload=configs.get("reload", True),
        log_level="info"
    )