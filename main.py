import os
import cv2
import uvicorn
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
from utils import load_configs
from ultralytics import YOLO
from urllib.parse import quote
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
    title="Cervical Lesion Cell Intelligent Detection System",
    description="病变宫颈细胞智能检测系统",
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
model = YOLO("pretrained/v10best.pt")
current_model_name = configs.get("model")


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


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith((".png", ".jpg", ".bmp")):
        raise HTTPException(
            status_code=400, detail="Only .png, .jpg, .bmp files are supported")
    try:
        save_path = Path(configs["upload_path"]) / file.filename
        # 异步写入文件
        async with aiofiles.open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
        logger.info(f"File uploaded successfully: {file.filename}")
        return JSONResponse({
            "success": True,
            "message": "File uploaded and processed successfully",
            "image_url": str(save_path),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect(request: Request):
    """处理检测请求"""
    try:
        upload_dir = configs["upload_path"]
        result_dir = configs["results_path"]

        files = sorted(os.listdir(upload_dir))
        if not files:
            raise HTTPException(status_code=400, detail="No uploaded images")
        image_name = files[-1]
        image_path = os.path.join(upload_dir, image_name)
        print("Detecting:", image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=500, detail="Image read failed")
        results = model.predict(source=image_path)
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
                cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    image,
                    f"{int(cls)} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        save_path = os.path.join(result_dir, image_name)
        cv2.imwrite(save_path, image)

        result_url = f"/results/{image_name}"

        return JSONResponse({
            "success": True,
            "message": "Detection completed",
            "result_path": result_url
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                upload_time = datetime.datetime.fromtimestamp(
                    stat.st_mtime).strftime("%Y-%m-%d %H:%M")

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
                logger.warning(
                    f"Failed to get info for model {model_file.name}: {e}")
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
        raise HTTPException(
            status_code=400, detail="Only .pt and .pth files are supported")

    try:
        # 安全检查：限制文件大小（例如500MB）
        max_size = 500 * 1024 * 1024  # 500MB
        file_size = 0

        save_path = MODEL_DIR / file.filename

        # 检查文件是否已存在
        if save_path.exists():
            raise HTTPException(
                status_code=400, detail="Model file already exists")

        # 异步写入文件
        async with aiofiles.open(save_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > max_size:
                    await out.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400, detail="File too large (max 500MB)")
                await out.write(chunk)

        logger.info(
            f"Model uploaded successfully: {file.filename}, size: {file_size/1024/1024:.2f}MB")

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


@app.post("/segment")
async def segment(request: Request):
    """简单的分割接口：接受 JSON {"filename": "xxx.png"}，返回 result_path 和 thumbnail_url"""
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}

        filename = data.get('filename') or data.get('base_name')
        if not filename:
            raise HTTPException(status_code=400, detail="filename is required")

        upload_dir = Path(configs['upload_path'])
        input_path = upload_dir / filename
        if not input_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Uploaded file not found: {filename}")

        # 打开图像并生成简单二值掩码作为示例分割
        with Image.open(input_path) as im:
            gray = im.convert('L')
            # 简单阈值分割
            mask = gray.point(lambda p: 255 if p > 128 else 0)

        results_dir = Path(configs['results_path'])
        results_dir.mkdir(parents=True, exist_ok=True)

        seg_name = f"{input_path.stem}_seg.png"
        seg_path = results_dir / seg_name
        mask.save(seg_path)

        thumb = mask.resize((512, 512))
        thumb_name = f"{input_path.stem}_seg_thumb.png"
        thumb_path = results_dir / thumb_name
        thumb.save(thumb_path)

        result_path = f"/{results_dir.name}/{seg_path.name}"
        thumb_url = f"/{results_dir.name}/{thumb_path.name}"

        return JSONResponse({
            "result_path": result_path,
            "thumbnail_url": thumb_url,
            "message": "Segmentation completed"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select_model")
async def select_model(request: Request):
    """选择当前使用的模型"""
    global model, current_model_name

    try:
        data = await request.json()
        model_name = data.get("name")

        if not model_name:
            raise HTTPException(
                status_code=400, detail="Model name is required")

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
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {e}")

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
            raise HTTPException(
                status_code=400, detail="Model name is required")

        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        # 检查是否是当前使用的模型
        if model_name == current_model_name:
            # 如果删除的是当前模型，需要重置
            model = None
            current_model_name = None
            patch_cache.clear()
            logger.info(
                f"Current model {model_name} was deleted, model cleared")

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

    uvicorn.run(
        "main:app",
        host=configs.get("host", "127.0.0.1"),
        port=configs.get("port", 8000),
        reload=configs.get("reload", True),
        log_level="info"
    )
