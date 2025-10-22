"""
DeepSeek OCR API 主应用
支持transformers和vllm两种推理模式
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.core.api.controller.ocr_controller import OCRController


def create_app():
    """创建FastAPI应用"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    app = FastAPI(title="DeepSeek-OCR Service")
    
    # 创建OCR控制器
    ocr_controller = OCRController(project_root)
    
    # 挂载API路由
    app.include_router(ocr_controller.router, prefix="/api")
    
    # 静态文件目录
    static_dir = project_root / 'server' / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # 主页
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """主页面"""
        return """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset='utf-8'>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek-OCR - AI 文档识别服务</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        input[type="file"] {
            display: none;
        }
        .file-name {
            margin-top: 15px;
            color: #667eea;
            font-weight: 500;
        }
        .prompt-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .progress-container {
            display: none;
            margin-top: 30px;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        .status-text {
            color: #666;
            text-align: center;
            margin-bottom: 10px;
        }
        .log-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #333;
        }
        .log-line {
            margin-bottom: 5px;
            padding: 3px 0;
        }
        .results {
            display: none;
            margin-top: 30px;
            padding: 20px;
            background: #f0f9ff;
            border-radius: 15px;
            border-left: 4px solid #667eea;
        }
        .results h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .download-link {
            display: block;
            padding: 12px 20px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 10px;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 10px;
            transition: all 0.3s;
            text-align: center;
            font-weight: 500;
        }
        .download-link:hover {
            background: #667eea;
            color: white;
        }
        .download-link.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
        }
        .download-link.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 DeepSeek-OCR</h1>
        <p class="subtitle">AI 驱动的智能文档识别服务</p>
        
        <form id="uploadForm">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📄</div>
                <div>点击或拖拽 PDF 文件到此处</div>
                <div class="file-name" id="fileName"></div>
                <input type="file" id="fileInput" name="file" accept="application/pdf" required />
            </div>
            
            <div class="prompt-group">
                <label for="prompt">自定义 Prompt（可选）</label>
                <input type="text" id="prompt" name="prompt" 
                       placeholder="留空使用默认：<image>\\n<|grounding|>Convert the document to markdown." />
            </div>
            
            <button type="submit" class="btn" id="submitBtn">
                开始识别
            </button>
        </form>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="status-text" id="statusText">准备中...</div>
            <div class="log-container" id="logContainer"></div>
        </div>
        
        <div class="results" id="results">
            <h3>✅ 识别完成！</h3>
            <a href="#" class="download-link" id="linkMmd" download>📝 下载 Markdown 文件</a>
            <a href="#" class="download-link" id="linkDetMmd" download>📋 下载完整标注文件</a>
            <a href="#" class="download-link" id="linkLayouts" download>🖼️ 下载可视化 PDF</a>
            <a href="#" class="download-link" id="linkImages" download>🎨 下载提取的图片 (ZIP)</a>
            <a href="#" class="download-link primary" id="linkAll" download>📦 下载全部文件 (ZIP)</a>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');
        const logContainer = document.getElementById('logContainer');
        const results = document.getElementById('results');
        
        let selectedFile = null;
        let ws = null;
        
        // File upload area interactions
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                fileInput.files = files;
                selectedFile = files[0];
                fileName.textContent = '已选择: ' + files[0].name;
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                fileName.textContent = '已选择: ' + selectedFile.name;
            }
        });
        
        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!selectedFile) {
                alert('请先选择一个 PDF 文件');
                return;
            }
            
            // Show progress, hide results
            progressContainer.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner"></span> 处理中...';
            progressFill.style.width = '10%';
            statusText.textContent = '上传文件中...';
            logContainer.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('prompt', document.getElementById('prompt').value);
            
            try {
                const response = await fetch('/api/ocr/pdf', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('上传失败: ' + response.statusText);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Connect WebSocket for progress updates
                connectWebSocket(data.job_id, data);
                
            } catch (error) {
                statusText.textContent = '❌ 错误: ' + error.message;
                submitBtn.disabled = false;
                submitBtn.textContent = '开始识别';
            }
        });
        
        function connectWebSocket(jobId, resultData) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/${jobId}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    progressFill.style.width = data.progress + '%';
                    statusText.textContent = data.message;
                } else if (data.type === 'log') {
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';
                    logLine.textContent = data.message;
                    logContainer.appendChild(logLine);
                    logContainer.scrollTop = logContainer.scrollHeight;
                } else if (data.type === 'complete') {
                    progressFill.style.width = '100%';
                    statusText.textContent = '✅ 处理完成！';
                    
                    // Show download links
                    document.getElementById('linkMmd').href = resultData.mmd;
                    document.getElementById('linkDetMmd').href = resultData.det_mmd;
                    document.getElementById('linkLayouts').href = resultData.layouts;
                    document.getElementById('linkImages').href = resultData.images;
                    document.getElementById('linkAll').href = resultData.all;
                    results.style.display = 'block';
                    
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始识别';
                } else if (data.type === 'error') {
                    statusText.textContent = '❌ 错误: ' + data.message;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始识别';
                }
            };
            
            ws.onerror = () => {
                statusText.textContent = '⚠️ WebSocket 连接失败，但处理可能仍在继续...';
            };
            
            ws.onclose = () => {
                console.log('WebSocket closed');
            };
        }
    </script>
</body>
</html>
"""
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)