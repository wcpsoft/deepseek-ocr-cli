# 调试指南

本文档介绍了如何在 DeepSeek OCR 项目中使用调试工具。

## 调试模式设置

要启用调试模式，需要设置环境变量 `DEBUG=TRUE`：

```bash
# Linux/macOS
export DEBUG=TRUE

# Windows (PowerShell)
$env:DEBUG="TRUE"

# Windows (命令提示符)
set DEBUG=TRUE
```

或者在运行命令时直接设置：

```bash
DEBUG=TRUE python your_script.py
```

## 使用 ipdb 调试器

### 安装 ipdb

ipdb 已经添加到项目的开发依赖中，可以通过以下命令安装：

```bash
# 使用 uv 安装
uv pip install ipdb

# 或者安装所有开发依赖
uv pip install -e .[dev]
```

### 在代码中使用调试器

项目中的日志模块已经集成了调试功能，可以通过以下方式使用：

1. **设置断点**：
   ```python
   from src.core.logging import debug_trace
   debug_trace()
   ```

2. **使用调试装饰器**：
   ```python
   from src.core.logging import debug_wrapper
   
   @debug_wrapper
   def my_function():
       # 函数代码
       pass
   ```

## 调试脚本

项目提供了一个专门的调试脚本 [dev/debug.py](file:///Users/zhangruixiang/aiworkspace/DeepSeek-OCR/dev/debug.py)，可以通过以下方式运行：

```bash
# 启用调试模式并运行调试脚本
DEBUG=TRUE python -m dev.debug
```

## 日志级别

在调试模式下，日志系统会自动切换到 DEBUG 级别，显示更详细的日志信息。

### 日志级别说明

- **DEBUG**：详细的调试信息，仅在调试时使用
- **INFO**：一般信息，记录系统正常运行状态
- **WARNING**：警告信息，系统可以处理但需要注意的情况
- **ERROR**：错误信息，系统无法正常处理的情况
- **CRITICAL**：严重错误，系统可能无法继续运行

## 调试最佳实践

1. **使用条件断点**：在关键位置设置断点，但只在特定条件下触发
2. **使用日志记录**：通过日志记录关键变量的值和程序执行流程
3. **逐步调试**：从高层次函数开始，逐步深入到具体实现
4. **清理调试代码**：调试完成后，移除或注释掉调试代码

## 常见调试场景

### 1. 模型加载问题

```bash
DEBUG=TRUE deepseek-ocr --download-models
```

### 2. OCR处理问题

```bash
DEBUG=TRUE deepseek-ocr input.pdf output/
```

### 3. 设备兼容性问题

```bash
DEBUG=TRUE deepseek-ocr-detect-gpu
```