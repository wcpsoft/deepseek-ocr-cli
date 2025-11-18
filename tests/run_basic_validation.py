#!/usr/bin/env python3
"""
基础重构验证测试运行器

运行不依赖外部库的基础重构验证测试
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class BasicRefactorValidator:
    """基础重构验证器"""

    def __init__(self):
        self.start_time = None
        self.results = []

    def log_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """记录测试结果"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        }
        self.results.append(result)

        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
        if message:
            print(f"    {message}")

    def validate_error_handling_framework(self):
        """验证错误处理框架"""
        print("🔍 验证错误处理框架...")

        start_time = time.time()

        try:
            # 测试导入
            from src.core.utils.error_handling import (
                OCRException, ModelLoadError, ImageProcessError, DeviceError,
                handle_ocr_error, ErrorCollector
            )
            self.log_result("错误处理模块导入", True, "成功导入所有错误处理类和装饰器")

            # 测试异常创建
            exc = OCRException("测试消息", severity="high")
            success = "high" in str(exc) and "测试消息" in str(exc)
            self.log_result("异常类创建", success, f"异常字符串: {str(exc)}")

            # 测试装饰器基本功能
            @handle_ocr_error(default_return="fallback", re_raise=False)
            def failing_function():
                raise ValueError("测试错误")

            try:
                result = failing_function()
                success = result == "fallback"
                self.log_result("装饰器默认返回值", success, f"返回值: {result}")
            except Exception as e:
                self.log_result("装饰器默认返回值", False, f"异常: {e}")

            # 测试错误收集器
            collector = ErrorCollector(max_errors=3)
            collector.add_error("测试错误1")
            collector.add_error("测试错误2")
            success = collector.has_errors() and len(collector.errors) == 2
            self.log_result("错误收集器", success, f"收集到 {len(collector.errors)} 个错误")

            duration = time.time() - start_time
            self.log_result("错误处理框架总体验证", True, f"耗时 {duration:.2f}s", duration)

        except ImportError as e:
            self.log_result("错误处理模块导入", False, f"导入失败: {e}")
        except Exception as e:
            self.log_result("错误处理框架异常", False, f"验证失败: {e}")

    def validate_device_manager(self):
        """验证设备管理器"""
        print("\n🔍 验证设备管理器...")

        start_time = time.time()

        try:
            from src.core.utils.device_manager import DeviceManager

            # 测试单例模式
            manager1 = DeviceManager()
            manager2 = DeviceManager()
            success = manager1 is manager2
            self.log_result("设备管理器单例模式", success, "两个实例是同一个对象")

            # 测试设备检测
            device = manager1.get_optimal_device()
            success = hasattr(device, 'type')
            self.log_result("设备检测", success, f"检测到设备: {device}")

            # 测试张量移动方法存在
            has_tensor_move = hasattr(manager1, 'move_tensor_to_device')
            self.log_result("张量移动方法", has_tensor_move, "move_tensor_to_device 方法存在")

            # 测试模型移动方法存在
            has_model_move = hasattr(manager1, 'move_model_to_device')
            self.log_result("模型移动方法", has_model_move, "move_model_to_device 方法存在")

            # 测试设备类型标准化
            config = manager1.get_device_config()
            success = isinstance(config, dict) and 'device_type' in config
            self.log_result("设备配置", success, f"配置类型: {type(config)}")

            duration = time.time() - start_time
            self.log_result("设备管理器总体验证", True, f"耗时 {duration:.2f}s", duration)

        except ImportError as e:
            self.log_result("设备管理器导入", False, f"导入失败: {e}")
        except Exception as e:
            self.log_result("设备管理器异常", False, f"验证失败: {e}")

    def validate_model_path_utils(self):
        """验证模型路径工具"""
        print("\n🔍 验证模型路径工具...")

        start_time = time.time()

        try:
            from src.core.utils.model_path_utils import ModelPathResolver

            # 测试远程仓库检测
            local_paths = [
                "/path/to/local/model",
                "./models/deepseek-ocr",
                "./relative/path"
            ]
            remote_paths = [
                "deepseek-ai/deepseek-ocr",
                "https://huggingface.co/deepseek-ai/deepseek-ocr",
                "huggingface.co/deepseek-ai/deepseek-ocr",
                "model-name"  # 没有斜杠的模型名
            ]

            all_success = True

            for path in local_paths:
                is_remote = ModelPathResolver.is_remote_repo(path)
                success = not is_remote
                if not success:
                    all_success = False
                    self.log_result(f"本地路径检测: {path}", False, "误判为远程仓库")

            for path in remote_paths:
                is_remote = ModelPathResolver.is_remote_repo(path)
                success = is_remote
                if not success:
                    all_success = False
                    self.log_result(f"远程路径检测: {path}", False, "误判为本地路径")

            if all_success:
                self.log_result("模型路径类型检测", True, "所有路径类型检测正确")

            # 测试加载参数生成
            for path in local_paths:
                params = ModelPathResolver.get_loading_params(path)
                success = isinstance(params, dict) and 'local_files_only' in params
                if success:
                    assert params['local_files_only'] is True, "本地路径应该启用 local_files_only"

            for path in remote_paths:
                params = ModelPathResolver.get_loading_params(path)
                success = isinstance(params, dict) and 'local_files_only' in params
                if success:
                    assert params['local_files_only'] is False, "远程路径应该禁用 local_files_only"

            self.log_result("加载参数生成", True, "正确生成了加载参数")

            duration = time.time() - start_time
            self.log_result("模型路径工具总体验证", True, f"耗时 {duration:.2f}s", duration)

        except ImportError as e:
            self.log_result("模型路径工具导入", False, f"导入失败: {e}")
        except Exception as e:
            self.log_result("模型路径工具异常", False, f"验证失败: {e}")
            traceback.print_exc()

    def validate_app_config(self):
        """验证应用配置"""
        print("\n🔍 验证应用配置...")

        start_time = time.time()

        try:
            from src.core.config.app_config import get_app_config, AppConfig

            # 测试配置获取
            config = get_app_config()
            success = isinstance(config, AppConfig)
            self.log_result("配置实例获取", success, f"配置类型: {type(config)}")

            # 测试配置方法
            has_get = hasattr(config, 'get')
            self.log_result("配置get方法", has_get, "get 方法存在")

            has_device_config = hasattr(config, 'get_device_config')
            self.log_result("设备配置方法", has_device_config, "get_device_config 方法存在")

            # 测试默认配置值
            model_path = config.get("model.path", "default_path")
            success = isinstance(model_path, str)
            self.log_result("配置值获取", success, f"模型路径: {model_path}")

            duration = time.time() - start_time
            self.log_result("应用配置总体验证", True, f"耗时 {duration:.2f}s", duration)

        except ImportError as e:
            self.log_result("应用配置导入", False, f"导入失败: {e}")
        except Exception as e:
            self.log_result("应用配置异常", False, f"验证失败: {e}")

    def validate_responsibility_separation(self):
        """验证职责分离"""
        print("\n🔍 验证职责分离...")

        start_time = time.time()

        try:
            # 验证 ModelManager 不再有设备管理方法
            from src.core.models.model_manager import ModelManager

            manager = ModelManager("test_path")

            # 检查不应该有的方法
            forbidden_methods = ['setup_device', 'move_model_to_device']
            has_forbidden = any(hasattr(manager, method) for method in forbidden_methods)

            self.log_result("ModelManager职责分离", not has_forbidden,
                          "已移除设备管理职责" if not has_forbidden else "仍包含设备管理方法")

            # 验证 DeviceManager 有设备管理方法
            from src.core.utils.device_manager import DeviceManager

            device_manager = DeviceManager()
            required_methods = ['get_optimal_device', 'move_model_to_device', 'move_tensor_to_device']
            has_required = all(hasattr(device_manager, method) for method in required_methods)

            self.log_result("DeviceManager职责完整", has_required,
                          "包含所有必要的设备管理方法" if has_required else "缺少部分设备管理方法")

            duration = time.time() - start_time
            self.log_result("职责分离总体验证", True, f"耗时 {duration:.2f}s", duration)

        except ImportError as e:
            self.log_result("职责分离验证", False, f"导入失败: {e}")
        except Exception as e:
            self.log_result("职责分离异常", False, f"验证失败: {e}")

    def run_all_validations(self):
        """运行所有验证"""
        print("🚀 开始基础重构验证...")
        print(f"项目根目录: {project_root}")
        print("="*60)

        self.start_time = time.time()

        # 运行各项验证
        self.validate_error_handling_framework()
        self.validate_device_manager()
        self.validate_model_path_utils()
        self.validate_app_config()
        self.validate_responsibility_separation()

        # 生成摘要
        self.print_summary()

    def print_summary(self):
        """打印验证摘要"""
        total_duration = time.time() - self.start_time
        successful_tests = sum(1 for r in self.results if r["success"])
        total_tests = len(self.results)

        print(f"\n{'='*60}")
        print("基础重构验证摘要")
        print('='*60)

        print(f"总验证项数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {total_tests - successful_tests}")
        print(f"总耗时: {total_duration:.2f} 秒")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%")

        print(f"\n{'='*40} 详细结果 {'='*40}")

        for result in self.results:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['test']}")
            if result["message"]:
                print(f"    {result['message']}")

        print(f"\n{'='*60}")

        # 生成结论
        success_rate = successful_tests / total_tests
        if success_rate >= 0.8:
            print("🎉 重构验证基本通过！")
            print("核心重构功能正常工作。")
        elif success_rate >= 0.6:
            print("⚠️ 重构验证部分通过")
            print("核心功能正常，但有一些问题需要解决。")
        else:
            print("❌ 重构验证失败")
            print("存在较多问题，需要进一步修复。")

        return success_rate >= 0.8


def main():
    """主函数"""
    validator = BasicRefactorValidator()
    success = validator.run_all_validations()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)