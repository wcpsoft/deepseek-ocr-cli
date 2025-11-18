#!/usr/bin/env python3
"""
独立重构测试

直接测试重构的文件内容，不依赖模块导入
"""

import os
import re
from pathlib import Path

class StandaloneRefactorValidator:
    """独立重构验证器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = []

    def log_result(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        self.results.append({
            "test": test_name,
            "success": success,
            "message": message
        })

        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
        if message:
            print(f"    {message}")

    def read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            full_path = self.project_root / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"无法读取文件 {file_path}: {e}")

    def test_error_handling_file_structure(self):
        """测试错误处理文件结构"""
        print("🔍 测试错误处理文件结构...")

        try:
            content = self.read_file_content("src/core/utils/error_handling.py")

            # 检查关键类和函数
            required_elements = [
                "class OCRException",
                "class ModelLoadError",
                "class ImageProcessError",
                "class DeviceError",
                "def handle_ocr_error",
                "class ErrorCollector",
                "def safe_execute"
            ]

            all_found = True
            for element in required_elements:
                if element in content:
                    self.log_result(f"包含 {element}", True)
                else:
                    self.log_result(f"包含 {element}", False, "未找到")
                    all_found = False

            # 检查装饰器导出
            decorators = [
                "handle_model_error",
                "handle_image_error",
                "handle_device_error",
                "handle_config_error"
            ]

            for decorator in decorators:
                if f"def {decorator}" in content or f"{decorator} = " in content:
                    self.log_result(f"装饰器 {decorator}", True)
                else:
                    self.log_result(f"装饰器 {decorator}", False, "未找到")

            self.log_result("错误处理文件结构", all_found, "所有必要元素都已找到")

        except Exception as e:
            self.log_result("错误处理文件结构", False, f"分析失败: {e}")

    def test_model_manager_refactor(self):
        """测试 ModelManager 重构"""
        print("\n🔍 测试 ModelManager 重构...")

        try:
            content = self.read_file_content("src/core/models/model_manager.py")

            # 检查已移除的方法
            removed_methods = ["def setup_device", "def move_model_to_device"]
            has_removed_methods = any(method in content for method in removed_methods)

            self.log_result("已移除设备管理方法", not has_removed_methods,
                          "设备管理方法已成功移除" if not has_removed_methods else "仍包含设备管理方法")

            # 检查仍然存在的必要方法
            required_methods = [
                "def load_model_and_tokenizer",
                "def _load_model",
                "def load_processor",
                "def adjust_vocab_size"
            ]

            for method in required_methods:
                if method in content:
                    self.log_result(f"保留 {method}", True)
                else:
                    self.log_result(f"保留 {method}", False, "方法缺失")

            # 检查错误处理装饰器
            has_error_decorator = "@handle_model_error" in content
            self.log_result("错误处理装饰器", has_error_decorator,
                          "已添加错误处理装饰器" if has_error_decorator else "缺少错误处理装饰器")

            # 检查设备相关注释
            has_device_comments = "# 移除了 self.device" in content or "设备管理交给 DeviceManager" in content
            self.log_result("设备管理注释", has_device_comments,
                          "已添加职责变更说明" if has_device_comments else "缺少职责变更说明")

        except Exception as e:
            self.log_result("ModelManager 重构", False, f"分析失败: {e}")

    def test_device_manager_extensions(self):
        """测试 DeviceManager 扩展"""
        print("\n🔍 测试 DeviceManager 扩展...")

        try:
            content = self.read_file_content("src/core/utils/device_manager.py")

            # 检查新增的方法
            added_methods = [
                "def move_model_to_device",
                "def move_tensor_to_device"
            ]

            for method in added_methods:
                if method in content:
                    self.log_result(f"新增 {method}", True)
                else:
                    self.log_result(f"新增 {method}", False, "方法缺失")

            # 检查方法实现
            has_model_implementation = "model.to(device)" in content and "def move_model_to_device" in content
            self.log_result("模型移动实现", has_model_implementation,
                          "包含正确的模型移动实现" if has_model_implementation else "缺少模型移动实现")

            has_tensor_implementation = "return tensor.to(device)" in content and "def move_tensor_to_device" in content
            self.log_result("张量移动实现", has_tensor_implementation,
                          "包含正确的张量移动实现" if has_tensor_implementation else "缺少张量移动实现")

        except Exception as e:
            self.log_result("DeviceManager 扩展", False, f"分析失败: {e}")

    def test_image_handler_refactor(self):
        """测试 ImageHandler 重构"""
        print("\n🔍 测试 ImageHandler 重构...")

        try:
            content = self.read_file_content("src/core/process/image_handler.py")

            # 检查 extract_tensors 方法的参数更新
            extract_method_pattern = r"def extract_tensors\(.*device_manager.*\)"
            has_device_manager_param = re.search(extract_method_pattern, content)
            self.log_result("extract_tensors 参数更新", bool(has_device_manager_param),
                          "已添加 device_manager 参数" if has_device_manager_param else "缺少 device_manager 参数")

            # 检查统一张量操作的使用
            has_unified_operation = "device_manager.move_tensor_to_device" in content
            self.log_result("统一张量操作", has_unified_operation,
                          "使用 DeviceManager 统一操作" if has_unified_operation else "未使用统一操作")

            # 检查回退机制
            has_fallback = "else:" in content and ".to(device)" in content
            self.log_result("回退机制", has_fallback,
                          "包含向后兼容的回退机制" if has_fallback else "缺少回退机制")

            # 检查错误处理装饰器
            has_error_decorator = "@handle_image_error" in content
            self.log_result("图像错误装饰器", has_error_decorator,
                          "已添加图像错误处理装饰器" if has_error_decorator else "缺少错误处理装饰器")

        except Exception as e:
            self.log_result("ImageHandler 重构", False, f"分析失败: {e}")

    def test_ocr_service_refactor(self):
        """测试 OCRService 重构"""
        print("\n🔍 测试 OCRService 重构...")

        try:
            content = self.read_file_content("src/core/service/ocr_service.py")

            # 检查错误处理装饰器的导入
            has_error_import = "from src.core.utils.error_handling import" in content
            self.log_result("错误处理导入", has_error_import,
                          "已导入错误处理模块" if has_error_import else "缺少错误处理导入")

            # 检查装饰器的使用
            decorators = [
                "@handle_ocr_error",
                "@handle_image_error"
            ]

            for decorator in decorators:
                if decorator in content:
                    self.log_result(f"使用 {decorator}", True)
                else:
                    self.log_result(f"使用 {decorator}", False, "装饰器未使用")

            # 检查简化的 try-except 块
            # 移除了 try-except 的方法应该更简洁
            init_method_pattern = r"def initialize\(.*?\):(.*?)@"
            init_match = re.search(init_method_pattern, content, re.DOTALL)
            if init_match:
                init_body = init_match.group(1)
                has_removed_try = "try:" not in init_body or "except" not in init_body
                self.log_result("简化错误处理", has_removed_try,
                              "已移除冗余的 try-except" if has_removed_try else "仍包含冗余的 try-except")

        except Exception as e:
            self.log_result("OCRService 重构", False, f"分析失败: {e}")

    def test_transformers_engine_refactor(self):
        """测试 TransformersEngine 重构"""
        print("\n🔍 测试 TransformersEngine 重构...")

        try:
            content = self.read_file_content("src/core/transformers/transformers_engine.py")

            # 检查依赖注入参数
            has_device_manager_param = "device_manager:" in content
            self.log_result("设备管理器注入", has_device_manager_param,
                          "支持设备管理器依赖注入" if has_device_manager_param else "缺少设备管理器注入")

            # 检查使用 DeviceManager 的代码
            has_device_usage = "device_manager.get_optimal_device" in content
            self.log_result("设备管理器使用", has_device_usage,
                          "使用注入的设备管理器" if has_device_usage else "未使用设备管理器")

            # 检查回退机制
            has_fallback = "else:" in content and "get_optimal_device()" in content
            self.log_result("回退支持", has_fallback,
                          "包含无注入时的回退机制" if has_fallback else "缺少回退机制")

        except Exception as e:
            self.log_result("TransformersEngine 重构", False, f"分析失败: {e}")

    def test_configuration_integration(self):
        """测试配置集成"""
        print("\n🔍 测试配置集成...")

        try:
            content = self.read_file_content("src/core/config/app_config.py")

            # 检查设备配置方法
            has_device_config = "def get_device_config" in content
            self.log_result("设备配置方法", has_device_config,
                          "包含设备配置获取方法" if has_device_config else "缺少设备配置方法")

            # 检查配置验证
            has_validation = "validate_config" in content or "is_valid" in content
            self.log_result("配置验证", has_validation,
                          "包含配置验证功能" if has_validation else "缺少配置验证")

        except Exception as e:
            self.log_result("配置集成", False, f"分析失败: {e}")

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        print("\n🔍 测试向后兼容性...")

        try:
            # 检查 ModelPathResolver 的弃用函数
            model_path_content = self.read_file_content("src/core/utils/model_path_utils.py")
            has_deprecated_funcs = "def is_remote_repo" in model_path_content and "def get_loading_params" in model_path_content
            self.log_result("ModelPathResolver 兼容函数", has_deprecated_funcs,
                          "保持向后兼容的便捷函数" if has_deprecated_funcs else "缺少兼容函数")

            # 检查 mps_utils 的弃用包装
            try:
                mps_content = self.read_file_content("src/core/utils/mps_utils.py")
                is_deprecation_wrapper = "弃用" in mps_content or "deprecated" in mps_content.lower()
                self.log_result("mps_utils 弃用包装", is_deprecation_wrapper,
                              "已转换为弃用包装器" if is_deprecation_wrapper else "缺少弃用包装")
            except FileNotFoundError:
                # 文件可能被删除或移动
                self.log_result("mps_utils 弃用包装", True, "文件已正确移除")

        except Exception as e:
            self.log_result("向后兼容性", False, f"分析失败: {e}")

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始独立重构验证...")
        print(f"项目根目录: {self.project_root}")
        print("="*60)

        # 运行所有测试
        self.test_error_handling_file_structure()
        self.test_model_manager_refactor()
        self.test_device_manager_extensions()
        self.test_image_handler_refactor()
        self.test_ocr_service_refactor()
        self.test_transformers_engine_refactor()
        self.test_configuration_integration()
        self.test_backward_compatibility()

        # 生成摘要
        self.print_summary()

    def print_summary(self):
        """打印测试摘要"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        success_rate = successful_tests / total_tests * 100

        print(f"\n{'='*60}")
        print("独立重构验证摘要")
        print('='*60)

        print(f"总验证项数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {total_tests - successful_tests}")
        print(f"成功率: {success_rate:.1f}%")

        print(f"\n{'='*40} 详细结果 {'='*40}")

        for result in self.results:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['test']}")
            if result["message"]:
                print(f"    {result['message']}")

        print(f"\n{'='*60}")

        # 生成结论
        if success_rate >= 80:
            print("🎉 重构验证成功！")
            print("重构目标已基本达成，代码结构良好。")
        elif success_rate >= 60:
            print("⚠️ 重构验证部分通过")
            print("主要功能正常，但有一些细节需要完善。")
        else:
            print("❌ 重构验证需要改进")
            print("存在较多问题，建议进一步优化。")

        # 生成建议
        print(f"\n📋 建议:")
        if success_rate >= 80:
            print("- 可以考虑创建发布版本")
            print("- 更新相关文档")
            print("- 继续进行 P2 优化（可选）")
        else:
            failed_tests = [r['test'] for r in self.results if not r['success']]
            print("- 重点关注以下失败的验证:")
            for test in failed_tests:
                print(f"  • {test}")

        return success_rate >= 80


def main():
    """主函数"""
    validator = StandaloneRefactorValidator()
    success = validator.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)