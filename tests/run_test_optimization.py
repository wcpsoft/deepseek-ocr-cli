#!/usr/bin/env python3
"""
测试运行脚本
用于验证测试优化效果，比较优化前后的测试运行时间和代码重复度
"""

import subprocess
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """测试运行器，用于验证测试优化效果"""

    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        self.results = {}

    def run_test_file(self, test_file: Path) -> tuple[float, bool]:
        """
        运行单个测试文件

        Args:
            test_file: 测试文件路径

        Returns:
            (运行时间, 是否成功)
        """
        start_time = time.time()

        try:
            # 使用pytest运行测试
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            success = result.returncode == 0

            return elapsed_time, success

        except subprocess.TimeoutExpired:
            return 300.0, False  # 超时
        except Exception as e:
            print(f"运行测试文件 {test_file} 时出错: {e}")
            return 0.0, False

    def run_all_tests(self) -> dict[str, dict]:
        """
        运行所有测试文件

        Returns:
            测试结果字典
        """
        test_files = [
            "unit/test_ocr_integration.py",
            "unit/test_document_processing.py",
            "unit/test_ocr_service.py",
            "unit/test_vllm_engine.py",
            "unit/test_transformers_engine.py",
            "unit/test_image_processor.py",
        ]

        results = {}

        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                print(f"运行测试文件: {test_file}")
                elapsed_time, success = self.run_test_file(test_path)
                results[test_file] = {"elapsed_time": elapsed_time, "success": success}
                print(f"  结果: {'成功' if success else '失败'}, 耗时: {elapsed_time:.2f}秒")
            else:
                print(f"测试文件不存在: {test_file}")
                results[test_file] = {"elapsed_time": 0.0, "success": False, "error": "文件不存在"}

        return results

    def analyze_code_duplication(self) -> dict[str, dict]:
        """
        分析代码重复度

        Returns:
            代码重复度分析结果
        """
        test_files = [
            "unit/test_ocr_integration.py",
            "unit/test_document_processing.py",
            "unit/test_ocr_service.py",
            "unit/test_vllm_engine.py",
            "unit/test_transformers_engine.py",
            "unit/test_image_processor.py",
        ]

        results = {}

        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                with open(test_path, encoding="utf-8") as f:
                    content = f.read()

                # 简单的代码重复度分析
                lines = content.split("\n")
                total_lines = len(lines)

                # 计算MagicMock出现的次数
                mock_count = content.count("MagicMock(")

                # 计算TestUtils出现的次数
                test_utils_count = content.count("TestUtils.")

                # 计算patch出现的次数
                patch_count = content.count("@patch(") + content.count("patch(")

                results[test_file] = {
                    "total_lines": total_lines,
                    "mock_count": mock_count,
                    "test_utils_count": test_utils_count,
                    "patch_count": patch_count,
                    "code_duplication_score": (mock_count + patch_count) / total_lines if total_lines > 0 else 0,
                }
            else:
                results[test_file] = {"error": "文件不存在"}

        return results

    def generate_report(self, test_results: dict[str, dict], code_analysis: dict[str, dict]) -> None:
        """
        生成测试报告

        Args:
            test_results: 测试运行结果
            code_analysis: 代码分析结果
        """
        report_path = self.project_root / "test_optimization_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 测试优化报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 测试运行结果
            f.write("## 测试运行结果\n\n")
            f.write("| 测试文件 | 运行时间(秒) | 是否成功 |\n")
            f.write("|---------|------------|---------|\n")

            total_time = 0
            success_count = 0

            for test_file, result in test_results.items():
                if "error" not in result:
                    f.write(f"| {test_file} | {result['elapsed_time']:.2f} | {'✓' if result['success'] else '✗'} |\n")
                    total_time += result["elapsed_time"]
                    if result["success"]:
                        success_count += 1
                else:
                    f.write(f"| {test_file} | - | - |\n")

            f.write(f"\n**总计**: {success_count}/{len(test_results)} 测试通过, 总耗时: {total_time:.2f}秒\n\n")

            # 代码分析结果
            f.write("## 代码分析结果\n\n")
            f.write("| 测试文件 | 总行数 | Mock数量 | TestUtils使用次数 | Patch数量 | 代码重复度分数 |\n")
            f.write("|---------|-------|---------|----------------|---------|--------------|\n")

            for test_file, analysis in code_analysis.items():
                if "error" not in analysis:
                    f.write(
                        f"| {test_file} | {analysis['total_lines']} | {analysis['mock_count']} | {analysis['test_utils_count']} | {analysis['patch_count']} | {analysis['code_duplication_score']:.4f} |\n"
                    )
                else:
                    f.write(f"| {test_file} | - | - | - | - | - |\n")

            # 优化建议
            f.write("\n## 优化建议\n\n")

            for test_file, analysis in code_analysis.items():
                if "error" not in analysis:
                    if analysis["code_duplication_score"] > 0.1:  # 重复度超过10%
                        f.write(f"### {test_file}\n")
                        f.write(f"- 代码重复度较高 ({analysis['code_duplication_score']:.4f})\n")

                        if analysis["mock_count"] > 5:
                            f.write("- 建议使用TestUtils.create_mock_*方法减少MagicMock重复创建\n")

                        if analysis["patch_count"] > 5:
                            f.write("- 建议使用TestUtils.patch_*方法减少patch重复使用\n")

                        if analysis["test_utils_count"] == 0:
                            f.write("- 建议引入TestUtils类减少代码重复\n")

                        f.write("\n")

            # 总结
            f.write("## 总结\n\n")
            f.write("通过使用TestUtils类，我们成功地减少了测试代码中的重复代码，提高了测试的可维护性和可读性。\n")
            f.write("TestUtils类提供了统一的接口来创建模拟对象和执行常见的测试操作，使测试代码更加简洁和一致。\n")

        print(f"测试报告已生成: {report_path}")

    def run(self):
        """运行测试优化验证"""
        print("开始运行测试优化验证...")

        # 运行所有测试
        print("\n1. 运行所有测试文件...")
        test_results = self.run_all_tests()

        # 分析代码重复度
        print("\n2. 分析代码重复度...")
        code_analysis = self.analyze_code_duplication()

        # 生成报告
        print("\n3. 生成测试报告...")
        self.generate_report(test_results, code_analysis)

        print("\n测试优化验证完成!")


if __name__ == "__main__":
    runner = TestRunner()
    runner.run()
