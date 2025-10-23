#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试功能单元测试
"""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDebug(unittest.TestCase):
    """调试功能测试类"""
    
    def test_ipdb_availability(self):
        """测试ipdb是否可用"""
        # 测试导入debug模块
        try:
            from dev.debug import HAS_IPDB
            # 这个测试只是检查ipdb是否可以导入
            # 在实际环境中，可能需要安装ipdb
            self.assertIsInstance(HAS_IPDB, bool)
        except ImportError:
            self.fail("无法导入debug模块")
    
    def test_debug_wrapper(self):
        """测试调试装饰器"""
        # 导入调试装饰器
        try:
            from dev.debug import debug_wrapper
        except ImportError:
            self.fail("无法导入debug模块")
            
        # 创建一个简单的测试函数
        @debug_wrapper
        def test_function(x, y):
            return x + y
        
        # 保存原始的DEBUG_MODE值
        original_debug_mode = os.environ.get("DEBUG", "")
        
        try:
            # 测试非调试模式
            os.environ["DEBUG"] = ""
            result = test_function(1, 2)
            self.assertEqual(result, 3)
            
            # 测试调试模式
            os.environ["DEBUG"] = "TRUE"
            result = test_function(2, 3)
            self.assertEqual(result, 5)
        finally:
            # 恢复原始的DEBUG_MODE值
            if original_debug_mode:
                os.environ["DEBUG"] = original_debug_mode
            else:
                os.environ.pop("DEBUG", None)
    
    @patch('builtins.print')
    def test_debug_trace(self, mock_print):
        """测试调试断点设置"""
        # 导入调试函数
        try:
            from dev.debug import debug_trace
        except ImportError:
            self.fail("无法导入debug模块")
        
        # 保存原始的DEBUG_MODE值
        original_debug_mode = os.environ.get("DEBUG", "")
        
        try:
            # 测试非调试模式
            os.environ["DEBUG"] = ""
            debug_trace()
            mock_print.assert_called_with("调试模式未启用，请设置环境变量 DEBUG=TRUE")
        finally:
            # 恢复原始的DEBUG_MODE值
            if original_debug_mode:
                os.environ["DEBUG"] = original_debug_mode
            else:
                os.environ.pop("DEBUG", None)


if __name__ == '__main__':
    unittest.main()