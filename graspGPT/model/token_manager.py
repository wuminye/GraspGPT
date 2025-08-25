# -*- coding: utf-8 -*-
"""
Token管理器 - 集中管理所有tokens，避免多次import时的覆盖问题
"""

import json
import re
from typing import List, Dict, Optional


class TokenManager:
    """
    Token管理器，负责管理所有类型的tokens
    """
    
    def __init__(self):
        # 基础形状标签
        self._base_shape_tags = ['unknow']
        self._base_shape_tags = self._base_shape_tags + ['object%02d' % i for i in range(88)]  # object00 - object87

        # 序列标签模式 - 根据文法定义：SERIAL → '<serial' INT '>'，范围1-240
        self._serial_pattern = re.compile(r'^<serial(\d+)>$')
        self._max_serial_value = 240
        
        # 命令标签，需要出现在AST中
        self._command_tokens = ['unlabel', 'segment', 'endunseg', 'fragment', 'inpaint', 'endinpaint', 'tagfragment', 'amodal', 'endamodal', 'end', 'feat']
        
        # 动态标签列表
        self._dynamic_tags = []
        
        # 缓存计算后的标签列表
        self._shape_tags = None
        self._all_tokens = None
    
    @property
    def base_shape_tags(self) -> List[str]:
        """获取基础形状标签"""
        return self._base_shape_tags.copy()
    
    @property
    def serial_pattern(self) -> re.Pattern:
        """获取序列标签的正则表达式模式"""
        return self._serial_pattern
    
    @property
    def max_serial_value(self) -> int:
        """获取序列标签的最大值"""
        return self._max_serial_value
    
    @property
    def serial_tokens(self) -> List[str]:
        """获取所有有效的序列标签列表（1-240）"""
        return [f'<serial{i}>' for i in range(1, self._max_serial_value + 1)]
    
    @property
    def command_tokens(self) -> List[str]:
        """获取命令标签"""
        return self._command_tokens.copy()
    
    @property
    def dynamic_tags(self) -> List[str]:
        """获取动态标签"""
        return self._dynamic_tags.copy()
    
    @property
    def shape_tags(self) -> List[str]:
        """获取所有形状标签（基础+动态）"""
        if self._shape_tags is None:
            self._shape_tags = self._base_shape_tags + self._dynamic_tags
        return self._shape_tags.copy()
    
    @property
    def all_tokens(self) -> List[str]:
        """获取所有tokens（包括所有序列标签1-240）"""
        if self._all_tokens is None:
            self._all_tokens = self._base_shape_tags + self._dynamic_tags + self._command_tokens + self.serial_tokens
        return self._all_tokens.copy()
    
    def load_vocabulary(self, vocab_file: str) -> bool:
        """
        从词汇文件加载动态标签
        
        Args:
            vocab_file: 词汇文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # 获取类别列表
            categories = vocab_data.get('categories', [])
            
            # 更新动态标签
            self._dynamic_tags = categories
            
            # 清除缓存，强制重新计算
            self._shape_tags = None
            self._all_tokens = None
            
            print(f"词汇加载成功，包含 {len(categories)} 个动态标签")
            print(f"总标签数: {len(self.shape_tags)} (基础: {len(self._base_shape_tags)}, 动态: {len(self._dynamic_tags)})")
            
            return True
            
        except Exception as e:
            print(f"加载词汇文件失败: {e}")
            return False
    
    def set_dynamic_tags(self, tags: List[str]) -> None:
        """
        设置动态标签列表
        
        Args:
            tags: 动态标签列表
        """
        self._dynamic_tags = tags.copy()
        # 清除缓存，强制重新计算
        self._shape_tags = None
        self._all_tokens = None
    
    def add_dynamic_tag(self, tag: str) -> None:
        """
        添加动态标签
        
        Args:
            tag: 要添加的标签
        """
        if tag not in self._dynamic_tags:
            self._dynamic_tags.append(tag)
            # 清除缓存，强制重新计算
            self._shape_tags = None
            self._all_tokens = None
    
    def remove_dynamic_tag(self, tag: str) -> bool:
        """
        移除动态标签
        
        Args:
            tag: 要移除的标签
            
        Returns:
            bool: 是否成功移除
        """
        if tag in self._dynamic_tags:
            self._dynamic_tags.remove(tag)
            # 清除缓存，强制重新计算
            self._shape_tags = None
            self._all_tokens = None
            return True
        return False
    
    def clear_dynamic_tags(self) -> None:
        """清空动态标签列表"""
        self._dynamic_tags.clear()
        # 清除缓存，强制重新计算
        self._shape_tags = None
        self._all_tokens = None
    
    def is_shape_tag(self, token: str) -> bool:
        """
        检查token是否为形状标签
        
        Args:
            token: 要检查的token
            
        Returns:
            bool: 是否为形状标签
        """
        return token in self.shape_tags
    
    def is_serial_token(self, token: str) -> bool:
        """
        检查token是否为有效的序列标签（1-240）
        
        Args:
            token: 要检查的token
            
        Returns:
            bool: 是否为有效的序列标签
        """
        match = self._serial_pattern.match(token)
        if not match:
            return False
        try:
            value = int(match.group(1))
            return 1 <= value <= self._max_serial_value
        except ValueError:
            return False
    
    def parse_serial_token(self, token: str) -> Optional[int]:
        """
        解析序列标签，提取其中的整数值
        
        Args:
            token: 序列标签，如 '<serial3>'
            
        Returns:
            Optional[int]: 解析出的整数值，如果不是有效的序列标签则返回None
        """
        match = self._serial_pattern.match(token)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= self._max_serial_value:
                    return value
            except ValueError:
                pass
        return None
    
    def create_serial_token(self, value: int) -> str:
        """
        根据整数值创建序列标签
        
        Args:
            value: 整数值（1-240）
            
        Returns:
            str: 序列标签，如 '<serial3>'
            
        Raises:
            ValueError: 如果值不在有效范围内
        """
        if not (1 <= value <= self._max_serial_value):
            raise ValueError(f"序列值必须在1到{self._max_serial_value}之间，当前值: {value}")
        return f'<serial{value}>'
    
    def is_command_token(self, token: str) -> bool:
        """
        检查token是否为命令标签
        
        Args:
            token: 要检查的token
            
        Returns:
            bool: 是否为命令标签
        """
        return token in self._command_tokens
    
    def is_valid_token(self, token: str) -> bool:
        """
        检查token是否为有效token
        
        Args:
            token: 要检查的token
            
        Returns:
            bool: 是否为有效token
        """
        return token in self.all_tokens
    
    def get_vocabulary_info(self) -> Dict:
        """
        获取当前词汇信息
        
        Returns:
            Dict: 词汇信息字典
        """
        return {
            'base_shape_tags': self._base_shape_tags,
            'dynamic_tags': self._dynamic_tags,
            'shape_tags': self.shape_tags,
            'serial_pattern': str(self._serial_pattern.pattern),
            'max_serial_value': self._max_serial_value,
            'serial_tokens_count': len(self.serial_tokens),
            'command_tokens': self._command_tokens,
            'all_tokens': self.all_tokens,
            'total_tags': len(self.shape_tags),
            'total_tokens': len(self.all_tokens)
        }
    
    def generate_mapping(self, x: int, y: int, z: int) -> Dict:
        """
        预先生成包含所有可能元素的映射表：
        - 使用集中定义的token列表（包括所有序列标签1-240）
        - 所有voxel坐标 (x,y,z) 范围为 (0~x-1, 0~y-1, 0~z-1)
        
        Args:
            x: 坐标X范围
            y: 坐标Y范围
            z: 坐标Z范围
            
        Returns:
            Dict: token到ID的映射表
        """
      
        mapping = {}
        next_id = 0
        
        # 使用集中定义的token列表
        for tag in self.all_tokens:
            if tag not in mapping:
                mapping[tag] = next_id
                next_id += 1
        # 坐标token
        for i in range(x):
            for j in range(y):
                for k in range(z):   
                    mapping[(i, j, k)] = next_id
                    next_id += 1
        
        return mapping



# 使用类级别的单例模式，避免模块导入问题
class TokenManagerSingleton:
    """TokenManager的单例包装器"""
    _instance = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = TokenManager()
        return cls._instance

def get_token_manager() -> TokenManager:
    """
    获取全局TokenManager实例（单例模式）
    
    Returns:
        TokenManager: 全局TokenManager实例
    """
    return TokenManagerSingleton()



def load_vocabulary(vocab_file: str) -> bool:
    """
    从词汇文件加载动态标签（全局函数，向后兼容）
    
    Args:
        vocab_file: 词汇文件路径
        
    Returns:
        bool: 是否加载成功
    """
    return get_token_manager().load_vocabulary(vocab_file)


def get_vocabulary_info() -> Dict:
    """
    获取当前词汇信息（全局函数，向后兼容）
    
    Returns:
        Dict: 词汇信息字典
    """
    return get_token_manager().get_vocabulary_info()


def generate_mapping(img_h: int, img_w: int) -> Dict:
    """
    预先生成包含所有可能元素的映射表（全局函数，向后兼容）
    
    Args:
        img_h: 图像高度
        img_w: 图像宽度
        
    Returns:
        Dict: token到ID的映射表
    """
    return get_token_manager().generate_mapping(img_h, img_w) 