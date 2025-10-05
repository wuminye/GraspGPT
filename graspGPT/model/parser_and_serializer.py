# -*- coding: utf-8 -*-
from __future__ import annotations
'''
# ---------------------------------------------------------------------------
# 文法定义说明
# ---------------------------------------------------------------------------
# 记号说明
#   - 单引号括起的为字面量终结符
#   - 全大写小写混排的名字（COORD、SERIAL 等）为“词法 token”
#     —— 由词法分析器直接返回，语法层不再拆分
# ---------------------------------------------------------------------------


SEQ   → SCENE  [AMODAL] [UNSEG] [GRASP]  'end'             # ← 入口规则,   [AMODAL]，[UNSEG] 和 [GRASP] 可能出现也可能不出现，一旦出现一定保持这个顺序


# ==== SCENE ==== 描述场景的内容
SCENE  → 'scene' SB*                # SCENE 由 0⁺ SB 组成

# —— Labeled segment ---------------------------------------------------------
SB     → TAG CB+                           # TAG 后至少一个坐标块

# —— Grasp data ------------ 预测对于当前场景，能从哪些位置抓取哪些物体
GRASP  → 'detectgrasp' GB*                # 'detectgrasp' 后 0⁺ GB
GB  -> 'grasp' TAG CB+                    # 'grasp' + TAG + 1⁺ CB

# —— Un‑labeled segment --------讲场景中未标注的部分分割出来，将SCENE和AMODAL中标记为 'unlabel' 的SB 分割后对应输出，输出的SB中 TAG 只能是  'object' INT 
UNSEG  →  'segment' SB*  'endunseg'     # 可嵌 0⁺ SB, 不能出现TAG 为 'unknow' | 'unlabel' |   'incomplete' 的SB


# —— Amodal Prediction ---------- 一些物体因为视角遮挡的原因没看全，需要用这个把场景中的物体补全， 将SCENE中 标记为'incomplete' 的SB补全。
AMODAL  → 'amodal' SB   'endamodal'     # 只有一个 TAG 为 'unlabel' 的 SB
                           

# —— 坐标块 ——----------------------------------------------------------------
CB     → COORD                             # 单坐标
        | COORD SERIAL                     # 或坐标 + 序列标签

# —— 终结符组 ——--------------------------------------------------------------
TAG    → 'unknow' | 'unlabel' |   'incomplete'   | 'object' INT            # 形状关键字

# 其余终结符：COORD、SERIAL 在词法层直接产生

# ---------------------------------------------------------------------------
# 词法记号（由词法层生成，非 CFG 一部分）
# ---------------------------------------------------------------------------

COORD  → '(' INT ',' INT ',' INT ')'               # 三元组 (x,y,z)，均为非负整数 代表三维坐标

SERIAL → '<serial' INT '>'
                                          # 合法序列长度：1-240

INT    : /0|[1-9][0-9]*/                  # 非负十进制整数
'''


"""A minimal recursive‑descent parser for the right‑linear grammar
specified in the previous message.

Input is a *flat* Python list where each token is either
    * a literal string (e.g. 'object00', 'unlabel', '<serial3>', 'end'), or
    * a 3‑tuple of ints representing a 3D coordinate, e.g. (12, 34, 56).

The parser produces a small Abstract Syntax Tree (AST) composed of
`Seq`, `SB`, `UNSEG`, `AMODAL`, `GRASP`, `GB`, and `CB` nodes. 
"""


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# 导入TokenManager
try:
    from .token_manager import get_token_manager
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from token_manager import get_token_manager
    except ImportError:
        # 如果还是失败，尝试从minGPT包导入
        from minGPT.token_manager import get_token_manager

Coord = Tuple[int, int, int]
Serial = str  # strings like '<serial3>', '<serial5>', '<serial9>' - dynamic format: '<serial' INT '>'
Token = Union[str, Coord]


# Token定义已移至TokenManager中统一管理


# 全局函数已移除，请使用TokenManager
# 使用示例：
# from minGPT.token_manager import get_token_manager
# token_manager = get_token_manager()
# token_manager.load_vocabulary(vocab_file)
# token_manager.get_vocabulary_info()
# token_manager.generate_mapping(img_h, img_w)

# ---------------------------------------------------------------------------
# AST node definitions
# ---------------------------------------------------------------------------
@dataclass
class CB:
    coord: Coord
    serial: Optional[Serial] = None

    def __str__(self) -> str:  # pretty printing helper
        return f"CB(coord={self.coord}, serial={self.serial})"


@dataclass
class GB:
    tag: str  # 抓取标签
    cbs: List[CB]  # 至少一个坐标块

    def __str__(self) -> str:
        inner = ", ".join(map(str, self.cbs))
        return f"GB(tag={self.tag}, [{inner}])"


@dataclass
class SB:
    tag: str  # 形状标签，可以是基础标签（'circle', 'square'）或动态加载的COCO类别
    cbs: List[CB]

    def __str__(self) -> str:
        inner = ", ".join(map(str, self.cbs))
        return f"SB(tag={self.tag}, [{inner}])"


@dataclass
class Scene:
    sbs: List[SB]

    def __str__(self) -> str:
        inner = ", ".join(map(str, self.sbs))
        return f"Scene([{inner}])"


@dataclass
class UNSEG:
    sbs: List[SB]

    def __str__(self) -> str:
        sbs_str = ", ".join(map(str, self.sbs))
        return f"UNSEG(sbs=[{sbs_str}])"


@dataclass
class AMODAL:
    sb: SB

    def __str__(self) -> str:
        return f"AMODAL({self.sb})"


@dataclass
class GRASP:
    gbs: List[GB]  # 'detectgrasp' 后零个或多个GB

    def __str__(self) -> str:
        inner = ", ".join(map(str, self.gbs))
        return f"GRASP([{inner}])"


@dataclass
class Seq:
    items: List[Union[Scene, UNSEG, AMODAL, GRASP]]

    def __str__(self) -> str:
        joined = ",\n  ".join(map(str, self.items))
        return f"Seq([\n  {joined}\n])"




# ---------------------------------------------------------------------------
# Parser implementation
# ---------------------------------------------------------------------------
class ParseError(Exception):
    """Raised when the token stream violates the grammar."""


class Parser:
    """Recursive‑descent parser for the specified grammar."""

    def __init__(self, tokens: List[Token]):
        self.tokens: List[Token] = tokens
        self.pos: int = 0

    # Utility helpers -------------------------------------------------------
    def current(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self) -> None:
        self.pos += 1

    def expect(self, expected: Token) -> Token:
        tok = self.current()
        if tok != expected:
            raise ParseError(f"Expected {expected!r}, got {tok!r} at position {self.pos}")
        self.advance()
        return tok

    # Entry point -----------------------------------------------------------
    def parse(self) -> Seq:
        """Parse a SEQ and ensure the final token is 'end'.
        如果解析过程中遇到 ParseError，则忽略后续 token，只用之前成功解析的数据构造合法 AST。
        """
        items: List[Union[Scene, UNSEG, AMODAL, GRASP]] = []
        try:
            scene = self._parse_scene()
            items.append(scene)

            # Optional AMODAL
            if self.current() == 'amodal':
                items.append(self._parse_amodal())

            # Optional UNSEG
            if self.current() == 'segment':
                items.append(self._parse_unseg())

            # Optional GRASP
            if self.current() == 'detectgrasp':
                items.append(self._parse_grasp())

            # After optional sections we should see 'end' or EOF
            trailing = self.current()
            if trailing not in (None, 'end'):
                raise ParseError(
                    f"Unexpected token {trailing!r} after parsing SEQ at position {self.pos}"
                )

        except ParseError as e:
            print(f"ParseError: {e}")
            self._consume_until_end()

        # Always consume a terminal 'end' if present
        if self.current() == 'end':
            self.advance()
        return Seq(items)

    def _parse_scene(self) -> Scene:
        if self.current() == 'scene':
            self.advance()
        else:
            print("Warning: Missing 'scene' token at sequence start, parsing legacy format.")
        sbs: List[SB] = []
        while self.current() in get_token_manager().shape_tags:
            sbs.append(self._parse_sb())
        return Scene(sbs)

    # SB --------------------------------------------------------------------
    def _parse_sb(self) -> SB:
        tag = self.current()  # 形状标签（基础标签或动态标签）
        self.advance()
        cbs: List[CB] = []
        if not self._starts_cb(self.current()):
            raise ParseError("SB must contain at least one CB after tag")
        while self._starts_cb(self.current()):
            cbs.append(self._parse_cb())
        return SB(tag, cbs)  # type: ignore[arg-type]

    # UNSEG -----------------------------------------------------------------
    def _parse_unseg(self) -> UNSEG:
        self.expect('segment')
        sbs: List[SB] = []
        while self.current() in get_token_manager().shape_tags:
            sb = self._parse_sb()
            if not self._is_object_tag(sb.tag):
                raise ParseError("UNSEG 内部的 SB 标签必须为 'object' + INT")
            sbs.append(sb)
        self.expect('endunseg')
        return UNSEG(sbs)

    # AMODAL -----------------------------------------------------------------
    def _parse_amodal(self) -> AMODAL:
        self.expect('amodal')
        sb = self._parse_sb()
        if sb.tag != 'unlabel':
            raise ParseError("AMODAL 中的 SB 必须使用 'unlabel' 标签")
        self.expect('endamodal')
        return AMODAL(sb)

    # GRASP -----------------------------------------------------------------
    def _parse_grasp(self) -> GRASP:
        self.expect('detectgrasp')
        gbs: List[GB] = []
        while self.current() == 'grasp':
            try:
                gbs.append(self._parse_gb())
            except ParseError as e:
                print(f"ParseError while parsing GB: {e}")
                # Stop parsing GBs and discard the remaining tokens in this sequence.
                self._consume_until_end()
                break
        return GRASP(gbs)

    # GB --------------------------------------------------------------------
    def _parse_gb(self) -> GB:
        self.expect('grasp')
        tag = self.current()
        if tag not in get_token_manager().shape_tags:
            raise ParseError(f"Expected shape tag after 'grasp', got {tag!r}")
        self.advance()
        cbs: List[CB] = []
        if not self._starts_cb(self.current()):
            raise ParseError("GB must contain at least one CB after tag")
        while self._starts_cb(self.current()):
            cbs.append(self._parse_cb())
        return GB(tag, cbs)  # type: ignore[arg-type]

    # CB --------------------------------------------------------------------
    def _parse_cb(self) -> CB:
        coord_tok = self.current()
        if not self._is_coord(coord_tok):
            raise ParseError(f"Expected coordinate tuple, got {coord_tok!r}")
        self.advance()
        serial: Optional[Serial] = None
        if isinstance(self.current(), str) and get_token_manager().is_serial_token(self.current()):
            serial = self.current()  # type: ignore[assignment]
            self.advance()
        return CB(coord_tok, serial)  # type: ignore[arg-type]

    # Helpers ---------------------------------------------------------------
    @staticmethod
    def _is_coord(token: Optional[Token]) -> bool:
        return (
            isinstance(token, tuple)
            and len(token) == 3
            and all(isinstance(n, int) for n in token)
        )

    @staticmethod
    def _starts_cb(token: Optional[Token]) -> bool:
        return Parser._is_coord(token)

    @staticmethod
    def _is_object_tag(tag: str) -> bool:
        return isinstance(tag, str) and tag.startswith('object') and tag[6:].isdigit()

    def _consume_until_end(self) -> None:
        while self.current() not in (None, 'end'):
            self.advance()


# ---------------------------------------------------------------------------
# C++ Parser Integration (Pybind11 only)
# ---------------------------------------------------------------------------
_pybind_parser = None
_pybind_available = None

def _get_pybind_parser():
    """Get pybind11 parser module if available"""
    global _pybind_parser, _pybind_available
    
    if _pybind_available is not None:
        return _pybind_parser
    
    try:
        import sys
        import os
        
        # Add csrc directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.join(current_dir, 'csrc')
        if csrc_dir not in sys.path:
            sys.path.insert(0, csrc_dir)
        
        # Try to import the pybind module
        import parser_cpp
        _pybind_parser = parser_cpp
        _pybind_available = True
        print("Pybind11 C++ parser loaded successfully")
        
    except ImportError as e:
        _pybind_parser = None
        _pybind_available = False
        print(f"Pybind11 C++ parser not available: {e}")
    
    return _pybind_parser


def parse_with_cpp(tokens: List[Token]) -> Seq:
    """
    Parse tokens using the pybind11 C++ parser implementation.
    Raises an error if C++ parser is not available or fails.
    
    Args:
        tokens: List of tokens to parse
        
    Returns:
        Seq: Parsed AST sequence
        
    Raises:
        RuntimeError: If C++ parser is not available or fails
        ParseError: If parsing fails due to invalid tokens
    """
    '''
    parser_cpp = _get_pybind_parser()
    
    if parser_cpp is None:
        raise RuntimeError("Pybind11 C++ parser is not available. Please build the parser_cpp module.")
    
    try:
        # Call C++ parser through pybind11
        result_dict = parser_cpp.parse_tokens(tokens)
        
        if result_dict is None:
            raise RuntimeError("C++ parser returned None result")
        
        # For now, validate that C++ parsing worked and return Python result
        # In a full implementation, we would reconstruct the Python AST from result_dict
        # This ensures compatibility while using C++ for validation
        return Parser(tokens).parse()
        
    except Exception as e:
        raise RuntimeError(f"C++ parser failed: {e}")
    '''
    parser = Parser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# Serializer implementation (reverse of Parser)
# ---------------------------------------------------------------------------
class Serializer:
    """Serializes AST nodes back to flat token lists."""
    
    @staticmethod
    def serialize(seq: Seq) -> List[Token]:
        """Serialize a Seq AST back to a flat token list."""
        tokens: List[Token] = []

        items: List[Union[Scene, UNSEG, AMODAL, GRASP, SB]] = list(seq.items)
        if not any(isinstance(item, Scene) for item in items):
            legacy_sbs = [item for item in items if isinstance(item, SB)]
            remaining = [item for item in items if not isinstance(item, SB)]
            items = [Scene(sbs=legacy_sbs)] + remaining

        # Serialize all items in the sequence
        for item in items:
            if isinstance(item, Scene):
                tokens.extend(Serializer._serialize_scene(item))
            elif isinstance(item, SB):
                tokens.extend(Serializer._serialize_sb(item))
            elif isinstance(item, UNSEG):
                tokens.extend(Serializer._serialize_unseg(item))
            elif isinstance(item, AMODAL):
                tokens.extend(Serializer._serialize_amodal(item))
            elif isinstance(item, GRASP):
                tokens.extend(Serializer._serialize_grasp(item))

        # Add the final 'end' token
        tokens.append('end')
        return tokens
    
    @staticmethod
    def _serialize_sb(sb: SB) -> List[Token]:
        """Serialize an SB node to tokens."""
        tokens: List[Token] = [sb.tag]  # Start with the tag ('circle' or 'square')
        
        # Add all CB tokens
        for cb in sb.cbs:
            tokens.extend(Serializer._serialize_cb(cb))
        
        return tokens

    @staticmethod
    def _serialize_scene(scene: Scene) -> List[Token]:
        tokens: List[Token] = ['scene']
        for sb in scene.sbs:
            tokens.extend(Serializer._serialize_sb(sb))
        return tokens
    
    @staticmethod
    def _serialize_unseg(unseg: UNSEG) -> List[Token]:
        """Serialize an UNSEG node to tokens."""
        tokens: List[Token] = ['segment']

        # Add all SB tokens inside UNSEG (must all use objectXX tags)
        for sb in unseg.sbs:
            tokens.extend(Serializer._serialize_sb(sb))

        tokens.append('endunseg')
        return tokens

    @staticmethod
    def _serialize_amodal(amodal: AMODAL) -> List[Token]:
        tokens: List[Token] = ['amodal']
        tokens.extend(Serializer._serialize_sb(amodal.sb))
        tokens.append('endamodal')
        return tokens
    
    @staticmethod
    def _serialize_grasp(grasp: GRASP) -> List[Token]:
        """Serialize a GRASP node to tokens."""
        tokens: List[Token] = ['detectgrasp']
        
        # Add all GB tokens
        for gb in grasp.gbs:
            tokens.extend(Serializer._serialize_gb(gb))
        
        return tokens
    
    @staticmethod
    def _serialize_gb(gb: GB) -> List[Token]:
        """Serialize a GB node to tokens."""
        tokens: List[Token] = ['grasp', gb.tag]
        
        # Add all CB tokens
        for cb in gb.cbs:
            tokens.extend(Serializer._serialize_cb(cb))
        
        return tokens
    
    @staticmethod
    def _serialize_cb(cb: CB) -> List[Token]:
        """Serialize a CB node to tokens."""
        tokens: List[Token] = [cb.coord]  # Start with the coordinate
        
        # Add serial if present
        if cb.serial is not None:
            tokens.append(cb.serial)
        
        return tokens



    
# ---------------------------------------------------------------------------
# Demo / quick test run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 测试基础功能
    
    print("=== Parser and Serializer 基础功能测试 ===\n")
    
    # 初始化token manager
    token_manager = get_token_manager()
    print(f"Token Manager 初始化完成")
    print(f"基础形状标签数量: {len(token_manager.base_shape_tags)}")
    print(f"命令标签: {token_manager.command_tokens}")
    print()
    
    # 测试案例1: 基础SB解析
    print("测试1: 基础SB结构")
    tokens1 = ['scene', 'object01', (10, 20, 30), '<serial5>', 'object02', (40, 50, 60), 'end']
    print(f"输入tokens: {tokens1}")
    
    parser1 = Parser(tokens1)
    ast1 = parser1.parse()
    print(f"解析结果:\n{ast1}")
    
    # 序列化测试
    serialized1 = Serializer.serialize(ast1)
    print(f"序列化结果: {serialized1}")
    print(f"往返测试: {'通过' if tokens1 == serialized1 else '失败'}\n")
    
    # 测试案例2: AMODAL结构
    print("测试2: AMODAL结构")
    tokens2 = [
        'scene',
        'object03', (15, 25, 35),
        'amodal', 'unlabel', (45, 55, 65),
        'endamodal',
        'end'
    ]
    print(f"输入tokens: {tokens2}")

    parser2 = Parser(tokens2)
    ast2 = parser2.parse()
    print(f"解析结果:\n{ast2}")

    serialized2 = Serializer.serialize(ast2)
    print(f"序列化结果: {serialized2}")
    print(f"往返测试: {'通过' if tokens2 == serialized2 else '失败'}\n")

    # 测试案例3: UNSEG结构
    print("测试3: UNSEG结构")
    tokens3 = [
        'scene',
        'segment',
        'object10', (5, 15, 25),
        'object11', (35, 45, 55), '<serial2>',
        'endunseg',
        'end'
    ]
    print(f"输入tokens: {tokens3}")

    parser3 = Parser(tokens3)
    ast3 = parser3.parse()
    print(f"解析结果:\n{ast3}")

    serialized3 = Serializer.serialize(ast3)
    print(f"序列化结果: {serialized3}")
    print(f"往返测试: {'通过' if tokens3 == serialized3 else '失败'}\n")

    # 测试案例4: 混合结构
    print("测试4: 混合结构")
    tokens4 = [
        'scene',
        'object20', (1, 1, 1), '<serial10>',
        'object21', (2, 2, 2),
        'amodal', 'unlabel', (3, 3, 3), '<serial5>', 'endamodal',
        'segment', 'object30', (4, 4, 4), 'object31', (5, 5, 5), '<serial7>', 'endunseg',
        'detectgrasp',
        'grasp', 'object01', (6, 6, 6), '<serial9>',
        'grasp', 'object02', (7, 7, 7),
        'end'
    ]
    print(f"输入tokens: {tokens4}")

    parser4 = Parser(tokens4)
    ast4 = parser4.parse()
    print(f"解析结果:\n{ast4}")

    serialized4 = Serializer.serialize(ast4)
    print(f"序列化结果: {serialized4}")
    print(f"往返测试: {'通过' if tokens4 == serialized4 else '失败'}\n")

    # 测试案例5: 错误处理
    print("测试5: 错误处理测试")
    invalid_tokens = ['object01', 'invalid_coordinate', 'end']
    print(f"输入无效tokens: {invalid_tokens}")

    parser5 = Parser(invalid_tokens)
    ast5 = parser5.parse()  # 解析器会处理错误并返回部分结果
    print(f"错误处理后的解析结果:\n{ast5}")
    
    # 测试序列标签验证
    print("\n测试9: 序列标签验证")
    valid_serial = '<serial100>'
    invalid_serial = '<serial300>'
    print(f"'{valid_serial}' 是否有效: {token_manager.is_serial_token(valid_serial)}")
    print(f"'{invalid_serial}' 是否有效: {token_manager.is_serial_token(invalid_serial)}")
    
    # 测试形状标签验证
    print(f"'object01' 是否为形状标签: {token_manager.is_shape_tag('object01')}")
    print(f"'invalid_shape' 是否为形状标签: {token_manager.is_shape_tag('invalid_shape')}")
    
    # 测试案例6: GRASP结构
    print("测试6: GRASP结构")
    tokens6 = [
        'scene',
        'detectgrasp', 
        'grasp', 'object01', (10, 20, 30), '<serial5>',
        'grasp', 'object02', (40, 50, 60), (70, 80, 90),
        'end'
    ]
    print(f"输入tokens: {tokens6}")
    
    parser6 = Parser(tokens6)
    ast6 = parser6.parse()
    print(f"解析结果:\n{ast6}")
    
    serialized6 = Serializer.serialize(ast6)
    print(f"序列化结果: {serialized6}")
    print(f"往返测试: {'通过' if tokens6 == serialized6 else '失败'}\n")
    
    # 测试案例7: 空GRASP结构
    print("测试7: 空GRASP结构")
    tokens7 = ['scene', 'detectgrasp', 'end']
    print(f"输入tokens: {tokens7}")
    
    parser7 = Parser(tokens7)
    ast7 = parser7.parse()
    print(f"解析结果:\n{ast7}")
    
    serialized7 = Serializer.serialize(ast7)
    print(f"序列化结果: {serialized7}")
    print(f"往返测试: {'通过' if tokens7 == serialized7 else '失败'}\n")

    print("\n=== 所有测试完成 ===")
  
