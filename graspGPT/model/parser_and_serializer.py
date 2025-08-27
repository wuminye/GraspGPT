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
#
# 顶层结构：零个或多个 ITEM，最后以字面量 'end' 结束
# ---------------------------------------------------------------------------

SEQ    → ITEM* 'end'                       # ← 入口规则

ITEM   → SB | UNSEG | INPAINT | AMODAL                       # ITEM 四种形态

# —— Labeled segment ---------------------------------------------------------
SB     → TAG CB+                           # TAG 后至少一个坐标块

# —— Un‑labeled segment ----------------------------------------
UNSEG  → 'unlabel' CB*                    # 开始：可带 0⁺ 坐标块
          'segment' SB*                    # 中段：可嵌 0⁺ SB
          'endunseg'                       # 结束

# —— Shape Inpainting ----------------------------------------
INPAINT  → 'fragment' CB*                    # 开始：可带 0⁺ 坐标块
          'inpaint' SB*                    # 中段：可嵌 0⁺ SB, 最多1个SB
          'endinpaint'                       # 结束


# —— Amodal Prediction ----------------------------------------
AMODAL  → 'tagfragment' SB*                    # 开始：可带 0⁺ SB
          'amodal' SB*                    # 中段：可嵌 0⁺ SB, 最多1个SB
          'endamodal'                       # 结束

# —— 坐标块 ——----------------------------------------------------------------
CB     → COORD                             # 单坐标
        | COORD SERIAL                     # 或坐标 + 序列标签

# —— 终结符组 ——--------------------------------------------------------------
TAG    → 'unknow'  | 'object' INT            # 形状关键字

# 其余终结符：COORD、SERIAL 在词法层直接产生

# ---------------------------------------------------------------------------
# 词法记号（由词法层生成，非 CFG 一部分）
# ---------------------------------------------------------------------------

COORD  → '(' INT ',' INT ')'               # 行、列元组 (row,col)

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
`Seq`, `SB`, `UNSEG`, and `CB` nodes.  See the `__main__` section at
the bottom for a runnable demo.
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
class SB:
    tag: str  # 形状标签，可以是基础标签（'circle', 'square'）或动态加载的COCO类别
    cbs: List[CB]

    def __str__(self) -> str:
        inner = ", ".join(map(str, self.cbs))
        return f"SB(tag={self.tag}, [{inner}])"


@dataclass
class UNSEG:
    cbs: List[CB]
    sbs: List[SB]

    def __str__(self) -> str:
        cbs_str = ", ".join(map(str, self.cbs))
        sbs_str = ", ".join(map(str, self.sbs))
        return f"UNSEG(cbs=[{cbs_str}], sbs=[{sbs_str}])"


@dataclass
class INPAINT:
    cbs: List[CB]
    sb: Optional[SB] = None  # 只允许0或1个SB

    def __str__(self) -> str:
        cbs_str = ", ".join(map(str, self.cbs))
        sb_str = str(self.sb) if self.sb else ""
        return f"INPAINT(cbs=[{cbs_str}], sb={sb_str})"


@dataclass
class AMODAL:
    fragment_sbs: List[SB]  # 'tagfragment' 后的SB列表
    amodal_sbs: List[SB]    # 'amodal' 后的SB列表

    def __str__(self) -> str:
        fragment_str = ", ".join(map(str, self.fragment_sbs))
        amodal_str = ", ".join(map(str, self.amodal_sbs))
        return f"AMODAL(fragment_sbs=[{fragment_str}], amodal_sbs=[{amodal_str}])"


@dataclass
class Seq:
    items: List[Union[SB, UNSEG, INPAINT, AMODAL]]

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
        items: List[Union[SB, UNSEG, INPAINT]] = []
        while self.current() not in (None, 'end'):
            try:
                items.append(self._parse_item())
            except ParseError as e:
                # 出错后，忽略后续 token，直接跳出循环
                print(f"ParseError: {e}")
                break
        # 无论是否出错，都尝试 consume 掉 'end'，保证 tree 合法
        if self.current() == 'end':
            self.advance()
        return Seq(items)

    # ITEM ------------------------------------------------------------------
    def _parse_item(self) -> Union[SB, UNSEG, INPAINT, AMODAL]:
        tok = self.current()
        # 检查是否是形状标签（包括基础标签和动态标签）
        if tok in get_token_manager().shape_tags:
            return self._parse_sb()
        if tok == 'unlabel':
            return self._parse_unseg()
        if tok == 'fragment':
            return self._parse_inpaint()
        if tok == 'tagfragment':
            return self._parse_amodal()
        raise ParseError(f"[parse_item] Unexpected token {tok!r} at position {self.pos}")

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
        self.expect('unlabel')
        cbs: List[CB] = []
        while self._starts_cb(self.current()):
            cbs.append(self._parse_cb())
        self.expect('segment')
        sbs: List[SB] = []
        while self.current() in get_token_manager().shape_tags:
            sbs.append(self._parse_sb())
        self.expect('endunseg')
        return UNSEG(cbs, sbs)

    # INPAINT -----------------------------------------------------------------
    def _parse_inpaint(self) -> INPAINT:
        self.expect('fragment')
        cbs: List[CB] = []
        while self._starts_cb(self.current()):
            cbs.append(self._parse_cb())
        self.expect('inpaint')
        sb: Optional[SB] = None
        if self.current() in get_token_manager().shape_tags:
            sb = self._parse_sb()
            # 检查是否还有多余的SB
            if self.current() in get_token_manager().shape_tags:
                raise ParseError("INPAINT最多只能有1个SB")
        self.expect('endinpaint')
        return INPAINT(cbs, sb)

    # AMODAL -----------------------------------------------------------------
    def _parse_amodal(self) -> AMODAL:
        self.expect('tagfragment')
        fragment_sbs: List[SB] = []
        while self.current() in get_token_manager().shape_tags:
            fragment_sbs.append(self._parse_sb())
        self.expect('amodal')
        amodal_sbs: List[SB] = []
        while self.current() in get_token_manager().shape_tags:
            amodal_sbs.append(self._parse_sb())
        self.expect('endamodal')
        return AMODAL(fragment_sbs, amodal_sbs)

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


# ---------------------------------------------------------------------------
# Serializer implementation (reverse of Parser)
# ---------------------------------------------------------------------------
class Serializer:
    """Serializes AST nodes back to flat token lists."""
    
    @staticmethod
    def serialize(seq: Seq) -> List[Token]:
        """Serialize a Seq AST back to a flat token list."""
        tokens: List[Token] = []
        
        # Serialize all items in the sequence
        for item in seq.items:
            if isinstance(item, SB):
                tokens.extend(Serializer._serialize_sb(item))
            elif isinstance(item, UNSEG):
                tokens.extend(Serializer._serialize_unseg(item))
            elif isinstance(item, INPAINT):
                tokens.extend(Serializer._serialize_inpaint(item))
            elif isinstance(item, AMODAL):
                tokens.extend(Serializer._serialize_amodal(item))
        
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
    def _serialize_unseg(unseg: UNSEG) -> List[Token]:
        """Serialize an UNSEG node to tokens."""
        tokens: List[Token] = ['unlabel']  # Start with 'unlabel'
        
        # Add all CB tokens
        for cb in unseg.cbs:
            tokens.extend(Serializer._serialize_cb(cb))
        
        # Add 'segment' separator
        tokens.append('segment')
        
        # Add all SB tokens
        for sb in unseg.sbs:
            tokens.extend(Serializer._serialize_sb(sb))
        
        tokens.append('endunseg')
        return tokens
    
    @staticmethod
    def _serialize_inpaint(inpaint: INPAINT) -> List[Token]:
        tokens: List[Token] = ['fragment']
        for cb in inpaint.cbs:
            tokens.extend(Serializer._serialize_cb(cb))
        tokens.append('inpaint')
        if inpaint.sb is not None:
            tokens.extend(Serializer._serialize_sb(inpaint.sb))
        tokens.append('endinpaint')
        return tokens
    
    @staticmethod
    def _serialize_amodal(amodal: AMODAL) -> List[Token]:
        tokens: List[Token] = ['tagfragment']
        for sb in amodal.fragment_sbs:
            tokens.extend(Serializer._serialize_sb(sb))
        tokens.append('amodal')
        for sb in amodal.amodal_sbs:
            tokens.extend(Serializer._serialize_sb(sb))
        tokens.append('endamodal')
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
    tokens1 = ['object01', (10, 20, 30), '<serial5>', 'object02', (40, 50, 60), 'end']
    print(f"输入tokens: {tokens1}")
    
    parser1 = Parser(tokens1)
    ast1 = parser1.parse()
    print(f"解析结果:\n{ast1}")
    
    # 序列化测试
    serialized1 = Serializer.serialize(ast1)
    print(f"序列化结果: {serialized1}")
    print(f"往返测试: {'通过' if tokens1 == serialized1 else '失败'}\n")
    
    # 测试案例2: UNSEG结构
    print("测试2: UNSEG结构")
    tokens2 = ['unlabel', (5, 15, 25), (35, 45, 55), 'segment', 'object01', (65, 75, 85), 'endunseg', 'end']
    print(f"输入tokens: {tokens2}")
    
    parser2 = Parser(tokens2)
    ast2 = parser2.parse()
    print(f"解析结果:\n{ast2}")
    
    serialized2 = Serializer.serialize(ast2)
    print(f"序列化结果: {serialized2}")
    print(f"往返测试: {'通过' if tokens2 == serialized2 else '失败'}\n")
    
    # 测试案例3: INPAINT结构
    print("测试3: INPAINT结构")
    tokens3 = ['fragment', (1, 2, 3), (4, 5, 6), 'inpaint', 'object01', (7, 8, 9), 'endinpaint', 'end']
    print(f"输入tokens: {tokens3}")
    
    parser3 = Parser(tokens3)
    ast3 = parser3.parse()
    print(f"解析结果:\n{ast3}")
    
    serialized3 = Serializer.serialize(ast3)
    print(f"序列化结果: {serialized3}")
    print(f"往返测试: {'通过' if tokens3 == serialized3 else '失败'}\n")
    
    # 测试案例4: AMODAL结构
    print("测试4: AMODAL结构")
    tokens4 = ['tagfragment', 'object02', (10, 11, 12), 'amodal', 'object01', (13, 14, 15), 'endamodal', 'end']
    print(f"输入tokens: {tokens4}")
    
    parser4 = Parser(tokens4)
    ast4 = parser4.parse()
    print(f"解析结果:\n{ast4}")
    
    serialized4 = Serializer.serialize(ast4)
    print(f"序列化结果: {serialized4}")
    print(f"往返测试: {'通过' if tokens4 == serialized4 else '失败'}\n")
    
    # 测试案例5: 混合复杂结构
    print("测试5: 混合复杂结构")
    tokens5 = [
        'object01', (1, 1, 1), '<serial10>',
        'unlabel', (2, 2, 2), 'segment', 'object02', (3, 3, 3), 'endunseg',
        'fragment', (4, 4, 4), 'inpaint', 'endinpaint',
        'tagfragment', 'object03', (5, 5, 5), 'amodal', 'endamodal',
        'end'
    ]
    print(f"输入tokens: {tokens5}")
    
    parser5 = Parser(tokens5)
    ast5 = parser5.parse()
    print(f"解析结果:\n{ast5}")
    
    serialized5 = Serializer.serialize(ast5)
    print(f"序列化结果: {serialized5}")
    print(f"往返测试: {'通过' if tokens5 == serialized5 else '失败'}\n")
    
    # 测试案例6: 错误处理
    print("测试6: 错误处理测试")
    invalid_tokens = ['object01', 'invalid_coordinate', 'end']
    print(f"输入无效tokens: {invalid_tokens}")
    
    parser6 = Parser(invalid_tokens)
    ast6 = parser6.parse()  # 解析器会处理错误并返回部分结果
    print(f"错误处理后的解析结果:\n{ast6}")
    
    # 测试序列标签验证
    print("\n测试7: 序列标签验证")
    valid_serial = '<serial100>'
    invalid_serial = '<serial300>'
    print(f"'{valid_serial}' 是否有效: {token_manager.is_serial_token(valid_serial)}")
    print(f"'{invalid_serial}' 是否有效: {token_manager.is_serial_token(invalid_serial)}")
    
    # 测试形状标签验证
    print(f"'object01' 是否为形状标签: {token_manager.is_shape_tag('object01')}")
    print(f"'invalid_shape' 是否为形状标签: {token_manager.is_shape_tag('invalid_shape')}")
    
    print("\n=== 所有测试完成 ===")
  