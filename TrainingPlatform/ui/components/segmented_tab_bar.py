# -*- coding: utf-8 -*-
"""
Segmented Navigation Tab Bar
Directly uses MonkeyQt's built-in SegmentedTabBar (MkSegmented) component with full backward-compatibility.
"""
from typing import Optional, List, Union, Dict, Any
from PySide6.QtWidgets import QWidget
from monkeyqt import SegmentedTabBar as _MkSegmentedTabBar

class SegmentedTabBar(_MkSegmentedTabBar):
    def __init__(
        self,
        items: Optional[List[Union[str, Dict[str, Any]]]] = None,
        radius: Optional[int] = None,
        pill: bool = True,
        size: str = "default",
        fill: bool = False,
        animated: bool = True,
        indicator_style: str = "auto",
        parent: Optional[QWidget] = None,
        size_mode: Optional[str] = None,
        **kwargs
    ):
        actual_size = size_mode or size or "default"
        super().__init__(
            items=items,
            radius=radius,
            pill=pill,
            size=actual_size,
            fill=fill,
            animated=animated,
            indicator_style=indicator_style,
            parent=parent,
            **kwargs
        )

__all__ = ["SegmentedTabBar"]
