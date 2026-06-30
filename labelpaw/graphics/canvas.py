# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsPathItem
from PySide6.QtGui import QPixmap, QPolygonF, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from labelpaw.graphics.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, HandleItem, PoseShape


class CanvasMode:
    EDIT = 0
    RECT = 1
    POLY = 2
    POINT = 3
    RBOX = 4

    @staticmethod
    def get_mode_name(mode):
        names = {1: "矩形", 2: "多边形", 3: "关键点", 4: "旋转框"}
        return names.get(mode, "未知")



def get_shape_scene_path(shape):
    from labelpaw.graphics.shapes import RectShape, PolyShape, RotatedRectShape, PoseShape, PointShape
    path = QPainterPath()
    if isinstance(shape, RectShape):
        path.addRect(shape.rect())
        return shape.mapToScene(path)
    elif isinstance(shape, PolyShape):
        path.addPolygon(shape.polygon())
        return shape.mapToScene(path)
    elif isinstance(shape, RotatedRectShape):
        path.addRect(QRectF(-shape.box_w/2, -shape.box_h/2, shape.box_w, shape.box_h))
        return shape.mapToScene(path)
    elif isinstance(shape, PoseShape):
        path.addRect(QRectF(-shape.box_w/2, -shape.box_h/2, shape.box_w, shape.box_h))
        return shape.mapToScene(path)
    elif isinstance(shape, PointShape):
        path.addEllipse(shape.rect())
        return shape.mapToScene(path)
    return shape.mapToScene(shape.shape())


class Canvas(QGraphicsScene):
    mouse_moved = Signal(int, int)
    shape_drawn = Signal(object)
    shape_double_clicked = Signal(object)
    state_changed = Signal()
    canvas_item_hovered = Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = CanvasMode.RECT
        self.img_item = None
        self.sam_client = None
        self.sam_enabled = False
        
        self.current_pose_template = None

        self.drawing = False
        self.start_pt = None
        self.temp_item = None
        self.poly_pts = []

        # 智能悬停提示图层
        self.sam_hover_item = None

        self.h_line = QGraphicsLineItem()
        self.v_line = QGraphicsLineItem()
        crosshair_pen = QPen(QColor(255, 255, 255, 200), 1, Qt.DashLine)
        self.h_line.setPen(crosshair_pen)
        self.v_line.setPen(crosshair_pen)
        self.h_line.setZValue(9999)
        self.v_line.setZValue(9999)
        self.h_line.hide()
        self.v_line.hide()
        self.addItem(self.h_line)
        self.addItem(self.v_line)

    def load_image(self, path):
        self.clear_shapes()
        pixmap = QPixmap(path)
        if self.img_item:
            self.removeItem(self.img_item)
        self.img_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.img_item)
        self.setSceneRect(pixmap.rect())
        # 十字线默认隐藏，由 CanvasView.enterEvent 在鼠标进入画布时显示
        self.h_line.hide()
        self.v_line.hide()

    def clear_shapes(self):
        # Clear all shape items from the canvas
        for item in self.items():
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                if getattr(item, 'is_temp', False):
                    if hasattr(self, 'pose_preview_item') and item == getattr(self, 'pose_preview_item', None):
                        self.pose_preview_item = None
                    if hasattr(self, 'sam_hover_item') and item == getattr(self, 'sam_hover_item', None):
                        self.sam_hover_item = None
                self.removeItem(item)

    def set_mode(self, mode):
        self.mode = mode
        self.cancel_drawing()
        for item in self.selectedItems():
            item.setSelected(False)

    def set_sam_enabled(self, enabled):
        self.sam_enabled = enabled
        if not enabled and self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

    def is_inside_image(self, pt):
        if not self.img_item: return False
        return self.sceneRect().contains(pt)

    def clamp_point(self, pt):
        rect = self.sceneRect()
        x = max(rect.left(), min(pt.x(), rect.right()))
        y = max(rect.top(), min(pt.y(), rect.bottom()))
        return QPointF(x, y)

    def update_crosshair(self, pt):
        if self.img_item:
            rect = self.sceneRect()
            x = max(rect.left(), min(pt.x(), rect.right()))
            y = max(rect.top(), min(pt.y(), rect.bottom()))
            self.h_line.setLine(rect.left(), y, rect.right(), y)
            self.v_line.setLine(x, rect.top(), x, rect.bottom())
            self.mouse_moved.emit(int(x), int(y))

    def mouseMoveEvent(self, event):
        pt = event.scenePos()
        self.update_crosshair(pt)
        super().mouseMoveEvent(event)
        clamped_pt = self.clamp_point(pt)

        # 预览骨架模板 (跟随鼠标)
        if self.mode == CanvasMode.POINT and self.current_pose_template:
            # 1. State Constraint: Hide preview if ANY item is selected (Edit Mode)
            has_selection = len(self.selectedItems()) > 0
            
            if has_selection:
                if hasattr(self, 'pose_preview_item') and self.pose_preview_item:
                    self.pose_preview_item.hide()
            else:
                # 检查鼠标下方是否有其他标注对象 (排除图片和线条)
                items_under_mouse = self.items(pt)
                hovering_on_shape = False
                for item in items_under_mouse:
                    from labelpaw.graphics.shapes import BaseShape, HandleItem, KeypointHandle
                    if isinstance(item, (BaseShape, HandleItem, KeypointHandle)) and not getattr(item, 'is_temp', False):
                        hovering_on_shape = True
                        break
                
                if hovering_on_shape:
                    if hasattr(self, 'pose_preview_item') and self.pose_preview_item:
                        self.pose_preview_item.hide()
                else:
                    if not hasattr(self, 'pose_preview_item') or not self.pose_preview_item:
                        from labelpaw.graphics.shapes import PoseShape
                        # Create a preview shape centered at 0,0 with default size
                        rect = QRectF(-50, -75, 100, 150)
                        self.pose_preview_item = PoseShape(rect, self.current_pose_template, is_temp=True)
                        self.pose_preview_item.setAcceptedMouseButtons(Qt.NoButton)
                        self.addItem(self.pose_preview_item)
                        self.pose_preview_item.setOpacity(0.6)
                    
                    self.pose_preview_item.show()
                    self.pose_preview_item.setPos(clamped_pt)

        # ---------------- SAM 智能辅助悬停 ----------------
        # 将 RBOX 加入 SAM 支持的模式列表
        is_sam_model = (getattr(self.sam_client, 'current_model_key', '') or '').startswith('sam')
        if self.sam_enabled and is_sam_model and not self.drawing and self.is_inside_image(pt) and self.mode in [CanvasMode.RECT, CanvasMode.POLY,
                                                                           CanvasMode.RBOX]:
            # 检查鼠标是否在已标注的非临时区域内部，如果是，则跳过 SAM 智能预选推理以提升性能并防止视觉遮挡
            from labelpaw.graphics.shapes import BaseShape
            hovering_inside_annotated = False
            for item in self.items(clamped_pt):
                if isinstance(item, BaseShape) and not getattr(item, 'is_temp', False):
                    hovering_inside_annotated = True
                    break
            
            if hovering_inside_annotated:
                if self.sam_hover_item:
                    self.removeItem(self.sam_hover_item)
                    self.sam_hover_item = None
                return

            if self.sam_client:
                self.sam_client.request_inference(clamped_pt.x(), clamped_pt.y(), is_click=False)
            return
        elif self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

        # ---------------- 常规绘图 ----------------
        if self.drawing and self.start_pt:
            rect = QRectF(min(self.start_pt.x(), clamped_pt.x()), min(self.start_pt.y(), clamped_pt.y()),
                          abs(clamped_pt.x() - self.start_pt.x()), abs(clamped_pt.y() - self.start_pt.y()))
            if self.temp_item: self.removeItem(self.temp_item)

            if self.mode == CanvasMode.RECT:
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.is_temp = True
                self.temp_item.setPen(QPen(QColor(28, 126, 214), 2, Qt.DashLine))

            # 手动拉框时，调用全新的 RotatedRectShape 参数格式
            elif self.mode == CanvasMode.RBOX:
                cx, cy = rect.center().x(), rect.center().y()
                w, h = max(1, rect.width()), max(1, rect.height())
                self.temp_item = RotatedRectShape(cx, cy, w, h, 0, is_temp=True)

            self.addItem(self.temp_item)

        elif self.mode == CanvasMode.POLY and not (self.sam_enabled and is_sam_model) and len(self.poly_pts) > 0:
            self.update_temp_poly(mouse_pos=clamped_pt)

    def handle_sam_result(self, poly_pts, rect_xywh, rect_obb, score, is_click):
        """处理来自 SAM 后台的推理结果，正确区分矩形、多边形 and 旋转框"""
        # 支持 RBOX
        if not self.sam_enabled or self.mode not in [CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX]:
            return

        if self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

        if not poly_pts or not rect_xywh:
            return

        # ---- 模式判断：矩形智能框 / 多边形点选 / 旋转框 ----
        if self.mode == CanvasMode.RECT:
            x, y, w, h = rect_xywh
            rect = QRectF(x, y, w, h)

            if is_click:
                shape = RectShape(rect)
                self.shape_drawn.emit(shape)
            else:
                path = QPainterPath()
                path.addRect(rect)
                self.sam_hover_item = QGraphicsPathItem(path)
                self.sam_hover_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                self.addItem(self.sam_hover_item)

        elif self.mode == CanvasMode.POLY:
            qpts = [QPointF(p[0], p[1]) for p in poly_pts]
            if is_click:
                shape = PolyShape(QPolygonF(qpts))
                self.shape_drawn.emit(shape)
            else:
                path = QPainterPath()
                path.addPolygon(QPolygonF(qpts))
                self.sam_hover_item = QGraphicsPathItem(path)
                self.sam_hover_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                self.addItem(self.sam_hover_item)

        # SAM 的 OBB 旋转框处理分支
        elif self.mode == CanvasMode.RBOX:
            if not rect_obb or len(rect_obb) < 5: return
            cx, cy, w, h, angle = rect_obb

            if is_click:
                shape = RotatedRectShape(cx, cy, w, h, angle)
                self.shape_drawn.emit(shape)
            else:
                # 构造旋转矩形在 scene 坐标下的 path
                temp_rbox = RotatedRectShape(cx, cy, w, h, angle, is_temp=True)
                local_path = QPainterPath()
                local_path.addRect(QRectF(-w/2, -h/2, w, h))
                path = temp_rbox.sceneTransform().map(local_path)
                self.sam_hover_item = QGraphicsPathItem(path)
                self.sam_hover_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
                self.addItem(self.sam_hover_item)

    def mousePressEvent(self, event):
        pt = event.scenePos()
        clamped_pt = self.clamp_point(pt)
        

        # 1. 优先检测是否点击在已有的图形或者手柄上，如果是，则直接进入编辑/选择状态
        clicked_item = None
        is_drawing_poly = (self.mode == CanvasMode.POLY and len(self.poly_pts) > 0)
        
        if not is_drawing_poly:
            items = self.items(clamped_pt)
            for item in items:
                from labelpaw.graphics.shapes import HandleItem, OBBHandle, KeypointHandle
                if isinstance(item, (HandleItem, OBBHandle, KeypointHandle)) and item.isVisible():
                    if not getattr(item.parentItem(), 'is_temp', False):
                        clicked_item = item
                        break
            if not clicked_item:
                for item in items:
                    from labelpaw.graphics.shapes import BaseShape
                    if isinstance(item, BaseShape):
                        if not getattr(item, 'is_temp', False):
                            clicked_item = item
                            break
                    elif item.parentItem() and isinstance(item.parentItem(), BaseShape):
                        if not getattr(item.parentItem(), 'is_temp', False):
                            clicked_item = item.parentItem()
                            break

        if clicked_item:
            # 首先让图元处理事件（如拖拽、调整大小等）
            super().mousePressEvent(event)
            
            # 自定义选中逻辑
            if event.button() == Qt.LeftButton:
                if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    # 取消其他所有图形的选中
                    for item in self.selectedItems():
                        if item != clicked_item and item != clicked_item.parentItem():
                            item.setSelected(False)
                    
                    # 选中当前被点击的图元（手柄则选中其父级图形）
                    from labelpaw.graphics.shapes import HandleItem, OBBHandle, KeypointHandle
                    if isinstance(clicked_item, (HandleItem, OBBHandle, KeypointHandle)):
                        parent = clicked_item.parentItem()
                        if parent:
                            parent.setSelected(True)
                    else:
                        clicked_item.setSelected(True)
            return

        # ---------------- SAM 确认生成 (仅限点击在空白处时触发) ----------------
        # 支持 RBOX, 但仅限 SAM 模型
        is_sam_model = (getattr(self.sam_client, 'current_model_key', '') or '').startswith('sam')
        if self.sam_enabled and is_sam_model and event.button() == Qt.LeftButton and self.mode in [CanvasMode.RECT, CanvasMode.POLY,
                                                                                   CanvasMode.RBOX]:
            if self.is_inside_image(pt):
                if self.mode in [CanvasMode.RECT, CanvasMode.RBOX]:
                    if self.sam_hover_item:
                        self.removeItem(self.sam_hover_item)
                        self.sam_hover_item = None
                    self.drawing = True
                    self.start_pt = clamped_pt
                elif self.sam_client:
                    self.sam_client.request_inference(clamped_pt.x(), clamped_pt.y(), is_click=True)
            return

        # 如果点击了空白处，取消所有选中状态 (退出编辑模式)
        if event.button() == Qt.LeftButton and not clicked_item:
            has_selection = len(self.selectedItems()) > 0
            if has_selection:
                for item in self.selectedItems():
                    item.setSelected(False)
                # 如果是点击空白处退出编辑模式，我们不应该继续向下执行绘制新图形的逻辑
                # 改动逻辑：仅在非矩形、非多边形、非OBB旋转框模式下才提前返回。
                # 在这些标注模式下，点击空白处可以直接开始绘制，无需额外点击一次来清除选中。
                if self.mode not in [CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX]:
                    return

        # ---------------- 常规绘图起点 ----------------
        if not self.is_inside_image(pt) and not self.drawing: return
        if event.button() == Qt.LeftButton:
            if self.mode in [CanvasMode.RECT, CanvasMode.RBOX]:
                self.drawing = True
                self.start_pt = clamped_pt
            elif self.mode == CanvasMode.POINT:
                if self.current_pose_template:
                    # 单击放置骨架模板
                    from labelpaw.graphics.shapes import PoseShape
                    rect = QRectF(-50, -75, 100, 150) # 默认大小缩小一半
                    shape = PoseShape(rect, self.current_pose_template)
                    shape.setPos(clamped_pt)
                    shape.is_temp = False
                    
                    # 默认创建后不选中（不进入编辑模式）
                    shape.setSelected(False)
                    shape._update_handle_visibility()
                    
                    self.shape_drawn.emit(shape)
                    
                    # 隐藏预览
                    if hasattr(self, 'pose_preview_item') and self.pose_preview_item:
                        self.removeItem(self.pose_preview_item)
                        self.pose_preview_item = None
                else:
                    shape = PointShape(clamped_pt)
                    self.shape_drawn.emit(shape)
            elif self.mode == CanvasMode.POLY:
                if len(self.poly_pts) > 2:
                    dist = ((clamped_pt.x() - self.poly_pts[0].x()) ** 2 + (
                            clamped_pt.y() - self.poly_pts[0].y()) ** 2) ** 0.5
                    if dist < 10:
                        self.finish_poly_shape()
                        return
                self.poly_pts.append(clamped_pt)
                self.update_temp_poly()
            elif self.mode == CanvasMode.POINT:
                pass # Already handled above
        elif event.button() == Qt.RightButton:
            if self.mode == CanvasMode.POLY and len(self.poly_pts) > 2:
                self.finish_poly_shape()

    def mouseReleaseEvent(self, event):
        # ---- 核心修复：防止 Qt 的选择切换 (toggle) 导致松开鼠标时图形被取消选中 ----
        # Qt 的 QGraphicsScene.mouseReleaseEvent 内部会对 "按下前已选中" 的图形执行取消选中。
        # 由于我们在 mousePressEvent 中先调用 super() 再手动 setSelected(True)，
        # Qt 误认为图形在按下前"已选中"，于是在释放时自动 toggle 取消。
        # 解决方案：在 super() 前快照当前选中的图形，调用 super() 后如果被清空则立即恢复。
        from labelpaw.graphics.shapes import BaseShape
        selected_before = [item for item in self.selectedItems()
                           if isinstance(item, BaseShape) and not getattr(item, 'is_temp', False)]

        # 阻止场景信号，防止 super() 中的 toggle 取消触发 selectionChanged 连锁反应
        self.blockSignals(True)
        super().mouseReleaseEvent(event)
        self.blockSignals(False)

        # 如果 super() 把我们选中的图形取消了，立即恢复
        if selected_before:
            selected_after = [item for item in self.selectedItems()
                              if isinstance(item, BaseShape) and not getattr(item, 'is_temp', False)]
            if not selected_after:
                for item in selected_before:
                    if item.scene() == self:
                        item.setSelected(True)

        is_sam_model = (getattr(self.sam_client, 'current_model_key', '') or '').startswith('sam')

        if event.button() == Qt.LeftButton and self.drawing and self.start_pt:
            self.drawing = False
            pt = self.clamp_point(event.scenePos())
            rect = QRectF(min(self.start_pt.x(), pt.x()), min(self.start_pt.y(), pt.y()),
                          abs(pt.x() - self.start_pt.x()), abs(pt.y() - self.start_pt.y()))
            if self.temp_item:
                self.removeItem(self.temp_item)
                self.temp_item = None

            if rect.width() > 5 and rect.height() > 5:
                if self.mode == CanvasMode.RECT:
                    self.shape_drawn.emit(RectShape(rect))
                elif self.mode == CanvasMode.RBOX:
                    cx, cy = rect.center().x(), rect.center().y()
                    w, h = rect.width(), rect.height()
                    self.shape_drawn.emit(RotatedRectShape(cx, cy, w, h, 0))
            elif self.sam_enabled and is_sam_model and self.mode in [CanvasMode.RECT, CanvasMode.RBOX] and self.sam_client:
                self.sam_client.request_inference(pt.x(), pt.y(), is_click=True)

            self.state_changed.emit()
            return

        if self.sam_enabled and is_sam_model: return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_item:
                pt = self.clamp_point(event.scenePos())
                rect = QRectF(min(self.start_pt.x(), pt.x()), min(self.start_pt.y(), pt.y()),
                              abs(pt.x() - self.start_pt.x()), abs(pt.y() - self.start_pt.y()))
                self.removeItem(self.temp_item)
                self.temp_item = None

                if rect.width() > 5 and rect.height() > 5:
                    if self.mode == CanvasMode.RECT:
                        self.shape_drawn.emit(RectShape(rect))
                    # 手动松开鼠标完成绘制时，实例化新的 RotatedRectShape
                    elif self.mode == CanvasMode.RBOX:
                        cx, cy = rect.center().x(), rect.center().y()
                        w, h = rect.width(), rect.height()
                        self.shape_drawn.emit(RotatedRectShape(cx, cy, w, h, 0))

        self.state_changed.emit()

    def mouseDoubleClickEvent(self, event):
        pt = event.scenePos()
        if not self.is_inside_image(pt): return

        for item in self.items(pt):
            from labelpaw.graphics.shapes import BaseShape, HandleItem
            if getattr(item, 'is_temp', False):
                continue
            if isinstance(item, BaseShape):
                self.shape_double_clicked.emit(item)
                return
            elif isinstance(item, HandleItem):
                parent = item.parentItem()
                if parent and not getattr(parent, 'is_temp', False):
                    self.shape_double_clicked.emit(parent)
                    return
            elif item.parentItem() and isinstance(item.parentItem(), BaseShape):
                parent = item.parentItem()
                if not getattr(parent, 'is_temp', False):
                    self.shape_double_clicked.emit(parent)
                    return

        is_sam_model = (getattr(self.sam_client, 'current_model_key', '') or '').startswith('sam')
        if event.button() == Qt.LeftButton and self.mode == CanvasMode.POLY and not (self.sam_enabled and is_sam_model) and len(
                self.poly_pts) > 2:
            self.finish_poly_shape()
        else:
            super().mouseDoubleClickEvent(event)

    def update_temp_poly(self, mouse_pos=None):
        display_pts = self.poly_pts.copy()
        if mouse_pos is not None: display_pts.append(mouse_pos)
        if len(display_pts) < 2:
            if self.temp_item: self.removeItem(self.temp_item); self.temp_item = None
            return
        if self.temp_item and isinstance(self.temp_item, PolyShape):
            self.temp_item.setPolygon(QPolygonF(display_pts))
        else:
            if self.temp_item: self.removeItem(self.temp_item)
            self.temp_item = PolyShape(QPolygonF(display_pts), is_temp=True)
            self.addItem(self.temp_item)

    def finish_poly_shape(self):
        shape = PolyShape(QPolygonF(self.poly_pts))
        self.poly_pts.clear()
        if self.temp_item:
            self.removeItem(self.temp_item)
            self.temp_item = None
        self.shape_drawn.emit(shape)

    def cancel_drawing(self):
        self.drawing = False
        self.poly_pts.clear()
        if self.temp_item:
            self.removeItem(self.temp_item)
            self.temp_item = None
        if self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None
        if hasattr(self, 'pose_preview_item') and self.pose_preview_item:
            self.removeItem(self.pose_preview_item)
            self.pose_preview_item = None

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key_Backspace or key == Qt.Key_Delete:
            for item in self.selectedItems():
                self.removeItem(item)
            self.state_changed.emit()
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            # 多边形顶点撤销已由 MainWindow.undo() 统一处理
            pass
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.mode == CanvasMode.POLY and len(self.poly_pts) > 2:
                self.finish_poly_shape()
        elif key == Qt.Key_Escape:
            self.cancel_drawing()
        elif key in [Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V]:
            items = self.selectedItems()
            if items and isinstance(items[0], RotatedRectShape):
                delta = 0
                if key == Qt.Key_Z:
                    delta = -5
                elif key == Qt.Key_X:
                    delta = -1
                elif key == Qt.Key_C:
                    delta = 1
                elif key == Qt.Key_V:
                    delta = 5
                if delta != 0:
                    item = items[0]
                    item.setRotation(item.rotation() + delta)
                    self.state_changed.emit()

        super().keyPressEvent(event)
