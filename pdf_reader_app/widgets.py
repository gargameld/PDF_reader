from pathlib import Path
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

import fitz
from PySide6.QtCore import Qt, QRectF, Signal, QTimer, QPoint, QPointF, QSize, QBuffer, QIODevice, QEvent
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QImage, QImageReader, QKeySequence, QPainter, QPen, QPixmap, QTextCursor, QTextDocument, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .constants import *
from .mime import builder_paths_from_mime_data, pdf_paths_from_mime_data
from .models import *

class PdfView(QGraphicsView):
    view_changed = Signal()
    clicked = Signal(int, float, float)
    right_clicked = Signal(int, float, float)
    link_click_requested = Signal(int, float, float)
    selection_line_requested = Signal(int, float, float)
    pdf_files_dropped = Signal(list)
    hover_moved = Signal(int, float, float, QPoint)
    hover_left = Signal()
    comment_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._page_pixmaps: Dict[int, QPixmap] = {}
        self._page_items: Dict[int, QGraphicsPixmapItem] = {}
        self._page_offsets: Dict[int, Tuple[float, float, float, float]] = {}
        self._page_order: List[int] = []
        self._page_tops: List[float] = []
        self._current_zoom: float = 1.0
        self._debug_rect: Optional[QGraphicsRectItem] = None
        self._debug_rect_timer = QTimer(self)
        self._debug_rect_timer.setSingleShot(True)
        self._debug_rect_timer.timeout.connect(self.clear_debug_box)
        self._selection_line_items: List[QGraphicsLineItem] = []
        self._search_highlight_items: List[QGraphicsRectItem] = []
        self._rectangle_hold_timer = QTimer(self)
        self._rectangle_hold_timer.setSingleShot(True)
        self._rectangle_hold_timer.timeout.connect(self._start_rectangle_selection)
        self._pending_rectangle_start: Optional[QPoint] = None
        self._rectangle_selecting = False
        self._rectangle_start_scene: Optional[QPointF] = None
        self._rectangle_start_page: Optional[int] = None
        self._rectangle_scene_rect: Optional[QRectF] = None
        self._selected_learning_rect: Optional[Tuple[int, float, float, float, float]] = None
        self._comment_overlays: List[Tuple[int, int, QRectF, str]] = []
        self._hovered_comment_index: Optional[int] = None
        self._comment_hover_timer = QTimer(self)
        self._comment_hover_timer.timeout.connect(self._check_comment_hover)
        self._comment_hover_timer.start(100)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.lightGray)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.viewport().setMouseTracking(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().installEventFilter(self)

        self.horizontalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())

    def dragEnterEvent(self, event) -> None:
        if pdf_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if pdf_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = pdf_paths_from_mime_data(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.pdf_files_dropped.emit(paths)
            return
        super().dropEvent(event)

    def set_document_images(
        self,
        pixmaps: List[QPixmap],
        preserve_zoom: bool = False,
        page_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        previous_zoom = self._current_zoom if preserve_zoom and self._page_pixmaps else 1.0

        self.scene().clear()
        self._page_pixmaps = {i + 1: pixmap for i, pixmap in enumerate(pixmaps)}
        self._page_items = {}
        self._page_offsets = {}
        self._page_order = []
        self._page_tops = []

        display_sizes = page_sizes or [(pixmap.width(), pixmap.height()) for pixmap in pixmaps]
        max_width = max((size[0] for size in display_sizes), default=0)
        y = 0.0
        for page_number, pixmap in self._page_pixmaps.items():
            page_w, page_h = display_sizes[page_number - 1]
            x = (max_width - page_w) / 2
            item = self.scene().addPixmap(pixmap)
            item.setPos(x, y)
            if pixmap.width() != page_w or pixmap.height() != page_h:
                sx = page_w / max(1, pixmap.width())
                sy = page_h / max(1, pixmap.height())
                item.setTransform(QTransform.fromScale(sx, sy))
            self._page_items[page_number] = item
            self._page_offsets[page_number] = (x, y, float(page_w), float(page_h))
            self._page_order.append(page_number)
            self._page_tops.append(y)
            y += page_h + PAGE_SPACING

        scene_height = max(0.0, y - PAGE_SPACING)
        self.scene().setSceneRect(QRectF(0, 0, max_width, scene_height))
        self.resetTransform()

        if abs(previous_zoom - 1.0) > 1e-9:
            self.scale(previous_zoom, previous_zoom)
            self._current_zoom = previous_zoom
        else:
            self._current_zoom = 1.0

        self._debug_rect = None
        self._selection_line_items = []
        self._search_highlight_items = []
        self._clear_rectangle_selection()
        self.view_changed.emit()

    def set_comment_overlays(self, comments: List[PdfComment]) -> None:
        self._comment_overlays = [
            (
                index,
                comment.page,
                QRectF(
                    comment.rect[0] * RENDER_SCALE,
                    comment.rect[1] * RENDER_SCALE,
                    max(1.0, (comment.rect[2] - comment.rect[0]) * RENDER_SCALE),
                    max(1.0, (comment.rect[3] - comment.rect[1]) * RENDER_SCALE),
                ),
                comment.color,
            )
            for index, comment in enumerate(comments)
        ]
        self._hovered_comment_index = None
        self.viewport().update()

    def _comment_scene_rect(self, page: int, page_rect: QRectF) -> Optional[QRectF]:
        offset = self._page_offsets.get(page)
        if offset is None:
            return None
        page_x, page_y, _page_w, _page_h = offset
        return QRectF(
            page_x + page_rect.x(),
            page_y + page_rect.y(),
            page_rect.width(),
            page_rect.height(),
        )

    def _comment_at_scene_pos(self, scene_pos: QPointF) -> Optional[int]:
        page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
        if page_pos is None:
            return None
        page, page_x, page_y = page_pos
        point = QPointF(page_x, page_y)
        for index, comment_page, rect, _color in self._comment_overlays:
            if comment_page == page and rect.contains(point):
                return index
        return None

    def _check_comment_hover(self) -> None:
        cursor_pos = self.viewport().mapFromGlobal(QCursor.pos())
        hovered = None
        if self.viewport().rect().contains(cursor_pos):
            scene_pos = self.mapToScene(cursor_pos)
            hovered = self._comment_at_scene_pos(scene_pos)
        if hovered == self._hovered_comment_index:
            return
        self._hovered_comment_index = hovered
        self.viewport().update()
        self.viewport().setCursor(Qt.PointingHandCursor if hovered is not None else Qt.ArrowCursor)

    def clear_page(self) -> None:
        self.scene().clear()
        self._page_pixmaps = {}
        self._page_items = {}
        self._page_offsets = {}
        self._page_order = []
        self._page_tops = []
        self._current_zoom = 1.0
        self._debug_rect = None
        self._selection_line_items = []
        self._search_highlight_items = []
        self._clear_rectangle_selection()
        self.view_changed.emit()

    def update_page_image(self, page: int, pixmap: QPixmap) -> None:
        item = self._page_items.get(page)
        if item is None:
            return
        self._page_pixmaps[page] = pixmap
        item.setPixmap(pixmap)
        item.setTransform(QTransform())

    def _clear_rectangle_selection(self) -> None:
        self._rectangle_hold_timer.stop()
        self._pending_rectangle_start = None
        self._rectangle_selecting = False
        self._rectangle_start_scene = None
        self._rectangle_start_page = None
        self._selected_learning_rect = None
        self._rectangle_scene_rect = None
        self.viewport().update()

    def _start_rectangle_selection(self) -> None:
        if self._pending_rectangle_start is None:
            return
        if not (QApplication.mouseButtons() & Qt.LeftButton):
            self._pending_rectangle_start = None
            return

        scene_pos = self.mapToScene(self._pending_rectangle_start)
        page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
        if page_pos is None:
            self._pending_rectangle_start = None
            return

        page, _page_x, _page_y = page_pos
        self._rectangle_selecting = True
        self._rectangle_start_scene = scene_pos
        self._rectangle_start_page = page
        self._selected_learning_rect = None
        self._rectangle_scene_rect = QRectF(scene_pos, scene_pos)
        self.viewport().update()

    def selected_learning_rect(self) -> Optional[Tuple[int, float, float, float, float]]:
        return self._selected_learning_rect

    def _update_rectangle_selection(self, scene_pos: QPointF) -> None:
        if (
            not self._rectangle_selecting
            or self._rectangle_start_scene is None
            or self._rectangle_start_page is None
        ):
            return

        page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
        if page_pos is None or page_pos[0] != self._rectangle_start_page:
            return

        rect = QRectF(self._rectangle_start_scene, scene_pos).normalized()
        self._rectangle_scene_rect = rect
        self.viewport().update()

    def _finish_rectangle_selection(self, scene_pos: QPointF) -> bool:
        if (
            not self._rectangle_selecting
            or self._rectangle_start_scene is None
            or self._rectangle_start_page is None
        ):
            return False

        page = self._rectangle_start_page
        offset = self._page_offsets.get(page)
        self._update_rectangle_selection(scene_pos)
        rect = self._rectangle_scene_rect.normalized() if self._rectangle_scene_rect is not None else QRectF()

        self._rectangle_hold_timer.stop()
        self._pending_rectangle_start = None
        self._rectangle_selecting = False
        self._rectangle_start_scene = None
        self._rectangle_start_page = None
        self.setDragMode(QGraphicsView.NoDrag)

        if offset is None or rect.width() < RECTANGLE_SELECT_MIN_SIZE or rect.height() < RECTANGLE_SELECT_MIN_SIZE:
            self._clear_rectangle_selection()
            self.view_changed.emit()
            return True

        page_x, page_y, page_w, page_h = offset
        x0 = max(0.0, min(rect.left() - page_x, page_w))
        y0 = max(0.0, min(rect.top() - page_y, page_h))
        x1 = max(0.0, min(rect.right() - page_x, page_w))
        y1 = max(0.0, min(rect.bottom() - page_y, page_h))
        if x1 - x0 < RECTANGLE_SELECT_MIN_SIZE or y1 - y0 < RECTANGLE_SELECT_MIN_SIZE:
            self._clear_rectangle_selection()
        else:
            self._selected_learning_rect = (page, x0, y0, x1, y1)
        self.view_changed.emit()
        self.viewport().update()
        return True

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setBrush(Qt.NoBrush)

        for index, page, page_rect, color_text in self._comment_overlays:
            scene_rect = self._comment_scene_rect(page, page_rect)
            if scene_rect is None:
                continue
            top_left = self.mapFromScene(scene_rect.topLeft())
            bottom_right = self.mapFromScene(scene_rect.bottomRight())
            viewport_rect = QRectF(QPointF(top_left), QPointF(bottom_right)).normalized()
            if not viewport_rect.intersects(QRectF(self.viewport().rect())):
                continue
            color = QColor(color_text)
            if not color.isValid():
                color = QColor("#2EAD4A")
            pen = QPen(color, 3 if index == self._hovered_comment_index else 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(viewport_rect)

        if self._rectangle_scene_rect is not None:
            rect = self._rectangle_scene_rect.normalized()
            top_left = self.mapFromScene(rect.topLeft())
            bottom_right = self.mapFromScene(rect.bottomRight())
            viewport_rect = QRectF(QPointF(top_left), QPointF(bottom_right)).normalized()
            if viewport_rect.width() >= 1 and viewport_rect.height() >= 1:
                painter.setPen(QPen(QColor("#1976D2"), 2))
                painter.drawRect(viewport_rect)
        painter.end()

    def clear_debug_box(self) -> None:
        if self._debug_rect is not None:
            try:
                self.scene().removeItem(self._debug_rect)
            except RuntimeError:
                pass
            self._debug_rect = None
            self.view_changed.emit()

    def show_debug_box(self, page: int, x: float, y: float, w: float, h: float) -> None:
        self._debug_rect_timer.stop()
        self.clear_debug_box()
        offset = self._page_offsets.get(page)
        if offset is None:
            return
        page_x, page_y, _page_w, _page_h = offset
        rect = QRectF(page_x + x - w / 2, page_y + y - h / 2, w, h)
        self._debug_rect = self.scene().addRect(rect, QPen(QColor("red"), 2))
        self._debug_rect_timer.start(DEBUG_BOX_DURATION_MS)

    def show_selection_lines(self, lines: List[SelectionLine]) -> None:
        for item in self._selection_line_items:
            self.scene().removeItem(item)
        self._selection_line_items = []

        pen = QPen(QColor("#1976D2"), 3)
        for line in lines:
            offset = self._page_offsets.get(line.page)
            if offset is None:
                continue
            page_x, page_y, page_w, page_h = offset
            if 0 <= line.y <= page_h:
                item = self.scene().addLine(page_x, page_y + line.y, page_x + page_w, page_y + line.y, pen)
                item.setZValue(10)
                self._selection_line_items.append(item)

    def clear_search_highlights(self) -> None:
        for item in self._search_highlight_items:
            try:
                self.scene().removeItem(item)
            except RuntimeError:
                pass
        self._search_highlight_items = []

    def show_search_highlights(self, hits: List[SearchHit], current_index: int = -1) -> None:
        self.clear_search_highlights()
        if not hits:
            return

        normal_pen = QPen(QColor(214, 160, 0, 190), 1)
        current_pen = QPen(QColor(213, 72, 0), 2)
        normal_brush = QColor(255, 230, 93, 95)
        current_brush = QColor(255, 174, 87, 135)

        for index, hit in enumerate(hits):
            offset = self._page_offsets.get(hit.page)
            if offset is None:
                continue
            page_x, page_y, _page_w, _page_h = offset
            x0, y0, x1, y1 = hit.rect
            rect = QRectF(
                page_x + x0 * RENDER_SCALE,
                page_y + y0 * RENDER_SCALE,
                max(1.0, (x1 - x0) * RENDER_SCALE),
                max(1.0, (y1 - y0) * RENDER_SCALE),
            )
            item = self.scene().addRect(rect, current_pen if index == current_index else normal_pen)
            item.setBrush(current_brush if index == current_index else normal_brush)
            item.setZValue(9 if index == current_index else 8)
            self._search_highlight_items.append(item)

    def base_pixmap(self, page: Optional[int] = None) -> Optional[QPixmap]:
        if page is not None:
            return self._page_pixmaps.get(page)
        current = self.current_visible_page()
        return self._page_pixmaps.get(current) if current is not None else None

    def page_at_scene_pos(self, scene_x: float, scene_y: float) -> Optional[Tuple[int, float, float]]:
        if not self._page_tops:
            return None

        index = bisect_right(self._page_tops, scene_y) - 1
        candidates: List[int] = []
        for candidate_index in (index, index + 1):
            if 0 <= candidate_index < len(self._page_order):
                candidates.append(self._page_order[candidate_index])

        nearest: Optional[Tuple[int, float, float]] = None
        nearest_distance = float("inf")
        for page in candidates:
            page_x, page_y, page_w, page_h = self._page_offsets[page]
            if not (page_x <= scene_x <= page_x + page_w):
                continue
            if page_y <= scene_y <= page_y + page_h:
                return page, scene_x - page_x, scene_y - page_y
            distance = min(abs(scene_y - page_y), abs(scene_y - (page_y + page_h)))
            if distance <= PAGE_SPACING and distance < nearest_distance:
                nearest = (page, scene_x - page_x, max(0.0, min(scene_y - page_y, page_h)))
                nearest_distance = distance
        return nearest

    def current_visible_page(self) -> Optional[int]:
        if not self._page_tops:
            return None
        viewport_center = self.mapToScene(self.viewport().rect().center())
        center_y = viewport_center.y()
        index = max(0, min(len(self._page_order) - 1, bisect_right(self._page_tops, center_y) - 1))
        best_page = self._page_order[index]
        best_distance = float("inf")
        for candidate_index in (index - 1, index, index + 1):
            if not (0 <= candidate_index < len(self._page_order)):
                continue
            page = self._page_order[candidate_index]
            _page_x, page_y, _page_w, page_h = self._page_offsets[page]
            if page_y <= center_y <= page_y + page_h + PAGE_SPACING:
                return page
            distance = abs(center_y - (page_y + page_h / 2))
            if distance < best_distance:
                best_page = page
                best_distance = distance
        return best_page

    def scroll_to_page(self, page: int, y: float = 0.0) -> None:
        offset = self._page_offsets.get(page)
        if offset is None:
            return
        page_x, page_y, page_w, _page_h = offset
        target = QPointF(page_x + page_w / 2, page_y + y)

        # Scrollbar values are affected by the current view transform. Align
        # the requested scene point through mapFromScene so jumps stay correct
        # at every zoom level.
        target_in_view = self.mapFromScene(target)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + target_in_view.y())

        target_in_view = self.mapFromScene(target)
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + target_in_view.x() - self.viewport().width() // 2
        )
        self.view_changed.emit()

    def page_scene_y(self, page: int) -> Optional[float]:
        offset = self._page_offsets.get(page)
        return offset[1] if offset is not None else None

    def zoom_factor(self) -> float:
        return self._current_zoom

    def _zoom_anchor_pos(self, anchor_pos: Optional[QPoint] = None) -> QPoint:
        if anchor_pos is not None and self.viewport().rect().contains(anchor_pos):
            return anchor_pos
        cursor_pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(cursor_pos):
            return cursor_pos
        return self.viewport().rect().center()

    def set_zoom_factor(self, factor: float, anchor_pos: Optional[QPoint] = None) -> None:
        if not self._page_pixmaps:
            return
        factor = max(0.2, min(6.0, factor))
        current = self._current_zoom
        if abs(current - factor) < 1e-9:
            return
        anchor = self._zoom_anchor_pos(anchor_pos)
        anchor_scene = self.mapToScene(anchor)
        scale_ratio = factor / current
        self.scale(scale_ratio, scale_ratio)
        self._current_zoom = factor
        shifted_anchor = self.mapFromScene(anchor_scene)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + shifted_anchor.x() - anchor.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + shifted_anchor.y() - anchor.y())
        self.view_changed.emit()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 1.15 if delta > 0 else 1 / 1.15
            self.set_zoom_factor(self._current_zoom * step, event.position().toPoint())
            event.accept()
            return

        super().wheelEvent(event)
        self.view_changed.emit()

    def _handle_native_zoom_gesture(self, event) -> bool:
        if event.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
            return False
        step = 1.0 + event.value()
        if step <= 0:
            return False
        self.set_zoom_factor(self._current_zoom * step, event.position().toPoint())
        event.accept()
        return True

    def nativeGestureEvent(self, event) -> None:
        if self._handle_native_zoom_gesture(event):
            return
        super().nativeGestureEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.viewport() and event.type() == QEvent.Type.NativeGesture:
            return self._handle_native_zoom_gesture(event)
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._rectangle_selecting:
            self._update_rectangle_selection(scene_pos)
            event.accept()
            return

        global_pos = self.viewport().mapToGlobal(event.position().toPoint())
        page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
        if page_pos is not None:
            page, page_x, page_y = page_pos
            self.hover_moved.emit(page, page_x, page_y, global_pos)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.set_zoom_factor(self._current_zoom * MOUSE_ZOOM_STEP, event.position().toPoint())
            event.accept()
            return

        if event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
            self._pending_rectangle_start = event.position().toPoint()
            self._rectangle_hold_timer.start(RECTANGLE_SELECT_HOLD_MS)

        if event.button() in (Qt.LeftButton, Qt.RightButton) and event.modifiers() & Qt.ControlModifier:
            scene_pos = self.mapToScene(event.position().toPoint())
            page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
            if page_pos is not None:
                page, page_x, page_y = page_pos
                self.link_click_requested.emit(page, page_x, page_y)
            event.accept()
            self.view_changed.emit()
            return

        # Shift+Click places a blue selection line only.
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            scene_pos = self.mapToScene(event.position().toPoint())
            page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
            if page_pos is not None:
                page, page_x, page_y = page_pos
                self.selection_line_requested.emit(page, page_x, page_y)
            event.accept()
            self.view_changed.emit()
            return
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_left.emit()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._rectangle_hold_timer.stop()
            if self._rectangle_selecting:
                scene_pos = self.mapToScene(event.position().toPoint())
                self._finish_rectangle_selection(scene_pos)
                event.accept()
                return
            self._pending_rectangle_start = None
            if event.modifiers() & Qt.ControlModifier:
                event.accept()
                self.view_changed.emit()
                return
            # Shift+Click was already handled on press. Avoid adding a duplicate
            # line on release.
            if event.modifiers() & Qt.ShiftModifier:
                event.accept()
                self.view_changed.emit()
                return
            scene_pos = self.mapToScene(event.position().toPoint())
            comment_index = self._comment_at_scene_pos(scene_pos)
            if comment_index is not None:
                self.comment_clicked.emit(comment_index)
                event.accept()
                self.view_changed.emit()
                return
            page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
            if page_pos is not None:
                page, page_x, page_y = page_pos
                self.clicked.emit(page, page_x, page_y)
        elif event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier:
                event.accept()
                self.view_changed.emit()
                return
            # Right-click means: read the equation reference under the cursor,
            # forget its cached destination, and search again from the PDF/TOC.
            # This is useful after a bad cached hit or after switching to a PDF
            # whose bookmarks/page offsets differ.
            scene_pos = self.mapToScene(event.position().toPoint())
            page_pos = self.page_at_scene_pos(scene_pos.x(), scene_pos.y())
            if page_pos is not None:
                page, page_x, page_y = page_pos
                self.right_clicked.emit(page, page_x, page_y)
            event.accept()
            self.view_changed.emit()
            return
        super().mouseReleaseEvent(event)
        self.view_changed.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.set_zoom_factor(self._current_zoom * MOUSE_ZOOM_STEP, event.position().toPoint())
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self.set_zoom_factor(self._current_zoom / MOUSE_ZOOM_STEP, event.position().toPoint())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._clear_rectangle_selection()
            self.view_changed.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Down:
            self._scroll_vertically(90)
            event.accept()
            return
        if event.key() == Qt.Key_Up:
            self._scroll_vertically(-90)
            event.accept()
            return
        super().keyPressEvent(event)

    def _scroll_vertically(self, delta: int) -> None:
        vertical_bar = self.verticalScrollBar()
        before = vertical_bar.value()
        target = max(vertical_bar.minimum(), min(vertical_bar.maximum(), before + delta))
        vertical_bar.setValue(target)

        self.view_changed.emit()

    def keyReleaseEvent(self, event) -> None:
        super().keyReleaseEvent(event)
        self.view_changed.emit()


class SearchLineEdit(QLineEdit):
    search_next_requested = Signal()
    search_previous_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                self.search_previous_requested.emit()
            else:
                self.search_next_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MathCommentEdit(QTextEdit):
    transform_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.transform_requested.emit()
            if getattr(self, "_math_transform_done", False):
                self._math_transform_done = False
                event.accept()
                return
        super().keyPressEvent(event)


class CommentDialog(QDialog):
    WORD_EQUATION_MACROS: Dict[str, str] = {
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\zeta": "ζ",
        r"\eta": "η",
        r"\theta": "θ",
        r"\iota": "ι",
        r"\kappa": "κ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\nu": "ν",
        r"\xi": "ξ",
        r"\pi": "π",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\phi": "φ",
        r"\chi": "χ",
        r"\psi": "ψ",
        r"\omega": "ω",
        r"\Gamma": "Γ",
        r"\Delta": "Δ",
        r"\Theta": "Θ",
        r"\Lambda": "Λ",
        r"\Xi": "Ξ",
        r"\Pi": "Π",
        r"\Sigma": "Σ",
        r"\Phi": "Φ",
        r"\Psi": "Ψ",
        r"\Omega": "Ω",
        r"\infty": "∞",
        r"\partial": "∂",
        r"\nabla": "∇",
        r"\pm": "±",
        r"\mp": "∓",
        r"\times": "×",
        r"\cdot": "·",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\propto": "∝",
        r"\rightarrow": "→",
        r"\leftarrow": "←",
        r"\Rightarrow": "⇒",
        r"\sum": "∑",
        r"\int": "∫",
    }

    def __init__(
        self,
        macros: Dict[str, str],
        parent: Optional[QWidget] = None,
        initial_text: str = "",
        initial_color: str = "#2EAD4A",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Comment")
        self.resize(620, 420)
        self.macros = dict(macros)
        self._color = QColor(initial_color if QColor(initial_color).isValid() else "#2EAD4A")

        layout = QVBoxLayout(self)
        self.comment_edit = MathCommentEdit()
        self.comment_edit.transform_requested.connect(self.transform_expression_before_cursor)
        if "<html" in initial_text.lower() or "<span" in initial_text.lower() or "<sup" in initial_text.lower():
            self.comment_edit.setHtml(initial_text)
        else:
            self.comment_edit.setPlainText(initial_text)
        layout.addWidget(self.comment_edit, 1)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Comment color:"))
        self.color_button = QPushButton()
        self.color_button.clicked.connect(self.choose_color)
        color_row.addWidget(self.color_button)
        color_row.addStretch()
        layout.addLayout(color_row)
        self._update_color_button()

        hint = QLabel(r"Type math directly in the comment. Space or Command+R converts the expression before the cursor, e.g. e^x, x_i, \frac{a}{b}, \sqrt{x}, \alpha.")
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        insert_action = QAction("Insert Equation", self)
        insert_action.setShortcuts([QKeySequence("Meta+R"), QKeySequence("Ctrl+R")])
        insert_action.triggered.connect(self.transform_expression_before_cursor)
        self.addAction(insert_action)

    def _expand_macros(self, text: str) -> str:
        expanded = text
        for macro, replacement in sorted(self.macros.items(), key=lambda item: len(item[0]), reverse=True):
            if macro:
                expanded = expanded.replace(macro, replacement)
        return expanded

    @staticmethod
    def _read_balanced(text: str, index: int, opener: str, closer: str) -> Tuple[str, int]:
        depth = 0
        start = index + 1
        pos = index
        while pos < len(text):
            char = text[pos]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:pos], pos + 1
            pos += 1
        return text[start:], len(text)

    def _read_math_atom(self, text: str, index: int) -> Tuple[str, int]:
        if index >= len(text):
            return "", index
        char = text[index]
        if char == "{":
            inner, pos = self._read_balanced(text, index, "{", "}")
            return self._render_math_inner(inner), pos
        if char == "(":
            inner, pos = self._read_balanced(text, index, "(", ")")
            return "(" + self._render_math_inner(inner) + ")", pos
        if char == "\\":
            match = re.match(r"\\[A-Za-z]+", text[index:])
            if match:
                command = match.group(0)
                return html.escape(self.WORD_EQUATION_MACROS.get(command, command)), index + len(command)
        return html.escape(char), index + 1

    def _read_required_group_html(self, text: str, index: int) -> Tuple[str, int]:
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == "{":
            inner, pos = self._read_balanced(text, index, "{", "}")
            return self._render_math_inner(inner), pos
        return self._read_math_atom(text, index)

    def _render_math_inner(self, text: str) -> str:
        out: List[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                out.append("&nbsp;")
                index += 1
                continue
            if char == "^":
                atom, index = self._read_required_group_html(text, index + 1)
                out.append(f"<sup>{atom}</sup>")
                continue
            if char == "_":
                atom, index = self._read_required_group_html(text, index + 1)
                out.append(f"<sub>{atom}</sub>")
                continue
            if text.startswith(r"\frac", index):
                numerator, pos = self._read_required_group_html(text, index + 5)
                denominator, index = self._read_required_group_html(text, pos)
                out.append(
                    "<span style='display:inline-block; vertical-align:middle; text-align:center;'>"
                    f"<span style='display:block; border-bottom:1px solid currentColor; padding:0 3px;'>{numerator}</span>"
                    f"<span style='display:block; padding:0 3px;'>{denominator}</span>"
                    "</span>"
                )
                continue
            if text.startswith(r"\sqrt", index):
                radicand, index = self._read_required_group_html(text, index + 5)
                out.append(f"<span style='white-space:nowrap;'>√<span style='border-top:1px solid currentColor;'>{radicand}</span></span>")
                continue
            if text.startswith(r"\hat", index):
                body, index = self._read_required_group_html(text, index + 4)
                out.append(f"<span style='text-decoration:overline;'>{body}</span>")
                continue
            if text.startswith(r"\bar", index) or text.startswith(r"\overline", index):
                command_len = 4 if text.startswith(r"\bar", index) else 9
                body, index = self._read_required_group_html(text, index + command_len)
                out.append(f"<span style='text-decoration:overline;'>{body}</span>")
                continue
            if char == "\\":
                match = re.match(r"\\[A-Za-z]+", text[index:])
                if match:
                    command = match.group(0)
                    out.append(html.escape(self.WORD_EQUATION_MACROS.get(command, command)))
                    index += len(command)
                    continue
            out.append(html.escape(char))
            index += 1
        return "".join(out)

    def _render_equation_html(self, text: str) -> str:
        expanded = self._expand_macros(text)
        body = self._render_math_inner(expanded)
        return (
            "<span style='font-family: Times New Roman, STIX Two Math, serif; "
            "font-size: 18pt; white-space: nowrap;'>"
            f"{body}</span>"
        )

    def _update_color_button(self) -> None:
        self.color_button.setText(self._color.name())
        self.color_button.setStyleSheet(f"background: {self._color.name()}; color: white;")

    def choose_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Comment Color")
        if color.isValid():
            self._color = color
            self._update_color_button()

    @staticmethod
    def _should_render_math(text: str) -> bool:
        return bool(
            text
            and (
                "\\" in text
                or "^" in text
                or "_" in text
                or re.search(r"[A-Za-z0-9]\{", text)
            )
        )

    @staticmethod
    def _expression_bounds_before_cursor(text: str) -> Tuple[int, int]:
        end = len(text)
        start = end
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        return start, end

    def transform_expression_before_cursor(self) -> None:
        cursor = self.comment_edit.textCursor()
        cursor_position = cursor.position()
        plain = self.comment_edit.toPlainText()
        before = plain[:cursor_position]
        start, end = self._expression_bounds_before_cursor(before)
        raw = before[start:end]
        if not self._should_render_math(raw):
            self.comment_edit._math_transform_done = False
            return

        equation = self._render_equation_html(raw)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.insertHtml(equation + "<span>&nbsp;</span>")
        self.comment_edit.setTextCursor(cursor)
        self.comment_edit._math_transform_done = True

    def comment_text(self) -> str:
        return self.comment_edit.toHtml()

    def has_comment_text(self) -> bool:
        return bool(self.comment_edit.toPlainText().strip())

    def comment_color(self) -> str:
        return self._color.name()


class MacroSettingsDialog(QDialog):
    def __init__(self, macros: Dict[str, str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Equation Macros")
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        info = QLabel(r"One custom macro per line: \e = exp(. Built-ins include \alpha, \sum, \int, \frac{}, \sqrt{}, ^ and _.")
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        self.editor = QPlainTextEdit()
        self.editor.setPlainText(
            "\n".join(f"{macro} = {replacement}" for macro, replacement in sorted(macros.items()))
        )
        layout.addWidget(self.editor, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def macros(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for raw_line in self.editor.toPlainText().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            macro, replacement = line.split("=", 1)
            macro = macro.strip()
            replacement = replacement.strip()
            if macro:
                out[macro] = replacement
        return out


class DocumentBuilderList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setViewMode(QListWidget.IconMode)
        self.setMovement(QListWidget.Snap)
        self.setResizeMode(QListWidget.Adjust)
        self.setWrapping(True)
        self.setSpacing(14)
        self.setIconSize(QSize(110, 150))
        self.setGridSize(QSize(150, 205))
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setStyleSheet(
            "QListWidget { background: #f6f7f8; border: 1px solid #cfd5dc; }"
            "QListWidget::item { padding: 8px; }"
            "QListWidget::item:selected { background: #d8e8ff; color: #111; }"
        )

    def dragEnterEvent(self, event) -> None:
        if builder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if builder_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = builder_paths_from_mime_data(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)
            return
        super().dropEvent(event)


class DocumentBuilderDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create New Document")
        self.resize(980, 720)
        self.saved_path: Optional[str] = None
        self._thumbnail_cache: Dict[Tuple[str, int], QPixmap] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        title = QLabel("Create New Document")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        top_row.addWidget(title)
        top_row.addStretch()

        add_button = QPushButton("Add Files")
        add_button.clicked.connect(self.add_files_from_dialog)
        top_row.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_pages)
        top_row.addWidget(remove_button)
        layout.addLayout(top_row)

        instructions = QLabel("Drag PDFs or images here. Reorder pages by dragging the rectangles.")
        instructions.setStyleSheet("color: #4b5563;")
        layout.addWidget(instructions)

        self.page_list = DocumentBuilderList(self)
        self.page_list.files_dropped.connect(self.add_files)
        self.page_list.model().rowsMoved.connect(lambda *_args: QTimer.singleShot(0, self._renumber_pages))
        layout.addWidget(self.page_list, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Save PDF...")
        buttons.accepted.connect(self.save_document)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_files_from_dialog(self) -> None:
        patterns = "Documents and Images (*.pdf *.png *.jpg *.jpeg *.bmp *.gif *.heic *.heif *.tif *.tiff *.webp)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Pages", "", patterns)
        self.add_files(paths)

    def add_files(self, paths: List[str]) -> None:
        added = 0
        for path in paths:
            file_path = Path(path)
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                added += self._add_pdf_pages(path)
            elif suffix in SUPPORTED_DOCUMENT_BUILDER_EXTENSIONS:
                if self._add_image_page(path):
                    added += 1

        if added:
            self._renumber_pages()

    def _add_pdf_pages(self, path: str) -> int:
        try:
            doc = fitz.open(path)
        except Exception as exc:
            QMessageBox.warning(self, "Add PDF", f"Could not open PDF:\n{path}\n\n{exc}")
            return 0

        count = 0
        try:
            for page_index in range(len(doc)):
                entry = {"kind": "pdf", "path": path, "page_index": page_index}
                label = f"{Path(path).name}\nPage {page_index + 1}"
                self._add_page_item(label, entry, self._pdf_page_thumbnail(path, page_index))
                count += 1
        finally:
            doc.close()
        return count

    def _add_image_page(self, path: str) -> bool:
        image = self._read_image(path)
        if image.isNull():
            QMessageBox.warning(self, "Add Image", f"Could not open image:\n{path}")
            return False
        self._add_page_item(f"{Path(path).name}\nImage", {"kind": "image", "path": path}, self._image_thumbnail(image))
        return True

    def _add_page_item(self, label: str, entry: dict, pixmap: QPixmap) -> None:
        item = QListWidgetItem(QIcon(pixmap), label)
        item.setData(Qt.UserRole, entry)
        item.setTextAlignment(Qt.AlignCenter)
        self.page_list.addItem(item)

    def _pdf_page_thumbnail(self, path: str, page_index: int) -> QPixmap:
        key = (path, page_index)
        if key in self._thumbnail_cache:
            return self._thumbnail_cache[key]

        pixmap = self._placeholder_thumbnail("PDF")
        try:
            doc = fitz.open(path)
            try:
                page = doc.load_page(page_index)
                page_pix = page.get_pixmap(matrix=fitz.Matrix(0.22, 0.22), alpha=False)
                image = QImage(
                    page_pix.samples,
                    page_pix.width,
                    page_pix.height,
                    page_pix.stride,
                    QImage.Format_RGB888,
                ).copy()
                pixmap = QPixmap.fromImage(image).scaled(110, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            finally:
                doc.close()
        except Exception:
            pass

        self._thumbnail_cache[key] = pixmap
        return pixmap

    def _read_image(self, path: str) -> QImage:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        return reader.read()

    def _image_thumbnail(self, image: QImage) -> QPixmap:
        return QPixmap.fromImage(image).scaled(110, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _placeholder_thumbnail(self, text: str) -> QPixmap:
        pixmap = QPixmap(110, 150)
        pixmap.fill(QColor("#ffffff"))
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#8a8f98")))
        painter.drawRect(0, 0, 109, 149)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
        painter.end()
        return pixmap

    def _renumber_pages(self) -> None:
        for index in range(self.page_list.count()):
            item = self.page_list.item(index)
            label = item.text()
            label = re.sub(r"^\d+\.\s+", "", label)
            item.setText(f"{index + 1}. {label}")

    def remove_selected_pages(self) -> None:
        for item in self.page_list.selectedItems():
            self.page_list.takeItem(self.page_list.row(item))
        self._renumber_pages()

    def ordered_entries(self) -> List[dict]:
        entries: List[dict] = []
        for index in range(self.page_list.count()):
            item = self.page_list.item(index)
            entry = item.data(Qt.UserRole)
            if isinstance(entry, dict):
                entries.append(dict(entry))
        return entries

    def save_document(self) -> None:
        entries = self.ordered_entries()
        if not entries:
            QMessageBox.information(self, "Create Document", "Add at least one page before saving.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save New PDF", "", "PDF Files (*.pdf)")
        if not path:
            return
        if Path(path).suffix.lower() != ".pdf":
            path += ".pdf"

        try:
            self._write_pdf(entries, path)
        except Exception as exc:
            QMessageBox.critical(self, "Create Document", f"Could not save PDF:\n{exc}")
            return

        self.saved_path = path
        self.accept()

    def _write_pdf(self, entries: List[dict], output_path: str) -> None:
        out = fitz.open()
        try:
            for entry in entries:
                path = str(entry.get("path", ""))
                if entry.get("kind") == "pdf":
                    source = fitz.open(path)
                    try:
                        page_index = int(entry.get("page_index", 0))
                        out.insert_pdf(source, from_page=page_index, to_page=page_index)
                    finally:
                        source.close()
                elif entry.get("kind") == "image":
                    image = self._read_image(path)
                    if image.isNull():
                        raise ValueError(f"Could not open image: {path}")
                    page_width, page_height = self._image_page_size(image)
                    page = out.new_page(width=page_width, height=page_height)
                    buffer = QBuffer()
                    buffer.open(QIODevice.WriteOnly)
                    export_image = image.convertToFormat(QImage.Format_RGB888)
                    if not export_image.save(buffer, "PNG") or buffer.size() == 0:
                        raise ValueError(f"Could not encode image for PDF: {path}")
                    page.insert_image(page.rect, stream=bytes(buffer.data()))
            out.save(output_path)
        finally:
            out.close()

    def _image_page_size(self, image: QImage) -> Tuple[float, float]:
        image_width = max(1, image.width())
        image_height = max(1, image.height())
        if image_width >= image_height:
            max_width, max_height = 792.0, 612.0
        else:
            max_width, max_height = 612.0, 792.0
        scale = min(max_width / image_width, max_height / image_height, 1.0)
        return max(1.0, image_width * scale), max(1.0, image_height * scale)
