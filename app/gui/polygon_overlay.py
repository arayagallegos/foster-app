from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath


class PolygonOverlay(QWidget):
    """
    Ventana top-level transparente que flota sobre el viewer VTK para capturar
    el dibujo del poligono de lazo. Se usa top-level porque en Windows el
    widget VTK es una ventana nativa (HWND) y los widgets Qt no-nativos quedan
    detras de ella aunque se llame raise_().
    """
    polygon_closed = pyqtSignal(list)  # list[tuple[int,int]]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setMouseTracking(True)
        self._vertices: list[QPoint] = []
        self._cursor_pos: QPoint = QPoint(0, 0)
        self._active = False
        self.hide()

    def activate(self) -> None:
        self._vertices.clear()
        self._cursor_pos = QPoint(0, 0)
        self._active = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.show()
        self.raise_()

    def deactivate(self) -> None:
        self._active = False
        self._vertices.clear()
        self.unsetCursor()
        self.hide()

    def mousePressEvent(self, event) -> None:
        if not self._active:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                len(self._vertices) >= 3
                and (event.pos() - self._vertices[0]).manhattanLength() < 15
            ):
                self._close_polygon()
                return
            self._vertices.append(event.pos())
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._close_polygon()

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._active:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._close_polygon()

    def mouseMoveEvent(self, event) -> None:
        if self._active:
            self._cursor_pos = event.pos()
            self.update()

    def _close_polygon(self) -> None:
        if len(self._vertices) < 3:
            self.deactivate()
            return
        verts = [(p.x(), p.y()) for p in self._vertices]
        self.deactivate()
        self.polygon_closed.emit(verts)

    def paintEvent(self, event) -> None:
        if not self._vertices:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if len(self._vertices) >= 3:
            fill_path = QPainterPath()
            fill_path.moveTo(self._vertices[0].x(), self._vertices[0].y())
            for v in self._vertices[1:]:
                fill_path.lineTo(v.x(), v.y())
            fill_path.closeSubpath()
            painter.fillPath(fill_path, QBrush(QColor(255, 255, 255, 30)))

        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        for i in range(len(self._vertices) - 1):
            a, b = self._vertices[i], self._vertices[i + 1]
            painter.drawLine(a.x(), a.y(), b.x(), b.y())

        if self._vertices:
            dashed_pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine)
            painter.setPen(dashed_pen)
            last = self._vertices[-1]
            cur = self._cursor_pos
            painter.drawLine(last.x(), last.y(), cur.x(), cur.y())

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        for v in self._vertices:
            painter.drawEllipse(v.x() - 5, v.y() - 5, 10, 10)

        if len(self._vertices) >= 3:
            if (self._cursor_pos - self._vertices[0]).manhattanLength() < 15:
                painter.setBrush(QBrush(QColor(255, 220, 0)))
                f = self._vertices[0]
                painter.drawEllipse(f.x() - 8, f.y() - 8, 16, 16)

        painter.end()
