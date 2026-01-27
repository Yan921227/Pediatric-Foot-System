#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gait Demo GUI (PyQt5)
====================
把 Demo_window.py 的展示介面，與 Mix.py 的「辨識姿態/步態」功能結合成同一個視窗作品。

重點：
- 不更動 Mix.py 的辨識/疊字/計算邏輯：全部沿用 Mix.py 的四個 Viewer。
- 只做「UI 整合」：把原本會獨立開視窗的 Viewer，嵌到 DemoWindow 左側的影片顯示區。
- 為了避免影片播完就整個 App 退出（Mix.py 原本會 qApp.quit()），這裡用子類別覆寫 _finish()：
  只停止播放並釋放資源，不退出 DemoWindow。

注意：
- 本版本已移除「自動辨識」相關的 UI 與功能（僅保留手動選擇分析模式）。

使用方式：
- 將本檔案與 Mix.py 放在同一個資料夾
- 執行：python Demo_window_no_auto.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

from PyQt5 import QtWidgets, QtCore, QtGui

# Mix.py 必須與本檔同資料夾（或在 PYTHONPATH 內）
import Mix as mix


# -----------------------------
# 影像顯示：維持比例縮放 QLabel
# -----------------------------
class AspectRatioLabel(QtWidgets.QLabel):
    """
    讓 QLabel 自動把 Pixmap 以 KeepAspectRatio 縮放到可用空間。
    這樣影片顯示區大小可變（不同影片解析度也沒問題）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pix: Optional[QtGui.QPixmap] = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:  # type: ignore[override]
        self._pix = pixmap
        self._update_scaled()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pix is None or self._pix.isNull():
            super().setPixmap(QtGui.QPixmap())
            return
        scaled = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        super().setPixmap(scaled)


# ---------------------------------------------------------
# Viewer 嵌入：沿用 Mix.py Viewer，僅改「結束行為」與顯示 label
# ---------------------------------------------------------
class _EmbeddedViewerMixin:
    """
    不改 Mix 的辨識邏輯，只做嵌入與結束行為調整（避免 qApp.quit）。
    """
    finished = QtCore.pyqtSignal()

    def _embed_setup(self) -> None:
        # 把 QMainWindow 當作一般 QWidget 嵌入 layout（去掉獨立視窗標題列）
        self.setWindowFlags(QtCore.Qt.Widget)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Viewer 在 super().__init__() 內就會 start timer，這裡先停掉再替換 label
        try:
            self.timer.stop()
        except Exception:
            pass

        # 以可維持比例的 Label 取代原本 label（不改畫面內容，只是讓縮放更友善）
        ar_label = AspectRatioLabel()
        ar_label.setStyleSheet("background-color: #000;")
        try:
            # 取代 central widget（Mix Viewer 本來就是 setCentralWidget(self.label)）
            self.setCentralWidget(ar_label)
            self.label = ar_label  # 讓原本 _next_frame() 用 self.label.setPixmap(...) 照舊工作
        except Exception:
            # 如果某些情況 setCentralWidget 不可用，退回不替換（仍可跑，只是縮放較不友善）
            pass

        # 重新啟動 timer（沿用 Mix 的 fps）
        try:
            self.timer.start(int(1000 / (self.fps or 30)))
        except Exception:
            pass

    def shutdown(self) -> None:
        """外部強制停止/切換時呼叫，確保釋放資源。"""
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            if getattr(self, "writer", None):
                self.writer.release()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass


class EmbeddedTiptoeViewer(_EmbeddedViewerMixin, mix.TiptoeViewer):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__(video_path, out_path, text_px)
        self._embed_setup()

    def _finish(self):
        # 复制 Mix 的資源釋放行為，但不 qApp.quit()
        self.shutdown()
        # 保留原本 Tiptoe 的 summary print（不影響辨識邏輯）
        try:
            if getattr(self, "frame_count", 0) > 0:
                ratio = self.tiptoe_frames / self.frame_count
                label = "Tiptoe" if ratio >= mix.TIPTOE_RATIO_TH else "Normal"
                print(f"[Summary] frames={self.frame_count}, tiptoe_frames={self.tiptoe_frames}, "
                      f"ratio={ratio:.3f}, video_gait={label}")
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！")
        self.finished.emit()


class EmbeddedHKAViewer(_EmbeddedViewerMixin, mix.HKAViewer):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__(video_path, out_path, text_px)
        self._embed_setup()

    def _finish(self):
        self.shutdown()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！")
        self.finished.emit()


class EmbeddedInOutViewer(_EmbeddedViewerMixin, mix.InOutViewer):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__(video_path, out_path, text_px)
        self._embed_setup()

    def _finish(self):
        self.shutdown()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！")
        self.finished.emit()


class EmbeddedXOViewer(_EmbeddedViewerMixin, mix.XOViewer):
    def __init__(self, video_path: str, out_path: Optional[str], text_px: int):
        super().__init__(video_path, out_path, text_px)
        self._embed_setup()

    def _finish(self):
        self.shutdown()
        QtWidgets.QMessageBox.information(self, "完成", "播放完畢！")
        self.finished.emit()


# -----------------------------
# Demo 主視窗
# -----------------------------
class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("幼童步態分析系統 Demo")
        self.resize(1200, 800)

        self.video_path: Optional[str] = None
        self._viewer: Optional[QtWidgets.QWidget] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # ===== 主 Layout（上下）=====
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ===== Header =====
        header = self._build_header()
        main_layout.addWidget(header)

        # ===== 中央區（影片 + 控制）=====
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(6)

        # 左：影片顯示區（不固定尺寸）
        self.video_group = self._build_video_area()
        splitter.addWidget(self.video_group)

        # 右：控制面板
        control_panel = self._build_control_panel()
        splitter.addWidget(control_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)

        # ===== Footer =====
        footer = self._build_footer()
        main_layout.addWidget(footer)

        # 初始化顯示
        self._update_current_mode()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QtWidgets.QLabel("幼童步態分析系統 Demo")
        title_font = QtGui.QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QtWidgets.QLabel("Gait Analysis for Children")
        subtitle.setStyleSheet("color: gray;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return widget

    # ------------------------------------------------------------------
    # Video Area (Resizable)
    # ------------------------------------------------------------------
    def _build_video_area(self):
        group = QtWidgets.QGroupBox("影片顯示區")
        self.video_layout = QtWidgets.QVBoxLayout(group)

        self.video_placeholder = QtWidgets.QLabel("分析畫面顯示區\n（請先選擇影片）")
        self.video_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.video_placeholder.setStyleSheet("""
            QLabel {
                background-color: #222;
                color: #DDD;
                border: 1px dashed #666;
                padding: 12px;
            }
        """)
        self.video_placeholder.setMinimumSize(320, 240)
        self.video_layout.addWidget(self.video_placeholder, stretch=1)
        return group

    # ------------------------------------------------------------------
    # Control Panel
    # ------------------------------------------------------------------
    def _build_control_panel(self):
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(260)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        # === 分析模式選擇 ===
        mode_group = QtWidgets.QGroupBox("分析模式選擇")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)

        # 已移除「自動辨識」選項：僅保留手動分析模式
        self.radio_tiptoe = QtWidgets.QRadioButton("踮腳步態（Tiptoe）")
        self.radio_inout = QtWidgets.QRadioButton("內 / 外八（In / Out）")
        self.radio_xo = QtWidgets.QRadioButton("X / O 型腿（XO）")
        self.radio_hka = QtWidgets.QRadioButton("HKA（長短腳）")

        # 預設選第一個模式（避免空狀態）
        self.radio_tiptoe.setChecked(True)

        for rb in (self.radio_tiptoe, self.radio_inout, self.radio_xo, self.radio_hka):
            rb.toggled.connect(self._update_current_mode)
            mode_layout.addWidget(rb)

        layout.addWidget(mode_group)

        # === 目前分析模式 ===
        status_group = QtWidgets.QGroupBox("目前分析模式")
        status_layout = QtWidgets.QVBoxLayout(status_group)

        self.current_mode_label = QtWidgets.QLabel("TIPTOE")
        self.current_mode_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(26)
        font.setBold(True)
        self.current_mode_label.setFont(font)
        status_layout.addWidget(self.current_mode_label)

        layout.addWidget(status_group)
        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    def _build_footer(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.btn_start = QtWidgets.QPushButton("▶ 開始分析")
        self.btn_open = QtWidgets.QPushButton("選擇影片")
        self.btn_reset = QtWidgets.QPushButton("重置")
        self.btn_exit = QtWidgets.QPushButton("離開")

        self.btn_open.clicked.connect(self.on_choose_video)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_exit.clicked.connect(self.close)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_reset)
        layout.addStretch(1)
        layout.addWidget(self.btn_exit)
        return widget

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _selected_mode(self) -> str:
        if self.radio_tiptoe.isChecked():
            return "tiptoe"
        if self.radio_inout.isChecked():
            return "inout"
        if self.radio_xo.isChecked():
            return "xo"
        if self.radio_hka.isChecked():
            return "hka"
        # 保底：避免任何原因導致無選取時回傳空值
        return "tiptoe"

    def _mode_to_label(self, mode: str) -> str:
        mode = (mode or "").lower()
        if mode == "tiptoe":
            return "TIPTOE"
        if mode == "inout":
            return "IN / OUT"
        if mode == "xo":
            return "XO"
        if mode == "hka":
            return "HKA"
        # 保底：不顯示 AUTO
        return "TIPTOE"

    def _default_out_path(self, video_path: str) -> Optional[str]:
        # 沿用 Mix.py 的預設行為：輸出 <來源>_annotated.mp4
        root, _ = os.path.splitext(video_path)
        return root + "_annotated.mp4"

    def _clear_viewer(self):
        if self._viewer is None:
            return

        # 嘗試停止播放與釋放
        try:
            if hasattr(self._viewer, "shutdown"):
                self._viewer.shutdown()  # type: ignore[attr-defined]
        except Exception:
            pass

        # 從 layout 移除
        try:
            self.video_layout.removeWidget(self._viewer)
        except Exception:
            pass
        try:
            self._viewer.setParent(None)
            self._viewer.deleteLater()
        except Exception:
            pass
        self._viewer = None

        # 顯示 placeholder
        self.video_placeholder.show()

    def _mount_viewer(self, viewer: QtWidgets.QWidget):
        self._clear_viewer()
        self.video_placeholder.hide()
        self._viewer = viewer
        self.video_layout.addWidget(viewer, stretch=1)
        viewer.show()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _update_current_mode(self):
        # 顯示「目前分析模式」：跟著右側選的類別（TIPTOE / INOUT / XO / HKA）
        label = self._mode_to_label(self._selected_mode())
        self.current_mode_label.setText(label)

    def on_choose_video(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇影片", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not fname:
            return
        self.video_path = fname
        self.video_placeholder.setText(f"已選擇影片：\n{os.path.basename(fname)}")
        # 換影片時，先清掉舊 viewer（避免同時播放兩個）
        self._clear_viewer()

    def on_start(self):
        # 沒選影片就跳選檔
        if not self.video_path:
            self.on_choose_video()
            if not self.video_path:
                return

        # 手動：直接開對應 viewer
        mode = self._selected_mode()
        self._launch_viewer(self.video_path, mode)

    def on_reset(self):
        # 清掉 viewer，但保留目前 radio 狀態（你可以繼續選模式）
        self._clear_viewer()
        self.video_path = None
        self.video_placeholder.setText("分析畫面顯示區\n（請先選擇影片）")
        self.video_placeholder.show()
        self._update_current_mode()

    # ------------------------------------------------------------------
    # Launch viewer
    # ------------------------------------------------------------------
    def _launch_viewer(self, video_path: str, mode: str):
        mode = (mode or "").lower()
        out_path = self._default_out_path(video_path)
        text_px = int(getattr(mix, "DEFAULT_TEXT_PX", 10))

        try:
            if mode == "tiptoe":
                viewer = EmbeddedTiptoeViewer(video_path, out_path, text_px)
            elif mode == "inout":
                viewer = EmbeddedInOutViewer(video_path, out_path, text_px)
            elif mode == "xo":
                viewer = EmbeddedXOViewer(video_path, out_path, text_px)
            else:
                viewer = EmbeddedHKAViewer(video_path, out_path, text_px)

            # 嵌入左側顯示區
            self._mount_viewer(viewer)

            # 同步「目前分析模式」
            self.current_mode_label.setText(self._mode_to_label(mode))

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "啟動分析失敗", f"{e}\n\n{traceback.format_exc()}"
            )


# -----------------------------
# main
# -----------------------------
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
