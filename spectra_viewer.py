#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# 设置matplotlib使用PySide6后端
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT


from PySide6.QtWidgets import QToolBar
from PySide6.QtGui import QIcon, QAction


from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QListWidget, QSplitter, QCheckBox, QComboBox,
                            QGroupBox, QGridLayout, QMessageBox, QSizePolicy,
                            QMenu, QTextEdit, QDialog, QTableWidget,
                            QTableWidgetItem, QTabWidget, QHeaderView,
                            QLineEdit, QListWidgetItem)
from PySide6.QtCore import Qt, QDir, QPoint
from PySide6.QtGui import QColor, QFont, QBrush, QTextDocument, QAction
from PySide6.QtPrintSupport import QPrinter, QPrintDialog
import random
import matplotlib
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import probplot, skew, kurtosis
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import io
import base64
import itertools

# 配置matplotlib使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'

# 创建一个简单的自定义工具栏，而不是使用NavigationToolbar2QT
def create_toolbar(canvas, parent):
    """创建一个完整的导航工具栏，使用NavigationToolbar2QT"""
    # 使用matplotlib的标准导航工具栏
    # 检查canvas是否为SpectralCanvas类型或FigureCanvas类型
    if hasattr(canvas, 'canvas'):
        # 如果是SpectralCanvas类型，它有canvas属性
        toolbar = NavigationToolbar2QT(canvas.canvas, parent)
    else:
        # 如果是FigureCanvas类型，直接使用它
        toolbar = NavigationToolbar2QT(canvas, parent)

    # 不再添加"重置视图"和"切换网格"按钮，按照用户要求

    return toolbar

class BaseAnalysisDialog(QDialog):
    """分析对话框的基类"""

    def __init__(self, parent=None, title="分析结果"):
        super().__init__(parent)
        self.setWindowTitle(title)

        # 获取屏幕尺寸并设置窗口为屏幕的80%大小
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry((screen.width() - width) // 2, (screen.height() - height) // 2, width, height)

        # 设置窗口标志，允许最大化
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.setModal(True)

        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # 创建顶部标题栏布局
        title_bar = QWidget()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet("background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)

        # 标题文本
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title_layout.addWidget(title_label)

        # 添加标题栏到主布局
        self.main_layout.addWidget(title_bar)

        # 创建内容区域
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.main_layout.addWidget(self.content_widget)

class AnalysisResultDialog(BaseAnalysisDialog):
    """用于显示光谱分析结果的对话框"""

    def __init__(self, parent=None, title="光谱差异分析结果"):
        super().__init__(parent, title)

        # 创建文本显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)  # 设置为只读
        self.result_text.setFont(QFont("Microsoft YaHei", 10))
        self.content_layout.addWidget(self.result_text)

    def set_result_text(self, text):
        """设置结果文本"""
        self.result_text.setHtml(text)

class SingleCurveAnalysisDialog(BaseAnalysisDialog):
    """单条光谱曲线分析对话框"""

    def __init__(self, parent=None, curve_data=None, curve_name="", curve_index=0):
        """初始化单条曲线分析对话框

        Args:
            parent: 父窗口
            curve_data: 元组(波长数组, 反射率数组)
            curve_name: 曲线名称
            curve_index: 曲线在主图表中的索引
        """
        super().__init__(parent, f"单条曲线分析 - {curve_name}")

        # 存储曲线数据
        self.wavelength, self.reflectance = curve_data
        self.curve_name = curve_name
        self.curve_index = curve_index
        self.parent = parent

        # 创建标签页控件
        self.tab_widget = QTabWidget()

        # 添加各个分析标签页
        self.add_statistics_tab()
        self.add_derivative_tab()
        self.add_region_analysis_tab()
        self.add_features_tab()

        # 如果存在标准曲线，添加比对分析标签页
        if hasattr(self.parent, 'reference_curve') and self.parent.reference_curve is not None:
            self.add_reference_comparison_tab()

        self.content_layout.addWidget(self.tab_widget)

        # 添加关闭按钮
        self.buttons_layout = QHBoxLayout()
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.close_btn)
        self.content_layout.addLayout(self.buttons_layout)

    def add_statistics_tab(self):
        """添加基础统计分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建统计信息表格
        stats_table = QTableWidget()
        stats_table.setColumnCount(2)
        stats_table.setHorizontalHeaderLabels(["统计指标", "数值"])
        stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]

        if len(valid_reflectance) > 0:
            # 计算统计指标
            stats_data = [
                ("样本数量", len(valid_reflectance)),
                ("最小值", f"{np.min(valid_reflectance):.2f}%"),
                ("最大值", f"{np.max(valid_reflectance):.2f}%"),
                ("平均值", f"{np.mean(valid_reflectance):.2f}%"),
                ("中位数", f"{np.median(valid_reflectance):.2f}%"),
                ("标准差", f"{np.std(valid_reflectance):.2f}"),
                ("偏度", f"{skew(valid_reflectance):.2f}"),
                ("峰度", f"{kurtosis(valid_reflectance):.2f}"),
                ("变异系数", f"{np.std(valid_reflectance)/np.mean(valid_reflectance):.2f}"),
                ("波长范围", f"{np.min(valid_wavelength):.0f} - {np.max(valid_wavelength):.0f} nm"),
                ("反射率范围", f"{np.min(valid_reflectance):.2f} - {np.max(valid_reflectance):.2f}%"),
                ("四分位数(Q1)", f"{np.percentile(valid_reflectance, 25):.2f}%"),
                ("四分位数(Q3)", f"{np.percentile(valid_reflectance, 75):.2f}%"),
                ("四分位距(IQR)", f"{np.percentile(valid_reflectance, 75) - np.percentile(valid_reflectance, 25):.2f}"),
            ]

            # 填充表格
            stats_table.setRowCount(len(stats_data))
            for i, (metric, value) in enumerate(stats_data):
                stats_table.setItem(i, 0, QTableWidgetItem(metric))
                stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # 创建直方图和QQ图
        if len(valid_reflectance) > 5:
            fig = Figure(figsize=(9, 5))
            canvas = FigureCanvas(fig)

            # 创建直方图
            ax1 = fig.add_subplot(121)
            ax1.hist(valid_reflectance, bins=min(20, len(valid_reflectance)//5),
                   density=True, alpha=0.7, color='skyblue')
            ax1.set_xlabel('反射率 (%)')
            ax1.set_ylabel('频率')
            ax1.set_title('反射率直方图')

            # 添加正态分布曲线进行比较
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            mean = np.mean(valid_reflectance)
            std = np.std(valid_reflectance)
            p = stats.norm.pdf(x, mean, std)
            ax1.plot(x, p, 'k', linewidth=1)

            # 创建QQ图
            ax2 = fig.add_subplot(122)
            # 直接使用已导入的probplot函数
            probplot(valid_reflectance, plot=ax2)
            ax2.set_title('QQ图 (检验正态性)')

            fig.tight_layout()
        else:
            canvas = QLabel("数据点不足，无法创建图表")

        layout.addWidget(stats_table)
        layout.addWidget(canvas)

        self.tab_widget.addTab(tab, "基础统计")

    def add_derivative_tab(self):
        """添加导数分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]

        if len(valid_reflectance) < 5:
            layout.addWidget(QLabel("数据点不足，无法计算导数"))
            self.tab_widget.addTab(tab, "导数分析")
            return

        # 计算一阶和二阶导数
        # 确定窗口大小 - 必须是奇数且小于数据点数
        window_length = min(11, len(valid_reflectance) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length -= 1

        if window_length >= 3:
            # 平滑处理
            smoothed = savgol_filter(valid_reflectance, window_length, 2)

            # 计算导数
            dx = np.diff(valid_wavelength)
            first_derivative = np.diff(smoothed) / dx
            second_derivative = np.diff(first_derivative) / dx[:-1]

            # 创建图表
            fig = Figure(figsize=(9, 8))

            # 反射率曲线
            ax1 = fig.add_subplot(311)
            ax1.plot(valid_wavelength, valid_reflectance, 'o-', markersize=2,
                   label='原始数据')
            ax1.plot(valid_wavelength, smoothed, '-', label='平滑后')
            ax1.set_xlabel('波长 (nm)')
            ax1.set_ylabel('反射率 (%)')
            ax1.set_title('反射率曲线')
            # 使用扁平化图例样式
            ax1.legend(
                loc='upper right',
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.0,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white'  # 白色背景
            )
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 一阶导数
            ax2 = fig.add_subplot(312)
            ax2.plot(valid_wavelength[:-1], first_derivative, '-')
            ax2.set_xlabel('波长 (nm)')
            ax2.set_ylabel('一阶导数')
            ax2.set_title('一阶导数曲线')
            ax2.grid(True, linestyle='--', alpha=0.7)

            # 寻找一阶导数的零点(极值点)
            zero_crosses = []
            for i in range(1, len(first_derivative)):
                if (first_derivative[i-1] < 0 and first_derivative[i] > 0) or \
                   (first_derivative[i-1] > 0 and first_derivative[i] < 0):
                    # 线性插值找到零点
                    wl_cross = valid_wavelength[i-1] + dx[i-1] * (-first_derivative[i-1]) / (first_derivative[i] - first_derivative[i-1])
                    zero_crosses.append((wl_cross, i))

            # 标记一阶导数的零点
            if zero_crosses:
                zero_x = [x for x, _ in zero_crosses[:10]]  # 最多显示10个
                zero_y = np.interp(zero_x, valid_wavelength, smoothed)
                ax1.plot(zero_x, zero_y, 'ro', label='极值点')

                # 最多标注前5个
                for i, (x, idx) in enumerate(zero_crosses[:5]):
                    y = np.interp(x, valid_wavelength, smoothed)
                    ax1.annotate(f"极值{i+1}", (x, y), xytext=(10, 10),
                               textcoords='offset points',
                               arrowprops=dict(arrowstyle='->'))

            # 二阶导数
            ax3 = fig.add_subplot(313)
            ax3.plot(valid_wavelength[:-2], second_derivative, '-')
            ax3.set_xlabel('波长 (nm)')
            ax3.set_ylabel('二阶导数')
            ax3.set_title('二阶导数曲线')
            ax3.grid(True, linestyle='--', alpha=0.7)

            # 找到二阶导数的零点(拐点)
            inflections = []
            for i in range(1, len(second_derivative)):
                if (second_derivative[i-1] < 0 and second_derivative[i] > 0) or \
                   (second_derivative[i-1] > 0 and second_derivative[i] < 0):
                    # 线性插值找到零点
                    wl_cross = valid_wavelength[i] + dx[i] * (-second_derivative[i-1]) / (second_derivative[i] - second_derivative[i-1])
                    inflections.append((wl_cross, i))

            # 在原曲线上标记拐点
            if inflections:
                infl_x = [x for x, _ in inflections[:10]]  # 最多显示10个
                infl_y = np.interp(infl_x, valid_wavelength, smoothed)
                ax1.plot(infl_x, infl_y, 'gx', label='拐点')

                # 最多标注前5个
                for i, (x, idx) in enumerate(inflections[:5]):
                    y = np.interp(x, valid_wavelength, smoothed)
                    ax1.annotate(f"拐点{i+1}", (x, y), xytext=(10, -20),
                               textcoords='offset points',
                               arrowprops=dict(arrowstyle='->'))

            # 使用扁平化图例样式
            ax1.legend(
                loc='upper right',
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.0,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white'  # 白色背景
            )
            fig.tight_layout()

            # 创建文本说明
            text_info = ""
            if zero_crosses:
                text_info += "检测到的极值点:\n"
                for i, (wl, _) in enumerate(zero_crosses[:10]):  # 最多显示10个
                    y = np.interp(wl, valid_wavelength, smoothed)
                    text_info += f"    极值点{i+1}: 波长={wl:.1f}nm, 反射率={y:.2f}%\n"

            if inflections:
                text_info += "\n检测到的拐点:\n"
                for i, (wl, _) in enumerate(inflections[:10]):  # 最多显示10个
                    y = np.interp(wl, valid_wavelength, smoothed)
                    text_info += f"    拐点{i+1}: 波长={wl:.1f}nm, 反射率={y:.2f}%\n"

            info_label = QLabel(text_info)
            info_label.setStyleSheet("font-family: monospace;")

            canvas = FigureCanvas(fig)

            layout.addWidget(canvas)
            layout.addWidget(info_label)
        else:
            layout.addWidget(QLabel("数据点不足，无法进行平滑和导数计算"))

        self.tab_widget.addTab(tab, "导数分析")

    def add_region_analysis_tab(self):
        """添加区域分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]

        if len(valid_reflectance) < 5:
            layout.addWidget(QLabel("数据点不足，无法进行区域分析"))
            self.tab_widget.addTab(tab, "区域分析")
            return

        # 定义常见的光谱区域
        regions = [
            (350, 450, "紫外-蓝光"),
            (450, 500, "蓝光"),
            (500, 570, "绿光"),
            (570, 620, "黄光"),
            (620, 750, "红光"),
            (750, 1050, "近红外")
        ]

        # 定义马卡龙色系映射，对应各个光谱区域的真实颜色
        region_colors = {
            "紫外-蓝光": "#9370DB",  # 紫罗兰色
            "蓝光": "#6495ED",      # 亮蓝色
            "绿光": "#98FB98",      # 淡绿色
            "黄光": "#FFD700",      # 金黄色
            "红光": "#FF6347",      # 番茄红
            "近红外": "#F08080"     # 浅珊瑚色
        }

        # 创建区域分析表格
        region_table = QTableWidget()
        region_table.setColumnCount(7)
        region_table.setHorizontalHeaderLabels([
            "波长区域", "平均反射率", "标准差",
            "最大值", "最大值波长", "最小值", "最小值波长"
        ])
        region_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # 区域分析图表
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel('波长 (nm)', fontsize=10)
        ax.set_ylabel('反射率 (%)', fontsize=10)
        ax.set_title('区域分析', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 首先绘制原始光谱曲线
        ax.plot(valid_wavelength, valid_reflectance, 'k-', linewidth=1.5, label=self.curve_name)

        # 为每个区域计算统计量
        region_stats = []
        valid_regions = []

        for wl_min, wl_max, region_name in regions:
            # 检查是否有数据在此区域
            mask = (valid_wavelength >= wl_min) & (valid_wavelength <= wl_max)
            if np.any(mask) and np.sum(mask) >= 3:  # 至少需要3个点
                region_refl = valid_reflectance[mask]
                region_wl = valid_wavelength[mask]

                # 计算区域统计量
                mean_val = np.mean(region_refl)
                std_val = np.std(region_refl)

                # 查找最大值和最小值，以及它们对应的波长
                max_idx = np.argmax(region_refl)
                min_idx = np.argmin(region_refl)
                max_val = region_refl[max_idx]
                min_val = region_refl[min_idx]
                max_wl = region_wl[max_idx]
                min_wl = region_wl[min_idx]

                region_stats.append((region_name, mean_val, std_val, max_val, max_wl, min_val, min_wl))
                valid_regions.append((wl_min, wl_max, region_name))

                # 添加区域背景，使用马卡龙色系映射
                region_color = region_colors.get(region_name, "#DDDDDD")  # 默认为浅灰色
                ax.axvspan(wl_min, wl_max, alpha=0.25, color=region_color)

                # 添加区域标签
                ax.text((wl_min + wl_max) / 2, ax.get_ylim()[1] * 0.95,
                       region_name, ha='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=region_color,
                                boxstyle='round,pad=0.2', linewidth=1))

                # 标记最大值和最小值
                ax.plot(max_wl, max_val, 'ro', markersize=4)
                ax.plot(min_wl, min_val, 'go', markersize=4)

                # 为最大值和最小值添加文本标注（位置更合理）
                ax.annotate(f"{max_val:.1f}%",
                           (max_wl, max_val), xytext=(0, 5),
                           textcoords='offset points', fontsize=7,
                           color='red', fontweight='bold', ha='center')

                ax.annotate(f"{min_val:.1f}%",
                           (min_wl, min_val), xytext=(0, -10),
                           textcoords='offset points', fontsize=7,
                           color='green', fontweight='bold', ha='center')

        # 填充表格
        region_table.setRowCount(len(region_stats))
        for i, (name, mean_val, std_val, max_val, max_wl, min_val, min_wl) in enumerate(region_stats):
            region_table.setItem(i, 0, QTableWidgetItem(name))
            region_table.setItem(i, 1, QTableWidgetItem(f"{mean_val:.2f}%"))
            region_table.setItem(i, 2, QTableWidgetItem(f"{std_val:.2f}"))
            region_table.setItem(i, 3, QTableWidgetItem(f"{max_val:.2f}%"))
            region_table.setItem(i, 4, QTableWidgetItem(f"{max_wl:.2f}nm"))
            region_table.setItem(i, 5, QTableWidgetItem(f"{min_val:.2f}%"))
            region_table.setItem(i, 6, QTableWidgetItem(f"{min_wl:.2f}nm"))

        # 设置表格自适应内容
        region_table.resizeRowsToContents()
        region_table.resizeColumnsToContents()

        # 添加区域高亮按钮
        buttons_widget = QWidget()
        highlight_layout = QHBoxLayout(buttons_widget)
        highlight_layout.setContentsMargins(0, 5, 0, 5)
        highlight_label = QLabel("高亮显示区域:")
        highlight_layout.addWidget(highlight_label)

        buttons = []
        for wl_min, wl_max, region_name in valid_regions:
            btn = QPushButton(region_name)
            btn.setProperty("wl_min", wl_min)
            btn.setProperty("wl_max", wl_max)
            btn.clicked.connect(self.highlight_region)
            highlight_layout.addWidget(btn)
            buttons.append(btn)

        # 添加取消高亮按钮
        if buttons:
            clear_btn = QPushButton("清除高亮")
            clear_btn.clicked.connect(self.clear_highlight)
            highlight_layout.addWidget(clear_btn)

        highlight_layout.addStretch()

        # 创建画布并设置尺寸策略
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建并添加导航工具栏 - 使用辅助函数创建PySide6兼容的工具栏
        toolbar = create_toolbar(canvas, tab)

        # 调整布局：表格在上，图在中间，按钮在底部
        layout.addWidget(region_table)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        layout.addWidget(buttons_widget)

        fig.tight_layout()

        self.tab_widget.addTab(tab, "区域分析")

    def highlight_region(self):
        """高亮显示特定波长区域"""
        sender = self.sender()
        wl_min = sender.property("wl_min")
        wl_max = sender.property("wl_max")
        region_name = sender.text()

        # 获取区域对应的马卡龙颜色
        region_colors = {
            "紫外-蓝光": "#9370DB",  # 紫罗兰色
            "蓝光": "#6495ED",      # 亮蓝色
            "绿光": "#98FB98",      # 淡绿色
            "黄光": "#FFD700",      # 金黄色
            "红光": "#FF6347",      # 番茄红
            "近红外": "#F08080"     # 浅珊瑚色
        }
        color = region_colors.get(region_name, "yellow")

        # 调用父窗口中的画布来高亮显示区域
        if hasattr(self.parent, 'canvas') and self.parent.canvas is not None:
            try:
                # 使用对应区域的马卡龙颜色
                self.parent.canvas.highlight_curve_segment(
                    self.curve_index, wl_min, wl_max,
                    color=color, alpha=0.5
                )
            except Exception as e:
                print(f"高亮区域出错: {e}")

    def clear_highlight(self):
        """清除高亮显示"""
        if hasattr(self.parent, 'canvas') and self.parent.canvas is not None:
            try:
                # 通过重新绘制当前图表来清除高亮
                self.parent.update_plot()
            except Exception as e:
                print(f"清除高亮出错: {e}")
                # 尝试简单地重绘画布
                try:
                    self.parent.canvas.draw()
                except:
                    pass

    def add_features_tab(self):
        """添加特征点分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]

        if len(valid_reflectance) < 5:
            # 数据点太少，无法进行特征点分析
            layout.addWidget(QLabel("数据点不足，无法进行特征点分析。"))
            self.tab_widget.addTab(tab, "特征点分析")
            return

        # 创建图表
        fig = Figure(figsize=(9, 6))
        canvas = FigureCanvas(fig)

        # 平滑处理
        window_length = min(15, len(valid_reflectance) - (len(valid_reflectance) % 2) - 1)
        if window_length < 5:
            window_length = 5

        smoothed = savgol_filter(valid_reflectance, window_length, 3)

        # 查找峰值和谷值
        # 使用相对高度参数来确保只找到明显的峰值和谷值
        peaks, _ = find_peaks(smoothed, height=np.mean(smoothed), distance=5, prominence=0.5)
        valleys, _ = find_peaks(-smoothed, distance=5, prominence=0.5)

        # 主图形：显示曲线、峰值和谷值
        ax = fig.add_subplot(111)
        ax.plot(valid_wavelength, valid_reflectance, 'b-', alpha=0.5, label='原始数据')
        ax.plot(valid_wavelength, smoothed, 'g-', label='平滑曲线')

        # 标记峰值
        if len(peaks) > 0:
            ax.plot(valid_wavelength[peaks], smoothed[peaks], 'ro', label='峰值')
            for i, peak in enumerate(peaks):
                ax.annotate(f"{valid_wavelength[peak]:.0f}nm",
                           (valid_wavelength[peak], smoothed[peak]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8)

        # 标记谷值
        if len(valleys) > 0:
            ax.plot(valid_wavelength[valleys], smoothed[valleys], 'mo', label='谷值')
            for i, valley in enumerate(valleys):
                ax.annotate(f"{valid_wavelength[valley]:.0f}nm",
                           (valid_wavelength[valley], smoothed[valley]),
                           xytext=(0, -15), textcoords='offset points',
                           ha='center', fontsize=8)

        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('反射率 (%)')
        ax.set_title(f'{self.curve_name} - 特征点分析')
        ax.grid(True, linestyle='--', alpha=0.7)
        # 使用扁平化图例样式
        ax.legend(
            loc='upper right',
            fontsize=10,
            frameon=True,
            framealpha=0.7,  # 更透明的背景
            fancybox=False,  # 扁平化设计，不使用圆角
            shadow=False,    # 扁平化设计，不使用阴影
            borderpad=0.6,
            labelspacing=0.8,
            handlelength=2.0,
            handleheight=0.8,
            handletextpad=0.6,
            edgecolor='lightgray',
            facecolor='white'  # 白色背景
        )

        # 创建特征点表格
        features_table = QTableWidget()
        features_table.setColumnCount(3)
        features_table.setHorizontalHeaderLabels(["特征点类型", "波长 (nm)", "反射率 (%)"])
        features_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 计算表格行数
        row_count = len(peaks) + len(valleys)
        features_table.setRowCount(row_count)

        # 填充峰值数据
        for i, peak in enumerate(peaks):
            features_table.setItem(i, 0, QTableWidgetItem("峰值"))
            features_table.setItem(i, 1, QTableWidgetItem(f"{valid_wavelength[peak]:.1f}"))
            features_table.setItem(i, 2, QTableWidgetItem(f"{smoothed[peak]:.2f}"))

        # 填充谷值数据
        for i, valley in enumerate(valleys):
            row = i + len(peaks)
            features_table.setItem(row, 0, QTableWidgetItem("谷值"))
            features_table.setItem(row, 1, QTableWidgetItem(f"{valid_wavelength[valley]:.1f}"))
            features_table.setItem(row, 2, QTableWidgetItem(f"{smoothed[valley]:.2f}"))

        # 添加特征点描述
        features_description = QLabel(
            f"<p>检测到 <b>{len(peaks)}</b> 个峰值和 <b>{len(valleys)}</b> 个谷值。</p>"
            f"<p>峰值代表光谱中反射率局部最大的波长位置，通常与材料的吸收特性相关。</p>"
            f"<p>谷值代表光谱中反射率局部最小的波长位置，通常指示材料对特定波长光的强吸收。</p>"
        )
        features_description.setWordWrap(True)

        # 添加控件到布局
        layout.addWidget(canvas)
        layout.addWidget(features_description)
        layout.addWidget(QLabel("<b>特征点数据表</b>"))
        layout.addWidget(features_table)

        # 添加到标签页
        self.tab_widget.addTab(tab, "特征点分析")

    def add_reference_comparison_tab(self):
        """添加与标准曲线比对分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 获取标准曲线数据
        reference_name = self.parent.reference_curve
        clean_ref_name = reference_name.replace(" [标准]", "")

        if clean_ref_name not in self.parent.spectra_data:
            layout.addWidget(QLabel(f"未找到标准曲线：{clean_ref_name}"))
            self.tab_widget.addTab(tab, "标准比对")
            return

        # 获取标准曲线数据
        ref_wavelength, ref_reflectance = self.parent.spectra_data[clean_ref_name]

        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]

        ref_mask = ~np.isnan(ref_reflectance)
        valid_ref_wavelength = ref_wavelength[ref_mask]
        valid_ref_reflectance = ref_reflectance[ref_mask]

        if len(valid_reflectance) < 5 or len(valid_ref_reflectance) < 5:
            layout.addWidget(QLabel("数据点不足，无法进行比对分析。"))
            self.tab_widget.addTab(tab, "标准比对")
            return

        # 创建比对分析结果的HTML
        html_result = "<h2>标准曲线比对分析</h2>"
        html_result += f"<p>当前曲线：<b>{self.curve_name}</b></p>"
        html_result += f"<p>标准曲线：<b>{clean_ref_name}</b></p>"

        # 1. 找到两条曲线共同的波长范围
        min_wl = max(np.min(valid_wavelength), np.min(valid_ref_wavelength))
        max_wl = min(np.max(valid_wavelength), np.max(valid_ref_wavelength))

        # 限制在共同波长范围内
        current_mask = (valid_wavelength >= min_wl) & (valid_wavelength <= max_wl)
        ref_mask = (valid_ref_wavelength >= min_wl) & (valid_ref_wavelength <= max_wl)

        # 检查是否有足够的重叠数据
        if np.sum(current_mask) < 5 or np.sum(ref_mask) < 5:
            html_result += "<p style='color:red'>警告：两条曲线的波长范围重叠度不足，比对结果可能不准确。</p>"

        # 创建图表容器
        fig_container = QWidget()
        fig_layout = QVBoxLayout(fig_container)

        # 2. 创建直观对比图
        fig1 = Figure(figsize=(10, 6))
        canvas1 = FigureCanvas(fig1)

        # 绘制对比图
        ax1 = fig1.add_subplot(111)
        ax1.plot(valid_wavelength, valid_reflectance, 'b-', label=self.curve_name)
        ax1.plot(valid_ref_wavelength, valid_ref_reflectance, 'r-', label=f"{clean_ref_name} (标准)")
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('反射率 (%)')
        ax1.set_title('光谱曲线对比')
        ax1.grid(True, linestyle='--', alpha=0.7)
        # 使用扁平化图例样式
        ax1.legend(
            loc='upper right',
            fontsize=10,
            frameon=True,
            framealpha=0.7,  # 更透明的背景
            fancybox=False,  # 扁平化设计，不使用圆角
            shadow=False,    # 扁平化设计，不使用阴影
            borderpad=0.6,
            labelspacing=0.8,
            handlelength=2.0,
            handleheight=0.8,
            handletextpad=0.6,
            edgecolor='lightgray',
            facecolor='white'  # 白色背景
        )

        # 3. 需要插值处理使两条曲线在相同波长点上进行比较
        # 使用当前曲线的波长点作为参考，插值计算标准曲线在这些点上的值
        common_wavelength = valid_wavelength[current_mask]
        current_reflectance = valid_reflectance[current_mask]

        # 对标准曲线进行插值
        ref_interp = np.interp(common_wavelength, valid_ref_wavelength, valid_ref_reflectance)

        # 4. 计算差异曲线
        diff = current_reflectance - ref_interp
        abs_diff = np.abs(diff)
        rel_diff = np.abs(diff / ref_interp) * 100  # 相对差异百分比

        # 5. 计算统计指标
        rmse = np.sqrt(np.mean(diff**2))  # 均方根误差
        mae = np.mean(abs_diff)  # 平均绝对误差
        max_diff = np.max(abs_diff)  # 最大绝对差异
        max_diff_wl = common_wavelength[np.argmax(abs_diff)]  # 最大差异处的波长

        # 计算相关系数
        corr_coef = np.corrcoef(current_reflectance, ref_interp)[0, 1]

        # 计算光谱角映射(SAM)
        norm_current = current_reflectance / np.linalg.norm(current_reflectance)
        norm_ref = ref_interp / np.linalg.norm(ref_interp)
        sam = np.arccos(np.clip(np.dot(norm_current, norm_ref), -1.0, 1.0))
        sam_deg = np.degrees(sam)

        # 添加统计指标表
        html_result += "<h3>统计比对指标</h3>"
        html_result += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        html_result += "<tr><th>指标</th><th>数值</th><th>解释</th></tr>"
        html_result += f"<tr><td>相关系数</td><td>{corr_coef:.4f}</td><td>衡量两曲线的线性相关性，值越接近1表示相关性越高</td></tr>"
        html_result += f"<tr><td>均方根误差(RMSE)</td><td>{rmse:.4f}%</td><td>衡量曲线间的平均差异程度</td></tr>"
        html_result += f"<tr><td>平均绝对误差(MAE)</td><td>{mae:.4f}%</td><td>曲线间的平均绝对差异</td></tr>"
        html_result += f"<tr><td>最大绝对差异</td><td>{max_diff:.4f}% (波长: {max_diff_wl:.0f}nm)</td><td>两曲线间的最大差异点</td></tr>"
        html_result += f"<tr><td>光谱角距离(SAM)</td><td>{sam_deg:.2f}°</td><td>衡量光谱形状相似度，值越小越相似</td></tr>"
        html_result += "</table>"

        # 绘制差异曲线
        fig2 = Figure(figsize=(10, 6))
        canvas2 = FigureCanvas(fig2)

        # 创建两个子图
        ax2 = fig2.add_subplot(211)
        ax2.plot(common_wavelength, diff, 'g-')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('差异 (%)')
        ax2.set_title('光谱差异曲线 (当前曲线 - 标准曲线)')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 绘制相对差异曲线
        ax3 = fig2.add_subplot(212)
        ax3.plot(common_wavelength, rel_diff, 'm-')
        ax3.set_xlabel('波长 (nm)')
        ax3.set_ylabel('相对差异 (%)')
        ax3.set_title('相对差异百分比')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # 添加图表说明
        fig2.tight_layout()

        # 添加到布局
        fig_layout.addWidget(canvas1)
        fig_layout.addWidget(canvas2)

        # 创建文本结果显示
        text_result = QTextEdit()
        text_result.setReadOnly(True)
        text_result.setHtml(html_result)

        # 创建区域分析比对
        html_region_analysis = "<h3>光谱区域对比分析</h3>"
        html_region_analysis += "<p>不同波长区域的特性比对：</p>"

        # 定义常见的光谱区域
        regions = [
            (350, 450, "紫外-蓝光"),
            (450, 500, "蓝光"),
            (500, 570, "绿光"),
            (570, 620, "黄光"),
            (620, 750, "红光"),
            (750, 1050, "近红外")
        ]

        # 为每个区域分析两条曲线的差异
        html_region_analysis += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        html_region_analysis += "<tr><th>区域</th><th>波长范围</th><th>当前曲线均值</th><th>标准曲线均值</th><th>差异</th><th>相对差异</th></tr>"

        region_diff_data = []
        for wl_min, wl_max, region_name in regions:
            # 检查区域是否在共同波长范围内
            if wl_min <= max_wl and wl_max >= min_wl:
                # 获取区域内的数据
                region_mask = (common_wavelength >= wl_min) & (common_wavelength <= wl_max)
                if np.sum(region_mask) > 0:
                    current_region = current_reflectance[region_mask]
                    ref_region = ref_interp[region_mask]

                    # 计算区域内的均值
                    current_mean = np.mean(current_region)
                    ref_mean = np.mean(ref_region)

                    # 计算差异
                    diff_mean = current_mean - ref_mean
                    rel_diff_mean = (diff_mean / ref_mean) * 100 if ref_mean != 0 else 0

                    # 添加到表格
                    html_region_analysis += f"<tr><td>{region_name}</td><td>{wl_min}-{wl_max}nm</td>"
                    html_region_analysis += f"<td>{current_mean:.2f}%</td><td>{ref_mean:.2f}%</td>"
                    html_region_analysis += f"<td>{diff_mean:.2f}%</td><td>{rel_diff_mean:.2f}%</td></tr>"

                    region_diff_data.append((wl_min, wl_max, region_name, diff_mean, rel_diff_mean))

        html_region_analysis += "</table>"

        # 添加特征点比对
        html_feature_analysis = "<h3>特征点比对分析</h3>"

        # 平滑处理
        window_length = min(15, len(valid_reflectance) - (len(valid_reflectance) % 2) - 1)
        if window_length < 5:
            window_length = 5

        current_smoothed = savgol_filter(valid_reflectance, window_length, 3)
        ref_smoothed = savgol_filter(valid_ref_reflectance, window_length, 3)

        # 查找峰值和谷值
        current_peaks, _ = find_peaks(current_smoothed, distance=5, prominence=0.5)
        current_valleys, _ = find_peaks(-current_smoothed, distance=5, prominence=0.5)

        ref_peaks, _ = find_peaks(ref_smoothed, distance=5, prominence=0.5)
        ref_valleys, _ = find_peaks(-ref_smoothed, distance=5, prominence=0.5)

        # 绘制特征点比对图
        fig3 = Figure(figsize=(10, 6))
        canvas3 = FigureCanvas(fig3)

        ax4 = fig3.add_subplot(111)
        ax4.plot(valid_wavelength, current_smoothed, 'b-', label=f"{self.curve_name} (平滑)")
        ax4.plot(valid_ref_wavelength, ref_smoothed, 'r-', label=f"{clean_ref_name} (平滑)")

        # 标记当前曲线的峰值和谷值
        if len(current_peaks) > 0:
            ax4.plot(valid_wavelength[current_peaks], current_smoothed[current_peaks], 'bo', label='当前峰值')
        if len(current_valleys) > 0:
            ax4.plot(valid_wavelength[current_valleys], current_smoothed[current_valleys], 'bv', label='当前谷值')

        # 标记标准曲线的峰值和谷值
        if len(ref_peaks) > 0:
            ax4.plot(valid_ref_wavelength[ref_peaks], ref_smoothed[ref_peaks], 'ro', label='标准峰值')
        if len(ref_valleys) > 0:
            ax4.plot(valid_ref_wavelength[ref_valleys], ref_smoothed[ref_valleys], 'rv', label='标准谷值')

        ax4.set_xlabel('波长 (nm)')
        ax4.set_ylabel('反射率 (%)')
        ax4.set_title('特征点比对')
        ax4.grid(True, linestyle='--', alpha=0.7)
        # 使用扁平化图例样式
        ax4.legend(
            loc='upper right',
            fontsize=10,
            frameon=True,
            framealpha=0.7,  # 更透明的背景
            fancybox=False,  # 扁平化设计，不使用圆角
            shadow=False,    # 扁平化设计，不使用阴影
            borderpad=0.6,
            labelspacing=0.8,
            handlelength=2.0,
            handleheight=0.8,
            handletextpad=0.6,
            edgecolor='lightgray',
            facecolor='white'  # 白色背景
        )

        fig3.tight_layout()

        # 特征点比对表格
        html_feature_analysis += "<p>峰值比对：</p>"
        html_feature_analysis += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        html_feature_analysis += "<tr><th>当前曲线峰值波长</th><th>当前峰值反射率</th><th>最接近的标准峰值波长</th><th>标准峰值反射率</th><th>波长差异</th></tr>"

        # 对每个当前曲线的峰值，找到最接近的标准曲线峰值
        for peak in current_peaks:
            current_peak_wl = valid_wavelength[peak]
            current_peak_refl = current_smoothed[peak]

            # 找到最接近的标准峰值
            if len(ref_peaks) > 0:
                ref_peak_wls = valid_ref_wavelength[ref_peaks]
                closest_idx = np.argmin(np.abs(ref_peak_wls - current_peak_wl))
                closest_ref_peak_wl = ref_peak_wls[closest_idx]
                closest_ref_peak_refl = ref_smoothed[ref_peaks[closest_idx]]
                wl_diff = current_peak_wl - closest_ref_peak_wl

                html_feature_analysis += f"<tr><td>{current_peak_wl:.1f}nm</td><td>{current_peak_refl:.2f}%</td>"
                html_feature_analysis += f"<td>{closest_ref_peak_wl:.1f}nm</td><td>{closest_ref_peak_refl:.2f}%</td>"
                html_feature_analysis += f"<td>{wl_diff:.1f}nm</td></tr>"
        else:
                html_feature_analysis += f"<tr><td>{current_peak_wl:.1f}nm</td><td>{current_peak_refl:.2f}%</td>"
                html_feature_analysis += "<td colspan='3'>标准曲线无峰值</td></tr>"

        html_feature_analysis += "</table>"

        # 谷值比对
        html_feature_analysis += "<p>谷值比对：</p>"
        html_feature_analysis += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        html_feature_analysis += "<tr><th>当前曲线谷值波长</th><th>当前谷值反射率</th><th>最接近的标准谷值波长</th><th>标准谷值反射率</th><th>波长差异</th></tr>"

        for valley in current_valleys:
            current_valley_wl = valid_wavelength[valley]
            current_valley_refl = current_smoothed[valley]

            # 找到最接近的标准谷值
            if len(ref_valleys) > 0:
                ref_valley_wls = valid_ref_wavelength[ref_valleys]
                closest_idx = np.argmin(np.abs(ref_valley_wls - current_valley_wl))
                closest_ref_valley_wl = ref_valley_wls[closest_idx]
                closest_ref_valley_refl = ref_smoothed[ref_valleys[closest_idx]]
                wl_diff = current_valley_wl - closest_ref_valley_wl

                html_feature_analysis += f"<tr><td>{current_valley_wl:.1f}nm</td><td>{current_valley_refl:.2f}%</td>"
                html_feature_analysis += f"<td>{closest_ref_valley_wl:.1f}nm</td><td>{closest_ref_valley_refl:.2f}%</td>"
                html_feature_analysis += f"<td>{wl_diff:.1f}nm</td></tr>"
            else:
                html_feature_analysis += f"<tr><td>{current_valley_wl:.1f}nm</td><td>{current_valley_refl:.2f}%</td>"
                html_feature_analysis += "<td colspan='3'>标准曲线无谷值</td></tr>"

        html_feature_analysis += "</table>"

        # 综合评估
        similarity_score = corr_coef * 100  # 基于相关系数的相似度得分
        feature_similarity = 0

        # 计算特征点相似度
        if len(current_peaks) > 0 and len(ref_peaks) > 0:
            peak_diffs = []
            for peak in current_peaks:
                current_peak_wl = valid_wavelength[peak]
                ref_peak_wls = valid_ref_wavelength[ref_peaks]
                min_diff = np.min(np.abs(ref_peak_wls - current_peak_wl))
                peak_diffs.append(min_diff)

            avg_peak_diff = np.mean(peak_diffs)
            max_allowed_diff = 20  # 最大允许差异 (nm)
            peak_similarity = max(0, 100 * (1 - avg_peak_diff / max_allowed_diff))
            feature_similarity = peak_similarity

        # 综合得分 (相关系数占70%，特征点相似度占30%)
        overall_score = 0.7 * similarity_score
        if feature_similarity > 0:
            overall_score += 0.3 * feature_similarity

        # 添加结论
        html_conclusion = "<h3>综合评估与结论</h3>"

        if overall_score >= 90:
            match_level = "极高"
            conclusion = "当前曲线与标准曲线高度匹配，各项特征基本一致。"
        elif overall_score >= 80:
            match_level = "高"
            conclusion = "当前曲线与标准曲线匹配度高，主要特征相似。"
        elif overall_score >= 70:
            match_level = "中高"
            conclusion = "当前曲线与标准曲线有较好的匹配度，主要特征相似但存在一些差异。"
        elif overall_score >= 60:
            match_level = "中等"
            conclusion = "当前曲线与标准曲线匹配度一般，存在明显差异。"
        else:
            match_level = "低"
            conclusion = "当前曲线与标准曲线差异较大，匹配度低。"

        html_conclusion += f"<p>相似度得分：<b>{overall_score:.1f}</b>/100</p>"
        html_conclusion += f"<p>匹配程度：<b>{match_level}</b></p>"
        html_conclusion += f"<p>结论：{conclusion}</p>"

        # 主要差异点分析
        html_conclusion += "<p><b>主要差异点分析：</b></p><ul>"

        # 找出最大差异的区域
        if region_diff_data:
            max_rel_diff_region = max(region_diff_data, key=lambda x: abs(x[4]))
            _, _, region_name, diff_mean, rel_diff_mean = max_rel_diff_region
            html_conclusion += f"<li>{region_name}区域存在最大相对差异：{rel_diff_mean:.2f}%</li>"

        # 分析峰值差异
        if len(current_peaks) != len(ref_peaks):
            html_conclusion += f"<li>峰值数量不同：当前曲线有{len(current_peaks)}个峰值，标准曲线有{len(ref_peaks)}个峰值</li>"

        # 分析谷值差异
        if len(current_valleys) != len(ref_valleys):
            html_conclusion += f"<li>谷值数量不同：当前曲线有{len(current_valleys)}个谷值，标准曲线有{len(ref_valleys)}个谷值</li>"

        # 最大差异点
        html_conclusion += f"<li>最大差异点在波长{max_diff_wl:.0f}nm处，差异值为{max_diff:.2f}%</li>"

        html_conclusion += "</ul>"

        # 添加建议
        html_conclusion += "<p><b>建议：</b></p>"
        if overall_score >= 80:
            html_conclusion += "<p>当前曲线与标准曲线十分接近，可以考虑作为标准曲线的替代品。</p>"
        elif overall_score >= 70:
            html_conclusion += "<p>当前曲线与标准曲线较为接近，需要关注差异较大的波长区域，评估影响。</p>"
        else:
            html_conclusion += "<p>当前曲线与标准曲线差异明显，建议分析差异原因，或考虑重新采集数据。</p>"

        # 添加到HTML结果
        text_result.setHtml(html_result + html_region_analysis + html_feature_analysis + html_conclusion)

        # 添加到布局
        layout.addWidget(text_result)
        layout.addWidget(fig_container)
        layout.addWidget(canvas3)

        # 添加到标签页
        self.tab_widget.addTab(tab, "标准比对")


class SpectralCanvas(QWidget):
    """用于显示光谱数据的画布类"""

    def __init__(self, width=10, height=6, dpi=100):
        # 初始化QWidget
        super().__init__()

        # 创建MatPlotLib图形和坐标轴
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        # 创建FigureCanvas
        self.canvas = FigureCanvas(self.fig)

        # 确保FigureCanvas能够接收鼠标事件
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setMouseTracking(True)

        # 将FigureCanvas添加到布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 在PySide6中，我们需要使用不同的方法来添加FigureCanvas
        try:
            # 尝试直接添加
            layout.addWidget(self.canvas)
        except TypeError:
            # 如果直接添加失败，尝试将canvas转换为QWidget
            from PySide6.QtWidgets import QWidget
            from shiboken6 import wrapInstance

            # 获取canvas的内部指针，并将其包装为QWidget
            widget_ptr = self.canvas.winId()
            if widget_ptr:
                canvas_widget = wrapInstance(int(widget_ptr), QWidget)
                layout.addWidget(canvas_widget)
            else:
                print("警告: 无法将FigureCanvas添加到布局，图表可能无法显示")

        # 启用鼠标跟踪，以便接收鼠标事件
        self.setMouseTracking(True)

        # 安装事件过滤器，以便捕获FigureCanvas上的事件
        self.canvas.installEventFilter(self)

        # 初始化绘图相关的属性
        self.lines = []  # 存储所有曲线
        self.legend_lines = []  # 图例中的线条
        self.highlighted_line = None  # 当前高亮的线条
        self.original_linewidths = {}  # 原始线宽
        self._original_xlim = None  # 原始x轴范围
        self._original_ylim = None  # 原始y轴范围
        self._show_grid = True      # 网格状态

        # 鼠标拖动相关变量
        self._pan_enabled = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._pan_start_xlim = None
        self._pan_start_ylim = None
        self._current_wavelengths = []  # 用于存储当前波长标记，在拖动过程中使用

        # 启用交互性支持
        # 这些连接使得我们能够对图中的对象进行交互式操作
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # 设置波长标记相关的属性
        self.wavelength_markers = []
        self.wavelength_annotations = []
        self.wavelength_vlines = []
        self.wavelength_labels = []

        # 添加视图变化事件的监听，以更新波长标记
        self.axes.callbacks.connect('xlim_changed', self._on_view_changed)
        self.axes.callbacks.connect('ylim_changed', self._on_view_changed)

        # 设置绘图样式
        self.fig.set_facecolor('white')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('波长 (nm)', fontsize=12)
        self.axes.set_ylabel('反射率 (%)', fontsize=12)

    def _on_view_changed(self, event):
        """当视图范围变化时调用此方法，更新波长标记"""
        # 防止在初始化或清除时触发
        if not hasattr(self, 'wavelength_vlines') or not self.wavelength_vlines:
            return

        # 获取当前所有波长值
        wavelengths = [vline.get_xdata()[0] for vline in self.wavelength_vlines]
        if wavelengths:
            # 使用延迟执行，避免在频繁缩放时过度重绘
            self.delayed_update_wavelength_markers(wavelengths)

    def delayed_update_wavelength_markers(self, wavelengths):
        """延迟更新波长标记，避免频繁重绘"""
        # 清除现有标记
        self.clear_wavelength_markers()
        # 重新添加标记
        self.add_wavelength_markers(wavelengths)
        # 使用draw_idle延迟绘制
        self.canvas.draw_idle()

    def plot_spectra(self, spectra_dict, title="反射率光谱对比图", reference_curve=None):
        """绘制多条光谱曲线

        Args:
            spectra_dict: 包含多个光谱数据的字典 {显示标签: (波长数组, 反射率数组)}
            title: 图表标题
            reference_curve: 标准曲线的名称
        """
        self.axes.clear()
        self.lines = []
        self.legend_lines = []
        self.highlighted_line = None
        self.original_linewidths = {}

        # 添加光谱区域背景
        # 定义常见光谱区域
        regions = [
            (350, 450, "紫外-蓝光"),
            (450, 500, "蓝光"),
            (500, 570, "绿光"),
            (570, 620, "黄光"),
            (620, 750, "红光"),
            (750, 1050, "近红外")
        ]

        # 添加马卡龙色系背景
        region_colors = {
            "紫外-蓝光": "#9370DB",  # 紫罗兰色
            "蓝光": "#6495ED",      # 亮蓝色
            "绿光": "#98FB98",      # 淡绿色
            "黄光": "#FFD700",      # 金黄色
            "红光": "#FF6347",      # 番茄红
            "近红外": "#F08080"     # 浅珊瑚色
        }

        # 获取全局波长范围
        global_min_wl = float('inf')
        global_max_wl = float('-inf')

        for label, (wavelength, reflectance) in spectra_dict.items():
            mask = ~np.isnan(reflectance)
            valid_wavelength = wavelength[mask]
            if len(valid_wavelength) > 0:
                global_min_wl = min(global_min_wl, np.min(valid_wavelength))
                global_max_wl = max(global_max_wl, np.max(valid_wavelength))

        # 确保有有效的全局波长范围
        if global_min_wl != float('inf') and global_max_wl != float('-inf'):
            # 为每个区域添加背景色
            for wl_min, wl_max, region_name in regions:
                # 检查区域是否在全局波长范围内
                if wl_min <= global_max_wl and wl_max >= global_min_wl:
                    # 调整区域边界以适应全局范围
                    effective_wl_min = max(wl_min, global_min_wl)
                    effective_wl_max = min(wl_max, global_max_wl)

                    # 添加区域背景
                    region_color = region_colors.get(region_name, "#DDDDDD")  # 默认为浅灰色
                    self.axes.axvspan(effective_wl_min, effective_wl_max, alpha=0.25, color=region_color)

                    # 为了错开标签，交替使用两种不同的y位置
                    y_pos = 0.02 if regions.index((wl_min, wl_max, region_name)) % 2 == 0 else 0.07

                    # 添加小的区域标签，使用交替的位置
                    self.axes.text((effective_wl_min + effective_wl_max) / 2, y_pos,
                               region_name, ha='center', va='bottom', fontsize=8,
                               transform=self.axes.get_xaxis_transform(), color='black',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor=region_color,
                                        boxstyle='round,pad=0.2', linewidth=1))

        # 获取峰谷标记点显示状态
        show_peakvalley = False
        parent = self.parent()
        if parent and hasattr(parent, 'peakvalley_cb'):
            show_peakvalley = parent.peakvalley_cb.isChecked()

        # 生成不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(spectra_dict)))

        # 找出标准曲线在列表中的位置
        reference_idx = -1
        for i, label in enumerate(spectra_dict.keys()):
            if reference_curve and label.replace(" [标准]", "") == reference_curve:
                reference_idx = i
                break

        # 先绘制非标准曲线
        for i, (label, (wavelength, reflectance)) in enumerate(spectra_dict.items()):
            # 如果是标准曲线，稍后绘制
            if i == reference_idx:
                continue

            # 移除NaN值
            mask = ~np.isnan(reflectance)

            # 绘制曲线，保存线条引用，并设置pickradius使曲线更容易被选中
            line, = self.axes.plot(wavelength[mask], reflectance[mask],
                                  label=label, color=colors[i],
                                  linewidth=1.5, picker=5,
                                  marker=None)  # 明确设置不显示标记点

            # 保存线条引用和原始线宽
            self.lines.append(line)
            self.original_linewidths[line] = line.get_linewidth()

            # 峰谷标记点
            if show_peakvalley:
                from scipy.signal import find_peaks
                x = wavelength[mask]
                y = reflectance[mask]
                # 峰
                peaks, _ = find_peaks(y, distance=5, prominence=0.5)
                # 谷
                valleys, _ = find_peaks(-y, distance=5, prominence=0.5)
                # 绘制峰
                self.axes.plot(x[peaks], y[peaks], 'ro', markersize=7, label=None)
                for px, py in zip(x[peaks], y[peaks]):
                    self.axes.annotate(f'{px:.0f}', (px, py), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8, color='red')
                # 绘制谷
                self.axes.plot(x[valleys], y[valleys], 'bo', markersize=7, label=None)
                for vx, vy in zip(x[valleys], y[valleys]):
                    self.axes.annotate(f'{vx:.0f}', (vx, vy), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8, color='blue')

        # 如果有标准曲线，最后绘制它（使其位于最上层）
        if reference_idx >= 0:
            label = list(spectra_dict.keys())[reference_idx]
            wavelength, reflectance = spectra_dict[label]

            # 移除NaN值
            mask = ~np.isnan(reflectance)

            # 使用更醒目的样式绘制标准曲线
            line, = self.axes.plot(wavelength[mask], reflectance[mask],
                                 label=f"{label} [标准]", color='red',
                                 linewidth=2.5, linestyle='-', picker=5,
                                 marker=None)  # 明确设置不显示标记点

            # 添加标准曲线的指示标签
            last_valid_idx = np.where(mask)[0][-1]
            self.axes.annotate('标准曲线',
                             xy=(wavelength[last_valid_idx], reflectance[last_valid_idx]),
                             xytext=(10, 0), textcoords='offset points',
                             fontsize=9, fontweight='bold', color='red',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

            # 保存线条引用和原始线宽
            self.lines.append(line)
            self.original_linewidths[line] = line.get_linewidth()

            # 峰谷标记点
            if show_peakvalley:
                from scipy.signal import find_peaks
                x = wavelength[mask]
                y = reflectance[mask]
                peaks, _ = find_peaks(y, distance=5, prominence=0.5)
                valleys, _ = find_peaks(-y, distance=5, prominence=0.5)
                self.axes.plot(x[peaks], y[peaks], 'ro', markersize=8, label=None)
                for px, py in zip(x[peaks], y[peaks]):
                    self.axes.annotate(f'{px:.0f}', (px, py), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color='red')
                self.axes.plot(x[valleys], y[valleys], 'bo', markersize=8, label=None)
                for vx, vy in zip(x[valleys], y[valleys]):
                    self.axes.annotate(f'{vx:.0f}', (vx, vy), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='blue')

        self.axes.set_xlabel('波长 (nm)')
        self.axes.set_ylabel('反射率 (%)')
        self.axes.set_title(title)

        # 应用当前的网格状态 - 使用属性而不是直接访问变量
        self.axes.grid(self._show_grid, linestyle='--', alpha=0.7)

        # 创建图例，并设置可点击 - 扁平化设计
        if len(spectra_dict) > 10:
            # 对于大量曲线，将图例放在图外右侧
            legend = self.axes.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.5,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white'  # 白色背景
            )
            self.fig.subplots_adjust(right=0.75)  # 为图例留出空间
        else:
            # 对于少量曲线，使用更美观的扁平化图例样式
            legend = self.axes.legend(
                loc='upper right',
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.0,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white',  # 白色背景
                ncol=1 if len(spectra_dict) <= 5 else 2  # 当曲线数量超过5条时使用两列布局
            )

        # 设置图例可点击
        for legline in legend.get_lines():
            legline.set_picker(10)  # 增大点击检测范围

        # 保存图例中的线条引用
        self.legend_lines = legend.get_lines()

        self.fig.tight_layout()

        # 保存原始视图范围
        self._original_xlim = self.axes.get_xlim()
        self._original_ylim = self.axes.get_ylim()

        self.canvas.draw()

    def reset_view(self):
        """重置视图到原始范围"""
        if self._original_xlim is not None and self._original_ylim is not None:
            self.axes.set_xlim(self._original_xlim)
            self.axes.set_ylim(self._original_ylim)

            # 在重置视图后更新波长标记
            wavelengths = []
            if hasattr(self, 'wavelength_vlines') and self.wavelength_vlines:
                for vline in self.wavelength_vlines:
                    wavelengths.append(vline.get_xdata()[0])

            if wavelengths:
                self.clear_wavelength_markers()
                self.add_wavelength_markers(wavelengths)

            self.canvas.draw()

    def home(self):
        """兼容NavigationToolbar2QT的home方法"""
        self.reset_view()

    def wheelEvent(self, event):
        """处理鼠标滚轮事件 - 仅用于SpectralCanvas本身，不处理FigureCanvas上的事件"""
        # 这个方法现在由eventFilter处理，但我们保留它以防有其他地方调用
        super().wheelEvent(event)

    def on_scroll(self, event):
        """处理matplotlib的滚轮事件（已废弃，改用Qt的wheelEvent）"""
        pass

    @property
    def show_grid(self):
        """获取网格显示状态"""
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value):
        """设置网格显示状态并立即应用"""
        self._show_grid = value
        self.axes.grid(value, linestyle='--', alpha=0.7)
        self.canvas.draw()

    def set_grid_visible(self, visible):
        """设置网格显示状态"""
        self.show_grid = visible

    def clear_plot(self):
        """清除当前图表"""
        self.axes.clear()
        self.lines = []
        self.legend_lines = []
        self.highlighted_line = None
        self.original_linewidths = {}
        # 保持网格状态一致
        self.axes.grid(self._show_grid, linestyle='--', alpha=0.7)
        self.canvas.draw()

    def on_pick(self, event):
        """处理点击事件"""
        # 检查是否点击了图例中的线条
        if isinstance(event.artist, matplotlib.lines.Line2D) and event.artist in self.legend_lines:
            # 获取被点击的图例线条的索引
            legend_line = event.artist
            index = self.legend_lines.index(legend_line)

            if index < len(self.lines):
                # 获取对应的实际曲线
                line = self.lines[index]
                # 切换可见性
                visible = not line.get_visible()
                line.set_visible(visible)

                # 调整图例线条的alpha值以显示状态
                if visible:
                    legend_line.set_alpha(1.0)
                else:
                    legend_line.set_alpha(0.2)

                # 重绘画布
                self.canvas.draw()
                return

        # 为了触发motion_notify_event更新高亮
        self.update_highlight(None)

    def on_motion(self, event):
        """处理鼠标移动事件"""
        if event.inaxes == self.axes:
            # 获取鼠标坐标
            x, y = event.xdata, event.ydata

            closest_line = None
            min_distance = float('inf')

            # 找到最近的线条
            for line in self.lines:
                if not line.get_visible():
                    continue

                # 获取线条数据
                xdata = line.get_xdata()
                ydata = line.get_ydata()

                # 找到最近的点
                distances = []
                for xi, yi in zip(xdata, ydata):
                    # 计算欧几里德距离
                    dist = np.sqrt((x - xi)**2 + (y - yi)**2)
                    distances.append(dist)

                if distances:
                    min_dist = min(distances)
                    if min_dist < min_distance:
                        min_distance = min_dist
                        closest_line = line

            # 设置阈值，确定是否足够近
            threshold = (self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) * 0.02

            if min_distance < threshold:
                self.update_highlight(closest_line)
            else:
                self.update_highlight(None)
        else:
            self.update_highlight(None)

    def update_highlight(self, highlight_line):
        """更新突出显示的曲线

        Args:
            highlight_line: 要突出显示的曲线索引或名称
        """
        # 恢复所有曲线的正常线宽
        for line in self.lines:
            line.set_linewidth(1.5)
            line.set_alpha(0.7)
            line.set_zorder(1)  # 设置普通曲线的z顺序为基础值

        # 如果指定了突出显示的曲线
        if highlight_line is not None:
            # 确定线索引
            line_idx = highlight_line
            if isinstance(highlight_line, str):
                for i, line in enumerate(self.lines):
                    if line.get_label() == highlight_line:
                        line_idx = i
                        break

            # 突出显示指定曲线（如果存在）
            if isinstance(line_idx, int) and 0 <= line_idx < len(self.lines):
                # 增强高亮效果
                self.lines[line_idx].set_linewidth(4.0)  # 增加线宽
                self.lines[line_idx].set_alpha(1.0)      # 完全不透明
                self.lines[line_idx].set_zorder(100)     # 确保高亮线条在最上层
                
                # 设置高亮曲线的标记，使其更加明显
                self.lines[line_idx].set_markeredgewidth(2.0)
                self.lines[line_idx].set_markeredgecolor('white')
                self.lines[line_idx].set_markersize(8)
            elif isinstance(line_idx, matplotlib.lines.Line2D):
                # 直接处理Line2D对象
                line = line_idx
                line.set_linewidth(4.0)
                line.set_alpha(1.0)
                line.set_zorder(100)
                
                # 设置高亮曲线的标记
                line.set_markeredgewidth(2.0)
                line.set_markeredgecolor('white')
                line.set_markersize(8)

        # 更新图例
        self.update_legend()

        # 重绘
        self.canvas.draw()

    def update_legend(self):
        """更新图例以适应当前的曲线和标记状态"""
        # 移除旧图例
        if hasattr(self, 'legend') and self.legend:
            self.legend.remove()

        # 如果有曲线，则创建新图例
        if self.lines:
            # 创建带有当前线宽和透明度的图例 - 扁平化设计
            # 确定图例的列数
            visible_lines = [line for line in self.lines if line.get_visible()]
            ncols = 1 if len(visible_lines) <= 5 else 2

            self.legend = self.axes.legend(
                loc='upper right',
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.0,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white',  # 白色背景
                ncol=ncols
            )

            # 提高图例的z-order以确保它显示在其他元素之上
            if self.legend:
                self.legend.set_zorder(100)

    def add_analysis_markers(self, markers_dict):
        """在图表上添加分析标记

        Args:
            markers_dict: 标记点字典 {名称: [(x1, y1), (x2, y2), ...]}
        """
        # 标记颜色和标记类型
        marker_styles = ['o', 's', '^', 'v', 'D', '*']

        for line_idx, markers in markers_dict.items():
            # 确保索引在有效范围内
            if line_idx < len(self.lines):
                line = self.lines[line_idx]
                line_color = line.get_color()

                for i, (x, y, marker_type, text) in enumerate(markers):
                    # 选择标记样式
                    marker_style = marker_styles[i % len(marker_styles)]

                    # 绘制标记点
                    self.axes.plot(x, y, marker=marker_style, markersize=8,
                                  markerfacecolor='none', markeredgecolor=line_color,
                                  markeredgewidth=2)

                    # 添加标注文本
                    if text:
                        self.axes.annotate(text, xy=(x, y), xytext=(10, 10),
                                          textcoords='offset points', color=line_color,
                                          fontsize=9, fontweight='bold',
                                          arrowprops=dict(arrowstyle='->', color=line_color))

        # 重绘画布
        self.canvas.draw()

    def highlight_curve_segment(self, line_idx, x_start, x_end, color='yellow', alpha=0.3):
        """高亮显示曲线的某一段区域

        Args:
            line_idx: 曲线索引
            x_start: 开始的x坐标(波长)
            x_end: 结束的x坐标(波长)
            color: 高亮颜色
            alpha: 透明度
        """
        if line_idx < 0 or line_idx >= len(self.lines):
            return  # 索引无效

        # 清除之前可能存在的高亮区域
        for collection in self.axes.collections[:]:
            if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                collection.remove()

        line = self.lines[line_idx]
        if not line.get_visible():
            return  # 曲线不可见

        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # 找到在指定波长范围内的点
        mask = (x_data >= x_start) & (x_data <= x_end)
        if np.any(mask):
            # 获取当前轴的y范围
            y_range = self.axes.get_ylim()
            y_min, y_max = y_range

            # 绘制填充区域，边框采用较深的颜色
            self.axes.axvspan(x_start, x_end, color=color, alpha=alpha, zorder=0,
                              edgecolor=color, linewidth=1.5)

            # 确保曲线在高亮区域的上层
            line.set_zorder(10)

            # 添加区域标记，使用更清晰的虚线样式
            self.axes.axvline(x_start, color='black', linestyle='--', alpha=0.6, linewidth=0.8)
            self.axes.axvline(x_end, color='black', linestyle='--', alpha=0.6, linewidth=0.8)

            # 添加波长标签，使用更美观的标注
            self.axes.annotate(f"{x_start:.0f}nm", (x_start, y_min + (y_max-y_min)*0.05),
                              ha='center', va='bottom', fontsize=8, rotation=90, alpha=0.8,
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            self.axes.annotate(f"{x_end:.0f}nm", (x_end, y_min + (y_max-y_min)*0.05),
                              ha='center', va='bottom', fontsize=8, rotation=90, alpha=0.8,
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

            # 居中添加区域名称标签
            region_name = self._get_region_name(x_start, x_end)
            if region_name:
                self.axes.annotate(region_name, ((x_start + x_end)/2, y_max - (y_max-y_min)*0.05),
                                  ha='center', va='top', fontsize=9, fontweight='bold',
                                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

            self.canvas.draw()

    def _get_region_name(self, x_start, x_end):
        """根据波长范围获取区域名称"""
        # 定义常见的光谱区域
        regions = [
            (350, 450, "紫外-蓝光"),
            (450, 500, "蓝光"),
            (500, 570, "绿光"),
            (570, 620, "黄光"),
            (620, 750, "红光"),
            (750, 1050, "近红外")
        ]

        # 查找最匹配的区域
        for wl_min, wl_max, name in regions:
            # 如果起始和结束波长接近某个预定义区域，则返回该区域名称
            if abs(wl_min - x_start) < 10 and abs(wl_max - x_end) < 10:
                return name
        return ""

    def add_wavelength_markers(self, wavelengths):
        """在所有曲线上添加特定波长的标记

        Args:
            wavelengths: 要标记的波长列表
        """
        if not self.lines or not wavelengths:
            return

        # 保存所有添加的标记以便稍后清除
        self.wavelength_markers = []
        self.wavelength_annotations = []
        self.wavelength_vlines = []
        self.wavelength_labels = []

        # 获取y轴范围
        y_range = self.axes.get_ylim()
        y_min, y_max = y_range
        y_height = y_max - y_min

        # 获取x轴范围，用于确定标签的合理位置
        x_min, x_max = self.axes.get_xlim()
        x_width = x_max - x_min

        # 计算合适的标记大小和线宽 - 根据当前视图范围动态调整
        markersize = max(6, min(10, 800 / len(self.lines) / len(wavelengths) * (10 / x_width * 500)))

        # 创建颜色映射，每个波长一个独特的颜色
        cmap = plt.cm.tab10
        wl_colors = [cmap(i % 10) for i in range(len(wavelengths))]

        # Step 1: 为每个波长添加垂直参考线和标注
        for i, wl in enumerate(wavelengths):
            # 如果波长不在当前视图范围内或接近边缘，调整标签位置
            in_view = x_min <= wl <= x_max
            near_left = x_min <= wl < x_min + x_width * 0.1
            near_right = x_max - x_width * 0.1 < wl <= x_max

            # 设置当前波长的基本颜色
            line_color = wl_colors[i]

            # 添加垂直参考线 - 使用更加明显的虚线样式，作为辅助线
            vline = self.axes.axvline(
                x=wl,
                color=line_color,
                linestyle='--',
                linewidth=1.8,  # 更粗的线宽
                alpha=0.85,     # 更高的不透明度
                zorder=5,       # 更高的z轴顺序确保可见性
                # 移除label参数，使垂直线不出现在图例中
            )
            self.wavelength_vlines.append(vline)

            # 动态计算标签位置，确保在缩放时仍然可见
            if in_view:
                # 标签水平位置调整
                if near_left:
                    # 靠近左边界时，将标签偏右
                    h_adjust = 20
                    ha_top = 'left'
                elif near_right:
                    # 靠近右边界时，将标签偏左
                    h_adjust = -20
                    ha_top = 'right'
                else:
                    # 中间位置居中对齐
                    h_adjust = 0
                    ha_top = 'center'

                # 在顶部添加波长标签 - 使用transform='figure'确保在缩放时标签位置固定
                top_label = self.axes.annotate(
                    f"λ={wl:.1f}nm",
                    xy=(wl, y_max),
                    xytext=(h_adjust, 20),  # 水平位置根据上面的逻辑调整
                    textcoords='offset points',
                    xycoords=('data', 'data'),
                    ha=ha_top, va='bottom',
                    fontsize=11,     # 字体大小
                    color=line_color,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        fc="white",
                        ec=line_color,
                        alpha=0.95,
                        linewidth=1.8
                    ),
                    zorder=40
                )
                self.wavelength_labels.append(top_label)

        # Step 2: 为每条曲线添加数据点和标签，只有当波长在视图范围内时才添加
        for line_idx, line in enumerate(self.lines):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            line_color = line.get_color()
            line_label = line.get_label()

            # 为每个波长找到并标记对应点
            for i, target_wl in enumerate(wavelengths):
                # 只有当波长在当前视图范围内时才添加标记
                if x_min <= target_wl <= x_max:
                    # 找到最接近目标波长的点
                    idx = np.abs(x_data - target_wl).argmin()
                    actual_wl = x_data[idx]
                    value = y_data[idx]

                    # 仅当找到的点与目标波长接近时添加标记
                    if abs(actual_wl - target_wl) <= 10:  # 允许10nm以内的误差
                        # 使用对应波长的颜色
                        wl_color = wl_colors[i]

                        # 添加标记点 - 使用更明显的标记样式
                        marker = self.axes.plot(
                            [actual_wl], [value],
                            marker='o',
                            markersize=markersize,  # 动态调整的标记大小
                            markerfacecolor='white',
                            markeredgecolor=line_color,
                            markeredgewidth=2.0,    # 更粗的边框
                            zorder=20               # 更高的z轴顺序
                        )[0]
                        self.wavelength_markers.append(marker)

                        # 计算标签位置和连接线样式
                        # 根据曲线和波长索引计算唯一的偏移位置
                        h_index = (line_idx + i) % 3 - 1  # -1, 0, 1
                        v_index = (line_idx % 4)  # 0, 1, 2, 3

                        # 水平和垂直偏移量
                        h_offset = 20 * h_index  # -20, 0, 20 像素
                        # 非线性增长的垂直偏移确保更好的分散
                        v_offset = 20 + v_index * 15  # 20, 35, 50, 65 像素

                        # 靠近边缘时进行调整
                        if near_left and h_offset < 0:
                            h_offset = abs(h_offset)
                        elif near_right and h_offset > 0:
                            h_offset = -abs(h_offset)

                        # 连接线弯曲度，避免交叉
                        rad = 0.2 * h_index

                        # 添加数据标签 - 仅显示数值，使用带箭头的连接线(只保留1位小数)
                        annotation = self.axes.annotate(
                            f"{value:.1f}",
                            xy=(actual_wl, value),
                            xytext=(h_offset, v_offset),
                            textcoords='offset points',
                            color=line_color,
                            fontsize=10,
                            fontweight='bold',
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                ec=line_color,
                                alpha=0.9,
                                linewidth=1.5
                            ),
                            arrowprops=dict(
                                arrowstyle='->',
                                color=line_color,
                                connectionstyle=f"arc3,rad={rad}",
                                linewidth=1.5,
                                alpha=0.9
                            ),
                            zorder=30
                        )
                        self.wavelength_annotations.append(annotation)

        # 重绘画布
        self.canvas.draw()

    def clear_wavelength_markers(self):
        """清除所有波长标记点"""
        if hasattr(self, 'wavelength_markers'):
            for marker in self.wavelength_markers:
                marker.remove()
            self.wavelength_markers = []

        if hasattr(self, 'wavelength_annotations'):
            for annotation in self.wavelength_annotations:
                annotation.remove()
            self.wavelength_annotations = []

        if hasattr(self, 'wavelength_vlines'):
            for vline in self.wavelength_vlines:
                vline.remove()
            self.wavelength_vlines = []

        if hasattr(self, 'wavelength_labels'):
            for label in self.wavelength_labels:
                label.remove()
            self.wavelength_labels = []

        self.canvas.draw()

    def eventFilter(self, obj, event):
        """事件过滤器，用于处理FigureCanvas上的事件"""
        if obj == self.canvas:
            # 处理鼠标按下事件
            if event.type() == event.Type.MouseButtonPress:
                # 检查是否是Ctrl+左键（用于拖拽）或中键
                if (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.ControlModifier) or \
                   event.button() == Qt.MouseButton.MiddleButton:
                    # 启用拖动模式
                    self._pan_enabled = True
                    self._pan_start_x = event.pos().x()
                    self._pan_start_y = event.pos().y()
                    self._pan_start_xlim = self.axes.get_xlim()
                    self._pan_start_ylim = self.axes.get_ylim()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)  # 修改鼠标指针为抓取手形
                    return True  # 事件已处理

            # 处理鼠标释放事件
            elif event.type() == event.Type.MouseButtonRelease:
                if ((event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.ControlModifier) or
                    event.button() == Qt.MouseButton.MiddleButton) and self._pan_enabled:
                    # 禁用拖动模式
                    self._pan_enabled = False
                    self.setCursor(Qt.CursorShape.ArrowCursor)  # 恢复默认指针

                    # 在拖动结束时更新波长标记
                    if hasattr(self, '_current_wavelengths') and self._current_wavelengths:
                        self.clear_wavelength_markers()
                        self.add_wavelength_markers(self._current_wavelengths)
                        self.canvas.draw_idle()

                    return True  # 事件已处理

            # 处理鼠标移动事件
            elif event.type() == event.Type.MouseMove:
                if self._pan_enabled:
                    # 计算鼠标移动的距离
                    dx = event.pos().x() - self._pan_start_x
                    dy = event.pos().y() - self._pan_start_y

                    # 如果移动距离太小，忽略以提高性能
                    if abs(dx) < 1 and abs(dy) < 1:
                        return True

                    # 获取当前视图范围
                    xmin, xmax = self.axes.get_xlim()
                    ymin, ymax = self.axes.get_ylim()

                    # 计算数据坐标的移动距离
                    width, height = self.canvas.get_width_height()
                    dx_data = (xmax - xmin) * dx / width
                    dy_data = (ymax - ymin) * dy / height

                    # 更新视图范围
                    self.axes.set_xlim(xmin - dx_data, xmax - dx_data)
                    self.axes.set_ylim(ymin + dy_data, ymax + dy_data)

                    # 更新起始位置
                    self._pan_start_x = event.pos().x()
                    self._pan_start_y = event.pos().y()

                    # 更新波长标记 - 使用延迟更新策略提高性能
                    self._current_wavelengths = []
                    if hasattr(self, 'wavelength_vlines') and self.wavelength_vlines:
                        for vline in self.wavelength_vlines:
                            self._current_wavelengths.append(vline.get_xdata()[0])

                    # 使用draw_idle()进行延迟绘制，提高性能
                    self.canvas.draw_idle()
                    return True  # 事件已处理

            # 处理鼠标滚轮事件
            elif event.type() == event.Type.Wheel:
                # 检查是否按下了Ctrl键
                if event.modifiers() & Qt.ControlModifier:
                    try:
                        # 获取原始鼠标位置
                        pos = event.position()

                        # 获取画布尺寸
                        width = self.canvas.width()
                        height = self.canvas.height()

                        # 确保宽度和高度有效
                        if width <= 0 or height <= 0:
                            return False

                        # 计算鼠标位置在画布中的相对位置
                        x_pos = pos.x() / width
                        y_pos = (height - pos.y()) / height  # 注意Y轴方向是反的

                        # 获取当前视图范围
                        ax = self.axes
                        xmin, xmax = ax.get_xlim()
                        ymin, ymax = ax.get_ylim()

                        # 计算鼠标所指向的数据点坐标
                        data_x = xmin + (xmax - xmin) * x_pos
                        data_y = ymin + (ymax - ymin) * y_pos

                        # 计算缩放系数 - 调整为更平滑的缩放体验
                        base_scale = 1.15  # 稍微减小缩放步长，使缩放更平滑

                        # 根据滚轮方向确定缩放方向
                        if event.angleDelta().y() > 0:
                            # 放大
                            scale_factor = 1.0 / base_scale
                        else:
                            # 缩小
                            scale_factor = base_scale

                        # 计算新的数据范围
                        new_width = (xmax - xmin) * scale_factor
                        new_height = (ymax - ymin) * scale_factor

                        # 计算新的坐标范围，确保鼠标位置指向的数据点保持不变
                        new_xmin = data_x - x_pos * new_width
                        new_xmin = max(new_xmin, 0)  # 防止x轴范围小于0
                        new_ymin = data_y - y_pos * new_height

                        # 设置新的坐标范围
                        ax.set_xlim(new_xmin, new_xmin + new_width)
                        ax.set_ylim(new_ymin, new_ymin + new_height)

                        # 保存当前的波长标记，以便在缩放后更新
                        wavelengths = []
                        if hasattr(self, 'wavelength_vlines') and self.wavelength_vlines:
                            # 从垂直线获取波长值
                            for vline in self.wavelength_vlines:
                                wavelengths.append(vline.get_xdata()[0])

                        # 如果有波长标记，先清除再重新添加
                        if wavelengths:
                            self.clear_wavelength_markers()
                            self.add_wavelength_markers(wavelengths)

                        # 使用draw_idle()进行延迟绘制，提高性能
                        self.canvas.draw_idle()
                        return True  # 事件已处理
                    except Exception as e:
                        print(f"缩放出错: {e}")

        # 如果事件未处理，则调用父类方法
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        """处理鼠标按下事件 - 仅用于SpectralCanvas本身，不处理FigureCanvas上的事件"""
        # 检查是否是Ctrl+左键（用于拖拽）或中键
        if (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.ControlModifier) or \
           event.button() == Qt.MouseButton.MiddleButton:
            # 启用拖动模式
            self._pan_enabled = True
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            self._pan_start_xlim = self.axes.get_xlim()
            self._pan_start_ylim = self.axes.get_ylim()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)  # 修改鼠标指针为抓取手形
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件 - 仅用于SpectralCanvas本身，不处理FigureCanvas上的事件"""
        # 这个方法现在由eventFilter处理，但我们保留它以防有其他地方调用
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件 - 仅用于SpectralCanvas本身，不处理FigureCanvas上的事件"""
        # 这个方法现在由eventFilter处理，但我们保留它以防有其他地方调用
        super().mouseMoveEvent(event)


class SpectralViewer(QMainWindow):
    """反射率光谱数据可视化应用程序"""

    def __init__(self):
        super().__init__()

        self.spectra_data = {}  # 用于存储所有已加载的光谱数据
        self.base_folder = ""   # 基础文件夹路径
        self.grid_visible = True  # 添加网格状态跟踪
        self.reference_curve = None  # 存储标准曲线的名称
        self.detail_btn = None  # 查看详细结果按钮

        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('光谱分析工具')

        # 设置主窗口大小为更适合的尺寸
        # 获取屏幕尺寸并设置窗口为屏幕的75%大小（更适中的尺寸）
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.75)
        height = int(screen.height() * 0.75)
        self.resize(width, height)

        # 居中显示窗口
        self.center_window()

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 数据加载区域
        load_group = QGroupBox("数据加载")
        load_layout = QVBoxLayout()

        # 创建上方按钮布局（两个加载按钮并排）
        top_btn_layout = QHBoxLayout()

        # 浏览按钮
        self.load_btn = QPushButton('加载数据文件夹')
        self.load_btn.clicked.connect(self.load_data_folder)
        self.load_btn.setMinimumWidth(120)
        top_btn_layout.addWidget(self.load_btn)

        # 添加加载单个CSV文件的按钮
        self.load_csv_btn = QPushButton('加载单个CSV文件')
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        self.load_csv_btn.setMinimumWidth(120)
        top_btn_layout.addWidget(self.load_csv_btn)

        # 添加按钮布局到主布局
        load_layout.addLayout(top_btn_layout)

        # 添加一些间距
        load_layout.addSpacing(8)

        # 创建下方按钮布局（三个操作按钮并排）
        bottom_btn_layout = QHBoxLayout()

        # 添加刷新按钮
        self.reload_btn = QPushButton('刷新显示')
        self.reload_btn.clicked.connect(self.update_plot)
        self.reload_btn.setEnabled(False)
        bottom_btn_layout.addWidget(self.reload_btn)

        # 清空按钮
        self.clear_btn = QPushButton('清空所有数据')
        self.clear_btn.clicked.connect(self.clear_all_data)
        self.clear_btn.setEnabled(False)
        bottom_btn_layout.addWidget(self.clear_btn)

        # 添加分析差异按钮
        self.analyze_btn = QPushButton('分析差异')
        self.analyze_btn.clicked.connect(self.analyze_spectra)
        self.analyze_btn.setEnabled(False)
        bottom_btn_layout.addWidget(self.analyze_btn)

        # 添加按钮布局到主布局
        load_layout.addLayout(bottom_btn_layout)

        # 添加一些间距
        load_layout.addSpacing(10)

        # 数据文件列表标签
        file_list_label = QLabel("已加载的数据文件 (右键可删除):")
        load_layout.addWidget(file_list_label)

        # 数据文件列表（用于显示已加载的文件，支持右键菜单删除）
        self.file_list = QListWidget()
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        self.file_list.setToolTip("右键点击列表项可以删除文件")
        # 添加选择变更事件处理
        self.file_list.itemSelectionChanged.connect(self.on_file_selection_changed)
        load_layout.addWidget(self.file_list)

        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)

        # 绘图控制区域
        plot_control_group = QGroupBox("绘图控制")
        plot_control_layout = QVBoxLayout()

        # 添加交互提示
        interaction_label = QLabel("交互提示：\n· 点击图例可显示/隐藏对应曲线\n· 鼠标移动到曲线上将高亮显示\n· 右键点击文件名可移除数据\n· 使用Ctrl+鼠标左键拖拽可移动图表\n· 使用Ctrl+鼠标滚轮可缩放图表")
        plot_control_layout.addWidget(interaction_label)

        # 添加其他控制选项
        self.grid_cb = QCheckBox("显示网格")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(self.on_grid_changed)
        plot_control_layout.addWidget(self.grid_cb)

        # 添加峰谷标记点显示选项
        self.peakvalley_cb = QCheckBox("显示峰谷标记点")
        self.peakvalley_cb.setChecked(False)
        self.peakvalley_cb.stateChanged.connect(self.update_plot)
        plot_control_layout.addWidget(self.peakvalley_cb)

        # 添加波长范围设置
        range_layout = QHBoxLayout()
        range_layout.setSpacing(5)  # 设置较小的间距使控件紧凑排列
        range_layout.addWidget(QLabel("波长范围:"))
        range_layout.addWidget(QLabel("从"))
        self.wl_min = QComboBox()
        self.wl_min.addItems(["350", "400", "450", "500", "550", "600", "650", "700", "750", "800"])
        range_layout.addWidget(self.wl_min)
        range_layout.addWidget(QLabel("到"))
        self.wl_max = QComboBox()
        self.wl_max.addItems(["1020", "950", "900", "850", "800", "750", "700", "650", "600"])
        range_layout.addWidget(self.wl_max)
        range_layout.addStretch(1)  # 添加弹性空间使组件左对齐
        self.wl_min.currentTextChanged.connect(self.update_plot)
        self.wl_max.currentTextChanged.connect(self.update_plot)

        plot_control_layout.addLayout(range_layout)

        # 添加波长标记功能
        wavelength_group = QGroupBox("波长标记")
        wavelength_layout = QVBoxLayout()

        # 添加波长选择UI
        wavelength_input_layout = QHBoxLayout()
        wavelength_input_layout.setSpacing(8)  # 设置控件之间的间距

        # 标签
        wl_label = QLabel("输入波长:")
        wavelength_input_layout.addWidget(wl_label)

        # 输入框
        self.wavelength_input = QLineEdit()
        self.wavelength_input.setPlaceholderText("输入波长值，如: 450")
        wavelength_input_layout.addWidget(self.wavelength_input)

        # 添加按钮
        add_btn = QPushButton("添加")
        add_btn.clicked.connect(self.add_wavelength)
        add_btn.setFixedWidth(60)  # 设置固定宽度
        wavelength_input_layout.addWidget(add_btn)

        wavelength_layout.addLayout(wavelength_input_layout)

        # 添加显示当前选择的波长列表
        wl_list_label = QLabel("当前选择的波长:")
        wavelength_layout.addWidget(wl_list_label)

        # 波长列表
        self.wavelength_list = QListWidget()
        self.wavelength_list.setMaximumHeight(120)
        self.wavelength_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.wavelength_list.customContextMenuRequested.connect(self.show_wavelength_context_menu)
        wavelength_layout.addWidget(self.wavelength_list)

        # 添加默认波长
        for default_wl in [532, 671, 780, 1020]:
            self.wavelength_list.addItem(f"{default_wl:.1f}")

        # 添加操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)  # 设置按钮之间的间距

        # 标记按钮
        mark_btn = QPushButton("标记选中波长")
        mark_btn.clicked.connect(self.mark_selected_wavelengths)
        mark_btn.setMinimumWidth(100)

        # 清除按钮
        clear_btn = QPushButton("清除标记")
        clear_btn.clicked.connect(self.clear_wavelength_marks)
        clear_btn.setMinimumWidth(80)

        btn_layout.addWidget(mark_btn)
        btn_layout.addWidget(clear_btn)

        # 添加一些顶部间距
        wavelength_layout.addSpacing(5)
        wavelength_layout.addLayout(btn_layout)

        # 添加波长点分析结果显示区域
        wavelength_layout.addSpacing(5)
        wavelength_layout.addWidget(QLabel("波长点反射率分析结果:"))
        self.wavelength_analysis_result = QTextEdit()
        self.wavelength_analysis_result.setReadOnly(True)
        self.wavelength_analysis_result.setMaximumHeight(150)
        self.wavelength_analysis_result.setStyleSheet("font-family: monospace;")
        wavelength_layout.addWidget(self.wavelength_analysis_result)

        # 应用布局
        wavelength_group.setLayout(wavelength_layout)
        plot_control_layout.addWidget(wavelength_group)

        plot_control_group.setLayout(plot_control_layout)
        control_layout.addWidget(plot_control_group)

        # 添加伸缩空间
        control_layout.addStretch(1)

        # 设置左侧面板的宽度
        control_panel.setMinimumWidth(300)
        control_panel.setMaximumWidth(400)

        # 创建右侧图表区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        # 创建绘图画布
        self.canvas = SpectralCanvas()

        # 创建导航工具栏 - 使用辅助函数创建PySide6兼容的工具栏
        self.toolbar = create_toolbar(self.canvas, plot_panel)

        # 添加到主布局
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # 创建分割器，允许调整两侧面板的大小
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(plot_panel)

        # 设置更合理的初始大小比例 (1:3)
        total_width = self.width()
        splitter.setSizes([int(total_width * 0.25), int(total_width * 0.75)])

        main_layout.addWidget(splitter)

    def center_window(self):
        """将窗口居中显示在屏幕上"""
        # 获取屏幕几何信息
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        # 获取窗口几何信息
        window_geometry = self.frameGeometry()

        # 计算居中位置
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)

        # 移动窗口到居中位置
        self.move(window_geometry.topLeft())

    def show_file_context_menu(self, position):
        """显示文件列表的右键菜单"""
        # 获取选中的项目
        selected_items = self.file_list.selectedItems()

        if not selected_items:
            return

        # 创建右键菜单
        context_menu = QMenu(self)

        # 添加删除动作
        remove_action = QAction("删除选中文件", self)
        remove_action.triggered.connect(self.remove_selected_files)
        context_menu.addAction(remove_action)

        # 添加单条曲线分析选项
        if len(selected_items) == 1:
            analyze_single_action = QAction("分析单条曲线", self)
            analyze_single_action.triggered.connect(self.analyze_single_curve)
            context_menu.addAction(analyze_single_action)

            # 添加设置/取消标准曲线选项
            selected_curve = selected_items[0].text()
            clean_name = selected_curve.replace(" [标准]", "")  # 移除可能存在的标记

            if self.reference_curve == clean_name:
                # 如果当前选中的曲线已经是标准曲线，则显示取消选项
                reference_action = QAction("取消设为标准曲线", self)
                reference_action.triggered.connect(self.remove_reference_curve)
            else:
                # 否则显示设置选项
                reference_action = QAction("设为标准曲线", self)
                reference_action.triggered.connect(lambda: self.set_reference_curve(clean_name))

            context_menu.addAction(reference_action)

        # 如果选中了多个项目，添加全部删除选项
        if len(selected_items) > 1:
            remove_all_selected_action = QAction(f"删除所有选中的 {len(selected_items)} 个文件", self)
            remove_all_selected_action.triggered.connect(self.remove_selected_files)
            context_menu.addAction(remove_all_selected_action)

        # 显示菜单
        context_menu.exec_(self.file_list.mapToGlobal(position))

    def remove_selected_files(self):
        """从列表和数据字典中移除选中的文件"""
        selected_items = self.file_list.selectedItems()

        if not selected_items:
            return

        # 确认是否删除
        message = f"确定要移除选中的 {len(selected_items)} 个文件吗？"
        reply = QMessageBox.question(self, '确认删除', message,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return

        # 移除选中的文件
        for item in selected_items:
            label = item.text()
            clean_label = label.replace(" [标准]", "")

            # 检查是否是标准曲线
            if self.reference_curve == clean_label:
                self.reference_curve = None  # 移除标准曲线引用

            if clean_label in self.spectra_data:
                del self.spectra_data[clean_label]
            elif label in self.spectra_data:
                del self.spectra_data[label]

            # 从列表中移除项目
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

        # 禁用清空按钮（如果没有数据）
        if self.file_list.count() == 0:
            self.clear_btn.setEnabled(False)
            self.reload_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)  # 禁用分析按钮

        # 更新图表
        self.update_plot()

    def clear_all_data(self):
        """清空所有加载的数据"""
        if not self.spectra_data:
            return

        # 确认是否清空
        reply = QMessageBox.question(self, '确认清空', '确定要清空所有已加载的数据吗？',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            return

        # 清空数据
        self.spectra_data.clear()
        self.file_list.clear()
        self.canvas.clear_plot()

        # 清空标准曲线设置
        self.reference_curve = None

        # 禁用按钮
        self.clear_btn.setEnabled(False)
        self.reload_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)  # 禁用分析按钮

        QMessageBox.information(self, "清空完成", "已清空所有数据。")

    def load_data_folder(self):
        """加载数据文件夹并处理所有CSV文件"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择数据文件夹",
                                                      QDir.currentPath())

        if not folder_path:
            return

        # 清空现有数据
        reply = QMessageBox.Yes
        if self.file_list.count() > 0:
            reply = QMessageBox.question(self, '确认操作',
                                       '是否先清空当前已加载的数据？\n点击"是"将清空当前数据\n点击"否"将追加新数据',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            self.file_list.clear()
            self.spectra_data.clear()

        self.base_folder = folder_path

        # 遍历文件夹中的所有子文件夹
        files_loaded = 0
        first_item = None
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        wavelength, reflectance = self.read_spectral_data(file_path)

                        # 获取相对路径并处理为显示标签
                        rel_path = os.path.relpath(file_path, folder_path)
                        # 移除rawdata前缀，如果存在
                        if rel_path.startswith('rawdata'):
                            rel_path = rel_path.replace('rawdata/', '', 1)
                            rel_path = rel_path.replace('rawdata\\', '', 1)

                        # 分解路径获取子文件夹名和文件名
                        path_parts = rel_path.split(os.path.sep)

                        if len(path_parts) > 1:
                            # 有子文件夹的情况
                            subfolder = path_parts[0]
                            filename = path_parts[-1]
                            display_label = f"{subfolder}/{filename}"
                        else:
                            # 直接在根目录下的文件
                            display_label = rel_path

                        # 存储数据
                        self.spectra_data[display_label] = (wavelength, reflectance)

                        # 添加到列表显示
                        item = QListWidgetItem(display_label)
                        self.file_list.addItem(item)
                        
                        # 记录第一个加载的文件
                        if files_loaded == 0:
                            first_item = item
                        
                        files_loaded += 1
                    except Exception as e:
                        print(f"无法读取文件 {file_path}: {e}")

        # 启用按钮
        if files_loaded > 0:
            self.reload_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)  # 启用分析按钮

            # 选中第一个加载的文件
            if first_item:
                self.file_list.setCurrentItem(first_item)

            # 显示加载的文件数量
            QMessageBox.information(self, "数据加载完成",
                                  f"已加载 {files_loaded} 个光谱数据文件，并在图表中显示。")

            # 自动更新图表，显示所有数据
            self.update_plot()
        else:
            QMessageBox.warning(self, "未找到数据",
                              f"在所选文件夹中未找到有效的CSV数据文件。")

    def read_spectral_data(self, file_path):
        """读取光谱数据文件并提取波长和反射率数据

        Args:
            file_path: CSV文件路径

        Returns:
            tuple: (波长数组, 反射率数组)
        """
        # 尝试不同的编码方式读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        lines = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break  # 如果成功读取，则跳出循环
            except UnicodeDecodeError:
                continue

        if lines is None:
            raise ValueError(f"无法以支持的编码格式读取文件: {file_path}")

        # 检查文件是否为简单CSV格式（第一列Wavelength，第二列Reflectance）
        is_simple_format = True
        for line in lines[:10]:  # 检查前10行
            if "---------------BEGIN--------------------" in line or "Wavelength(nm)" in line:
                is_simple_format = False
                break

        # 如果是简单格式，直接解析每一行
        if is_simple_format:
            wavelength = []
            reflectance = []

            for line in lines:
                if ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            wl = float(parts[0])
                            # 处理反射率值，可能有空格
                            ref_str = parts[1].strip()
                            ref = float(ref_str) if ref_str.lower() != 'nan' else np.nan

                            wavelength.append(wl)
                            reflectance.append(ref)
                        except (ValueError, IndexError):
                            continue

            if len(wavelength) > 0:
                return np.array(wavelength), np.array(reflectance)

        # 如果不是简单格式或者简单格式解析失败，使用原有方法解析

        # 查找数据开始的行
        start_line = 0
        for i, line in enumerate(lines):
            if "---------------BEGIN--------------------" in line:
                start_line = i + 2  # 数据从BEGIN行的后两行开始
                break

        if start_line == 0:
            # 如果没有找到BEGIN标记，尝试查找标题行
            for i, line in enumerate(lines):
                if "Wavelength(nm)" in line and "Reflectance(%)" in line:
                    start_line = i + 1
                    break

        if start_line == 0:
            # 如果仍然找不到数据起始位置，尝试将第一行作为数据起始点
            start_line = 0

        # 从确定的行开始解析数据
        data_lines = lines[start_line:]
        wavelength = []
        reflectance = []

        for line in data_lines:
            if ',' in line:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        # 处理反射率值，可能是NaN
                        ref_str = parts[1].strip()
                        ref = float(ref_str) if ref_str.lower() != 'nan' else np.nan

                        wavelength.append(wl)
                        reflectance.append(ref)
                    except (ValueError, IndexError):
                        continue

        return np.array(wavelength), np.array(reflectance)

    def update_plot(self):
        """更新绘图，显示所有光谱数据"""
        if not self.spectra_data:
            self.canvas.clear_plot()
            return

        # 获取波长范围
        try:
            wl_min = float(self.wl_min.currentText())
            wl_max = float(self.wl_max.currentText())

            # 筛选波长范围内的数据
            filtered_data = {}
            for label, (wavelength, reflectance) in self.spectra_data.items():
                # 筛选波长范围
                mask = (wavelength >= wl_min) & (wavelength <= wl_max)
                if np.any(mask):  # 确保有数据点
                    filtered_data[label] = (wavelength[mask], reflectance[mask])

            # 绘制所有光谱数据
            if filtered_data:
                # 传递标准曲线信息
                reference_curve = None
                if self.reference_curve is not None:
                    reference_curve = self.reference_curve.replace(" [标准]", "")

                # 记住当前选中的文件以便在更新后重新高亮
                selected_items = self.file_list.selectedItems()
                selected_curve = None
                if selected_items:
                    selected_curve = selected_items[0].text().replace(" [标准]", "")
                
                # 更新图表
                self.canvas.plot_spectra(filtered_data, reference_curve=reference_curve)

                # 如果有选中文件，高亮对应曲线
                if selected_curve:
                    for line in self.canvas.lines:
                        if line.get_label() == selected_curve:
                            self.canvas.update_highlight(line)
                            break
                        # 模糊匹配
                        elif selected_curve in line.get_label() or line.get_label() in selected_curve:
                            self.canvas.update_highlight(line)
                            break

                # 如果有波长标记，重新添加
                if hasattr(self, 'wavelength_list') and self.wavelength_list.count() > 0:
                    wavelengths = []
                    for i in range(self.wavelength_list.count()):
                        wavelengths.append(float(self.wavelength_list.item(i).text()))

                    if wavelengths:
                        self.canvas.add_wavelength_markers(wavelengths)
        except Exception as e:
            QMessageBox.warning(self, "绘图错误", f"无法绘制图表: {str(e)}")

    def on_grid_changed(self, state):
        """处理网格显示状态改变"""
        is_checked = state == Qt.CheckState.Checked

        # 更新网格状态并立即应用
        self.canvas._show_grid = is_checked

        # 强制清除当前网格
        self.canvas.axes.grid(False)

        # 如果需要显示网格，则重新绘制网格
        if is_checked:
            self.canvas.axes.grid(True, linestyle='--', alpha=0.7)

        # 重绘画布
        self.canvas.draw()

    def analyze_spectra(self):
        """分析光谱数据的差异并生成分析报告"""
        if len(self.spectra_data) < 2:
            QMessageBox.warning(self, "无法分析", "需要至少2个光谱数据才能进行差异分析。")
            return

        try:
            # 获取当前波长范围
            wl_min = float(self.wl_min.currentText())
            wl_max = float(self.wl_max.currentText())

            # 筛选当前显示的数据
            filtered_data = {}
            for label, (wavelength, reflectance) in self.spectra_data.items():
                mask = (wavelength >= wl_min) & (wavelength <= wl_max)
                if np.any(mask):
                    filtered_data[label] = (wavelength[mask], reflectance[mask])

            if not filtered_data or len(filtered_data) < 2:
                QMessageBox.warning(self, "无法分析", "在当前波长范围内至少需要2个有效的光谱数据才能进行差异分析。")
                return

            # 1. 进行统计分析并生成报告
            result_html, markers_dict = self.perform_spectral_analysis(filtered_data)

            # 2. 在图表上标记关键点
            self.canvas.add_analysis_markers(markers_dict)

            # 3. 显示分析结果对话框
            result_dialog = AnalysisResultDialog(self)
            result_dialog.set_result_text(result_html)
            result_dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, "分析错误", f"进行差异分析时出错: {str(e)}")

    def perform_spectral_analysis(self, spectra_dict):
        """执行光谱数据的统计分析

        Args:
            spectra_dict: 包含光谱数据的字典 {显示标签: (波长数组, 反射率数组)}

        Returns:
            tuple: (HTML格式的分析报告文本, 标记点字典)
        """
        # 结果报告
        html_result = "<h2>光谱差异分析报告</h2>"
        html_result += f"<p>分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        html_result += f"<p>数据样本数: {len(spectra_dict)} 个光谱曲线</p>"

        # 标记点字典 - 用于在图上标记关键点
        markers_dict = {}

        # 1. 基本统计分析
        html_result += "<h3>1. 基本统计特征</h3>"
        stats_table = "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        stats_table += "<tr><th>样本名称</th><th>最大反射率(%)</th><th>最小反射率(%)</th><th>平均反射率(%)</th><th>标准差</th></tr>"

        # 计算各曲线的最大值、最小值、均值等统计量
        max_values = []
        min_values = []
        mean_values = []
        std_values = []
        labels = []

        for i, (label, (wavelength, reflectance)) in enumerate(spectra_dict.items()):
            # 过滤掉NaN值
            mask = ~np.isnan(reflectance)
            valid_wavelength = wavelength[mask]
            valid_reflectance = reflectance[mask]

            if len(valid_reflectance) == 0:
                continue

            # 计算统计量
            max_val = np.max(valid_reflectance)
            min_val = np.min(valid_reflectance)
            mean_val = np.mean(valid_reflectance)
            std_val = np.std(valid_reflectance)

            # 保存结果
            max_values.append(max_val)
            min_values.append(min_val)
            mean_values.append(mean_val)
            std_values.append(std_val)
            labels.append(label)

            # 找到最大值和最小值对应的波长
            max_idx = np.argmax(valid_reflectance)
            min_idx = np.argmin(valid_reflectance)
            max_wl = valid_wavelength[max_idx]
            min_wl = valid_wavelength[min_idx]

            # 添加标记点
            markers_dict[i] = [
                (max_wl, max_val, 'max', f"最大值: {max_val:.2f}%"),
                (min_wl, min_val, 'min', f"最小值: {min_val:.2f}%")
            ]

            # 添加到表格
            stats_table += f"<tr><td>{label}</td><td>{max_val:.2f}% (波长={max_wl:.0f}nm)</td>"
            stats_table += f"<td>{min_val:.2f}% (波长={min_wl:.0f}nm)</td>"
            stats_table += f"<td>{mean_val:.2f}%</td><td>{std_val:.2f}</td></tr>"

        stats_table += "</table>"
        html_result += stats_table

        # 添加区域分析部分
        html_result += "<h3>1.1 区域分析</h3>"
        html_result += "<p>各光谱曲线在不同波长区域的反射率特征：</p>"

        # 定义常见的光谱区域
        regions = [
            (350, 450, "紫外-蓝光"),
            (450, 500, "蓝光"),
            (500, 570, "绿光"),
            (570, 620, "黄光"),
            (620, 750, "红光"),
            (750, 1050, "近红外")
        ]

        # 定义马卡龙色系映射，对应各个光谱区域的真实颜色
        region_colors = {
            "紫外-蓝光": "#9370DB",  # 紫罗兰色
            "蓝光": "#6495ED",      # 亮蓝色
            "绿光": "#98FB98",      # 淡绿色
            "黄光": "#FFD700",      # 金黄色
            "红光": "#FF6347",      # 番茄红
            "近红外": "#F08080"     # 浅珊瑚色
        }

        # 创建不同色彩方案
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectra_dict)))

        # 创建单张带有区域划分的图表
        fig = Figure(figsize=(16, 10), dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlabel('波长 (nm)', fontsize=12)
        ax.set_ylabel('反射率 (%)', fontsize=12)
        ax.set_title('区域分析', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 获取全局波长范围
        global_min_wl = float('inf')
        global_max_wl = float('-inf')
        global_min_refl = float('inf')
        global_max_refl = float('-inf')

        for label, (wavelength, reflectance) in spectra_dict.items():
            mask = ~np.isnan(reflectance)
            valid_wavelength = wavelength[mask]
            valid_reflectance = reflectance[mask]

            if len(valid_wavelength) > 0:
                global_min_wl = min(global_min_wl, np.min(valid_wavelength))
                global_max_wl = max(global_max_wl, np.max(valid_wavelength))
                global_min_refl = min(global_min_refl, np.min(valid_reflectance))
                global_max_refl = max(global_max_refl, np.max(valid_reflectance))

        # 为每个区域添加背景色
        for i, (wl_min, wl_max, region_name) in enumerate(regions):
            # 检查区域是否在全局波长范围内
            if wl_min <= global_max_wl and wl_max >= global_min_wl:
                # 调整区域边界以适应全局范围
                effective_wl_min = max(wl_min, global_min_wl)
                effective_wl_max = min(wl_max, global_max_wl)

                # 添加区域背景
                region_color = region_colors.get(region_name, "#DDDDDD")  # 默认为浅灰色
                ax.axvspan(effective_wl_min, effective_wl_max, alpha=0.25, color=region_color)

                # 添加区域标签
                ax.text((effective_wl_min + effective_wl_max) / 2, ax.get_ylim()[1] * 0.95,
                       region_name, ha='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=region_color,
                                boxstyle='round,pad=0.2', linewidth=1))

        # 收集每个区域的分析数据
        region_analysis_data = []
        valid_region_names = []

        # 为每个曲线在每个区域添加分析
        for i, (label, (wavelength, reflectance)) in enumerate(spectra_dict.items()):
            # 过滤NaN值
            mask = ~np.isnan(reflectance)
            valid_wavelength = wavelength[mask]
            valid_reflectance = reflectance[mask]

            if len(valid_reflectance) == 0:
                continue

            # 绘制曲线
            line, = ax.plot(valid_wavelength, valid_reflectance, '-', color=colors[i % len(colors)],
                          label=label, linewidth=2.0)

            # 对每个区域进行分析
            for wl_min, wl_max, region_name in regions:
                # 获取特定区域的数据
                region_mask = (valid_wavelength >= wl_min) & (valid_wavelength <= wl_max)
                if np.any(region_mask) and np.sum(region_mask) >= 3:  # 至少需要3个点
                    region_wl = valid_wavelength[region_mask]
                    region_refl = valid_reflectance[region_mask]

                    # 计算区域统计量
                    mean_val = np.mean(region_refl)
                    std_val = np.std(region_refl)

                    # 查找最大值和最小值及其对应波长
                    max_idx = np.argmax(region_refl)
                    min_idx = np.argmin(region_refl)
                    max_val = region_refl[max_idx]
                    min_val = region_refl[min_idx]
                    max_wl = region_wl[max_idx]
                    min_wl = region_wl[min_idx]

                    # 标记最大值和最小值
                    ax.plot(max_wl, max_val, 'o', color=colors[i % len(colors)], markersize=8, markeredgewidth=1.5, markeredgecolor='white')

                    # 只给每个区域中最大值最高的两条曲线添加标签，避免过多标签重叠
                    region_max_values = [d['max'] for d in region_analysis_data if d.get('region_name') == region_name]
                    if region_max_values:  # 确保列表不为空
                        percentile_value = np.percentile(region_max_values, 75) if len(region_max_values) >= 4 else float('-inf')
                        if len(region_max_values) < 3 or max_val > percentile_value:
                            ax.annotate(f"{label}\n最大: {max_val:.1f}%\n({max_wl:.0f}nm)",
                                      (max_wl, max_val), xytext=(10, 10),
                                      textcoords='offset points', fontsize=9,
                                      color=colors[i % len(colors)], fontweight='bold',
                                      bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[i % len(colors)], pad=2),
                                      arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=1.5))
                    else:
                        # 如果是该区域的第一条曲线，添加标签
                        ax.annotate(f"{label}\n最大: {max_val:.1f}%\n({max_wl:.0f}nm)",
                                   (max_wl, max_val), xytext=(10, 10),
                                   textcoords='offset points', fontsize=9,
                                   color=colors[i % len(colors)], fontweight='bold',
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[i % len(colors)], pad=2),
                                   arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], lw=1.5))

                    # 标记最小值 (但标签较小，以减少视觉混乱)
                    ax.plot(min_wl, min_val, 's', color=colors[i % len(colors)], markersize=8, markeredgewidth=1.5, markeredgecolor='white')

                    # 存储区域分析数据
                    region_analysis_data.append({
                        'region_name': region_name,
                        'wl_min': wl_min,
                        'wl_max': wl_max,
                        'label': label,
                        'mean': mean_val,
                        'std': std_val,
                        'max': max_val,
                        'max_wl': max_wl,
                        'min': min_val,
                        'min_wl': min_wl,
                        'curve_idx': i,
                        'region_wl': region_wl,
                        'region_refl': region_refl
                    })

                    # 记录有效的区域名称
                    if region_name not in valid_region_names:
                        valid_region_names.append(region_name)

                    # 添加标记点到主图
                    if i in markers_dict:
                        markers_dict[i].append(
                            (max_wl, max_val, 'region_max', f"{region_name}最大值: {max_val:.2f}%")
                        )
                    else:
                        markers_dict[i] = [
                            (max_wl, max_val, 'region_max', f"{region_name}最大值: {max_val:.2f}%")
                        ]

        # 设置图例 - 扁平化设计
        if len(spectra_dict) > 10:
            # 对于大量曲线，将图例放在图外右侧
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.5,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white'  # 白色背景
            )
            fig.tight_layout(rect=[0, 0, 0.8, 1])  # 为图例留出空间
        else:
            # 对于少量曲线，使用更美观的扁平化图例样式
            ax.legend(
                loc='upper right',
                fontsize=10,
                frameon=True,
                framealpha=0.7,  # 更透明的背景
                fancybox=False,  # 扁平化设计，不使用圆角
                shadow=False,    # 扁平化设计，不使用阴影
                borderpad=0.6,
                labelspacing=0.8,
                handlelength=2.0,
                handleheight=0.8,
                handletextpad=0.6,
                edgecolor='lightgray',
                facecolor='white',  # 白色背景
                ncol=1 if len(spectra_dict) <= 5 else 2  # 当曲线数量超过5条时使用两列布局
            )
            fig.tight_layout()

        # 设置字体大小
        ax.tick_params(axis='both', which='major', labelsize=11)

        # 将图表嵌入到HTML中
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # 添加可点击放大的图片
        html_result += f'''
        <div style="text-align: center;">
            <a href="data:image/png;base64,{img_str}" target="_blank" title="点击查看大图">
                <img src="data:image/png;base64,{img_str}" alt="区域分析" style="max-width:100%; cursor:zoom-in;">
            </a>
            <p style="font-style:italic; color:#555;">点击图片可查看大图</p>
        </div>
        '''

        # 按区域生成表格并在每个区域表格下显示对应的详细比对图
        if region_analysis_data:  # 检查是否有分析数据
            # 获取所有区域名称
            region_names = list(set(data['region_name'] for data in region_analysis_data))

            for region_name in region_names:
                # 筛选当前区域的数据
                region_data = [data for data in region_analysis_data if data['region_name'] == region_name]

                if not region_data:  # 如果没有数据，跳过
                    continue

                # 获取区域波长范围
                wl_min = region_data[0]['wl_min']
                wl_max = region_data[0]['wl_max']

                html_result += f"<h4>{region_name} ({wl_min}-{wl_max}nm)</h4>"

                # 对区域内的曲线按最大值排序
                region_data.sort(key=lambda x: x['max'], reverse=True)

                # 生成表格
                region_table = "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
                region_table += "<tr><th>样本名称</th><th>平均反射率</th><th>标准差</th>"
                region_table += "<th>最大值</th><th>最大值波长</th><th>最小值</th><th>最小值波长</th></tr>"

                # 填充表格所有数据
                for data in region_data:
                    region_table += f"<tr><td>{data['label']}</td><td>{data['mean']:.2f}%</td><td>{data['std']:.2f}</td>"
                    region_table += f"<td>{data['max']:.2f}%</td><td>{data['max_wl']:.1f}nm</td>"
                    region_table += f"<td>{data['min']:.2f}%</td><td>{data['min_wl']:.1f}nm</td></tr>"

                region_table += "</table>"
                html_result += region_table

                # 在表格下方添加该区域的详细比对图
                try:
                    # 创建该区域的详细图表
                    region_fig = Figure(figsize=(12, 8), dpi=120)
                    region_ax = region_fig.add_subplot(111)
                    region_ax.set_xlabel('波长 (nm)', fontsize=12)
                    region_ax.set_ylabel('反射率 (%)', fontsize=12)
                    region_ax.set_title(f'{region_name} ({wl_min}-{wl_max}nm) 区域详细分析', fontsize=14, fontweight='bold')
                    region_ax.grid(True, linestyle='--', alpha=0.7)

                    # 添加马卡龙色系背景
                    region_colors = {
                        "紫外-蓝光": "#9370DB",  # 紫罗兰色
                        "蓝光": "#6495ED",      # 亮蓝色
                        "绿光": "#98FB98",      # 淡绿色
                        "黄光": "#FFD700",      # 金黄色
                        "红光": "#FF6347",      # 番茄红
                        "近红外": "#F08080"     # 浅珊瑚色
                    }
                    region_color = region_colors.get(region_name, "#DDDDDD")  # 默认为浅灰色
                    region_ax.axvspan(wl_min, wl_max, alpha=0.25, color=region_color)

                    # 获取当前区域内的y值范围
                    all_y_values = []
                    for data in region_data:
                        if 'region_refl' in data and len(data['region_refl']) > 0:
                            all_y_values.extend(data['region_refl'])

                    if all_y_values:
                        y_min = min(all_y_values) * 0.95
                        y_max = max(all_y_values) * 1.05
                        region_ax.set_ylim(y_min, y_max)  # 设置y轴范围以突显差异

                    # 绘制每条曲线
                    for data in region_data:
                        i = data['curve_idx']
                        label = data['label']

                        # 确保数据存在
                        if 'region_wl' not in data or 'region_refl' not in data or len(data['region_wl']) == 0:
                            continue

                        region_wl = data['region_wl']
                        region_refl = data['region_refl']
                        max_wl = data['max_wl']
                        max_val = data['max']
                        min_wl = data['min_wl']
                        min_val = data['min']

                        # 绘制曲线
                        region_ax.plot(region_wl, region_refl, '-', color=colors[i % len(colors)],
                                      label=label, linewidth=2.5)

                        # 标记最大值和最小值
                        region_ax.plot(max_wl, max_val, 'o', color=colors[i % len(colors)],
                                      markersize=10, markeredgewidth=1.5, markeredgecolor='white')
                        region_ax.annotate(f"{label}: {max_val:.2f}%",
                                          (max_wl, max_val), xytext=(5, 5),
                                          textcoords='offset points', fontsize=10,
                                          color=colors[i % len(colors)], fontweight='bold')

                        region_ax.plot(min_wl, min_val, 's', color=colors[i % len(colors)],
                                      markersize=10, markeredgewidth=1.5, markeredgecolor='white')

                    # 设置图例
                    if len(region_data) > 8:
                        region_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
                        region_fig.tight_layout(rect=[0, 0, 0.8, 1])
                    else:
                        region_ax.legend(loc='best', fontsize='large')
                        region_fig.tight_layout()

                    # 设置字体大小
                    region_ax.tick_params(axis='both', which='major', labelsize=11)

                    # 将子图嵌入到HTML中
                    region_buf = io.BytesIO()
                    region_fig.savefig(region_buf, format='png', dpi=120, bbox_inches='tight')
                    region_buf.seek(0)
                    region_img_str = base64.b64encode(region_buf.read()).decode('utf-8')

                    html_result += f'''
                    <div style="text-align: center; margin-top: 20px;">
                        <a href="data:image/png;base64,{region_img_str}" target="_blank" title="点击查看大图">
                            <img src="data:image/png;base64,{region_img_str}" alt="{region_name} 区域详细分析" style="max-width:100%; cursor:zoom-in; border: 1px solid #ddd;">
                        </a>
                    </div>
                    '''
                except Exception as e:
                    print(f"处理区域 {region_name} 详细图表时出错: {str(e)}")
                    # 记录错误但继续处理其他区域

        # 2. 波段特征分析
        html_result += "<h3>2. 关键波段差异</h3>"

        # 如果有多个光谱数据，分析关键波段差异
        if len(labels) >= 2:
            try:
                # 获取所有波长的最小值和最大值
                min_wavelengths = []
                max_wavelengths = []

                for label in labels:
                    wavelength, _ = spectra_dict[label]
                    # 确保波长是数组且有值
                    if len(wavelength) > 0:
                        min_wavelengths.append(np.min(wavelength))
                        max_wavelengths.append(np.max(wavelength))

                if min_wavelengths and max_wavelengths:
                    # 创建公共波长范围
                    wl_start = max(min_wavelengths)  # 取最大的最小值
                    wl_end = min(max_wavelengths)    # 取最小的最大值

                    # 确保波长范围有效
                    if wl_start < wl_end:
                        common_wavelengths = np.linspace(wl_start, wl_end, 1000)

                        # 重采样数据到相同的波长点
                        resampled_data = {}
                        for i, label in enumerate(labels):
                            wavelength, reflectance = spectra_dict[label]
                            # 过滤掉NaN值
                            mask = ~np.isnan(reflectance)
                            valid_wavelength = wavelength[mask]
                            valid_reflectance = reflectance[mask]

                            # 确保有数据点进行插值
                            if len(valid_wavelength) >= 2:
                                # 线性插值到公共波长
                                resampled = np.interp(common_wavelengths,
                                                    valid_wavelength,
                                                    valid_reflectance,
                                                    left=np.nan, right=np.nan)
                                resampled_data[label] = resampled

                        # 如果有足够的数据点且至少有两个光谱，计算显著性差异区域
                        if len(resampled_data) >= 2 and common_wavelengths.size > 10:
                            # 存储显著差异的波段
                            diff_bands = []

                            # 计算每对曲线之间的差异
                            for i in range(len(labels)):
                                for j in range(i+1, len(labels)):
                                    label1, label2 = labels[i], labels[j]
                                    # 确保两个标签都在重采样数据中
                                    if label1 in resampled_data and label2 in resampled_data:
                                        data1 = resampled_data[label1]
                                        data2 = resampled_data[label2]

                                        # 计算差异
                                        diff = data1 - data2

                                        # 找到差异显著的区域
                                        # 使用一个滑动窗口计算局部差异的均值和标准差
                                        window_size = min(50, len(common_wavelengths) // 5)  # 调整窗口大小，确保适应数据
                                        if window_size < 3:  # 窗口太小
                                            continue

                                        # 确保窗口步长至少为1
                                        step_size = max(1, window_size // 2)

                                        for k in range(0, len(common_wavelengths) - window_size, step_size):
                                            window_diff = diff[k:k+window_size]
                                            window_wl = common_wavelengths[k:k+window_size]

                                            # 如果窗口中有NaN值，跳过
                                            if np.isnan(window_diff).any():
                                                continue

                                            window_mean = np.mean(window_diff)
                                            window_std = np.std(window_diff)

                                            # 如果差异足够大，认为是显著差异区域
                                            if abs(window_mean) > 1.5 * np.mean(std_values):
                                                # 获取窗口中心波长
                                                center_wl = np.mean(window_wl)
                                                # 窗口中心在原始数据中的位置
                                                center_idx = np.argmin(np.abs(common_wavelengths - center_wl))

                                                # 计算这个位置上原始数据1和数据2的值
                                                val1 = data1[center_idx]
                                                val2 = data2[center_idx]

                                                if not np.isnan(val1) and not np.isnan(val2):
                                                    # 找到图表中的位置
                                                    line1_idx = i
                                                    line2_idx = j

                                                    # 记录差异区域信息
                                                    diff_bands.append({
                                                        'wavelength': center_wl,
                                                        'diff': abs(val1 - val2),
                                                        'label1': label1,
                                                        'label2': label2,
                                                        'line1_idx': line1_idx,
                                                        'line2_idx': line2_idx,
                                                        'val1': val1,
                                                        'val2': val2
                                                    })

                            # 对差异显著的波段进行排序
                            if diff_bands:
                                diff_bands.sort(key=lambda x: x['diff'], reverse=True)

                                # 最多取前5个差异最显著的波段进行展示
                                top_diff_bands = diff_bands[:5]

                                if top_diff_bands:
                                    html_result += "<p>以下波段显示出显著的反射率差异:</p>"
                                    diff_table = "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
                                    diff_table += "<tr><th>中心波长(nm)</th><th>样本1</th><th>样本2</th><th>反射率差异(%)</th></tr>"

                                    for band in top_diff_bands:
                                        # 添加到表格
                                        diff_table += (f"<tr><td>{band['wavelength']:.0f}</td>"
                                                      f"<td>{band['label1']} ({band['val1']:.2f}%)</td>"
                                                      f"<td>{band['label2']} ({band['val2']:.2f}%)</td>"
                                                      f"<td>{band['diff']:.2f}</td></tr>")

                                        # 添加标记点
                                        if band['line1_idx'] in markers_dict:
                                            markers_dict[band['line1_idx']].append(
                                                (band['wavelength'], band['val1'], 'diff', f"差异点: {band['diff']:.2f}%")
                                            )
                                        else:
                                            markers_dict[band['line1_idx']] = [
                                                (band['wavelength'], band['val1'], 'diff', f"差异点: {band['diff']:.2f}%")
                                            ]

                                        if band['line2_idx'] in markers_dict:
                                            markers_dict[band['line2_idx']].append(
                                                (band['wavelength'], band['val2'], 'diff', f"差异点: {band['diff']:.2f}%")
                                            )
                                        else:
                                            markers_dict[band['line2_idx']] = [
                                                (band['wavelength'], band['val2'], 'diff', f"差异点: {band['diff']:.2f}%")
                                            ]

                                    diff_table += "</table>"
                                    html_result += diff_table
                                else:
                                    html_result += "<p>在当前波长范围内未发现显著的反射率差异。</p>"
                            else:
                                html_result += "<p>在当前波长范围内未发现显著的反射率差异。</p>"
                        else:
                            html_result += "<p>数据点不足以进行波段差异分析。</p>"
                    else:
                        html_result += "<p>光谱数据波长范围不重叠，无法进行波段差异分析。</p>"
                else:
                    html_result += "<p>无法确定波长范围，跳过波段差异分析。</p>"
            except Exception as e:
                html_result += f"<p>波段差异分析过程中出错: {str(e)}</p>"
        else:
            html_result += "<p>需要至少两个样本才能进行波段差异分析。</p>"

        # 3. 峰值分析
        html_result += "<h3>3. 光谱峰值与谷值分析</h3>"

        # 对每个光谱曲线查找峰值和谷值
        for i, (label, (wavelength, reflectance)) in enumerate(spectra_dict.items()):
            try:
                # 过滤掉NaN值
                mask = ~np.isnan(reflectance)
                valid_wavelength = wavelength[mask]
                valid_reflectance = reflectance[mask]

                if len(valid_reflectance) < 10:  # 确保有足够的数据点
                    continue

                # 平滑处理
                window_length = min(11, len(valid_reflectance) // 5 * 2 + 1)  # 确保是奇数
                if window_length < 3:  # 如果窗口长度太小，使用原始数据
                    smoothed = valid_reflectance
                else:
                    # 使用滚动窗口平滑数据
                    smoothed = pd.Series(valid_reflectance).rolling(window=window_length, center=True).mean().values
                    # 填充两端的NaN值
                    nan_mask = np.isnan(smoothed)
                    if np.any(~nan_mask):  # 确保有非NaN值
                        # 用前/后有效值填充开头/结尾的NaN
                        first_valid = np.where(~nan_mask)[0][0]
                        last_valid = np.where(~nan_mask)[0][-1]

                        # 填充开头
                        if first_valid > 0:
                            smoothed[:first_valid] = smoothed[first_valid]

                        # 填充结尾
                        if last_valid < len(smoothed) - 1:
                            smoothed[last_valid+1:] = smoothed[last_valid]
                    else:
                        # 如果全是NaN，则使用原始数据
                        smoothed = valid_reflectance

                # 如果数据点太少，则跳过峰值检测
                if len(smoothed) < 5:
                    continue

                # 查找峰值 - 确保距离参数不大于数据点数
                peak_distance = min(20, len(smoothed) // 4)
                peak_indices, _ = find_peaks(smoothed, height=np.mean(smoothed), distance=peak_distance)

                # 查找谷值 - 同样确保距离参数合理
                trough_indices, _ = find_peaks(-smoothed, height=-np.mean(smoothed), distance=peak_distance)

                # 只保留最显著的3个峰值
                if len(peak_indices) > 0:
                    peak_heights = smoothed[peak_indices]
                    # 确保索引有效
                    valid_peaks = [i for i in range(len(peak_indices)) if peak_indices[i] < len(valid_wavelength)]
                    if valid_peaks:
                        peak_idx_sorted = sorted(valid_peaks, key=lambda i: peak_heights[i], reverse=True)
                        top_peaks = [peak_indices[i] for i in peak_idx_sorted[:min(3, len(peak_idx_sorted))]]

                        # 添加峰值标记
                        for peak_idx in top_peaks:
                            if peak_idx < len(valid_wavelength) and peak_idx < len(valid_reflectance):
                                peak_wl = valid_wavelength[peak_idx]
                                peak_val = valid_reflectance[peak_idx]

                                # 添加到标记字典
                                if i in markers_dict:
                                    markers_dict[i].append((peak_wl, peak_val, 'peak', f"峰值: {peak_val:.2f}%"))
                                else:
                                    markers_dict[i] = [(peak_wl, peak_val, 'peak', f"峰值: {peak_val:.2f}%")]

                # 只保留最显著的3个谷值
                if len(trough_indices) > 0:
                    trough_depths = smoothed[trough_indices]
                    # 确保索引有效
                    valid_troughs = [i for i in range(len(trough_indices)) if trough_indices[i] < len(valid_wavelength)]
                    if valid_troughs:
                        trough_idx_sorted = sorted(valid_troughs, key=lambda i: trough_depths[i])
                        top_troughs = [trough_indices[i] for i in trough_idx_sorted[:min(3, len(trough_idx_sorted))]]

                        # 添加谷值标记
                        for trough_idx in top_troughs:
                            if trough_idx < len(valid_wavelength) and trough_idx < len(valid_reflectance):
                                trough_wl = valid_wavelength[trough_idx]
                                trough_val = valid_reflectance[trough_idx]

                                # 添加到标记字典
                                if i in markers_dict:
                                    markers_dict[i].append((trough_wl, trough_val, 'trough', f"谷值: {trough_val:.2f}%"))
                                else:
                                    markers_dict[i] = [(trough_wl, trough_val, 'trough', f"谷值: {trough_val:.2f}%")]
            except Exception as e:
                # 记录错误但继续处理其他曲线
                print(f"处理曲线 {label} 的峰谷分析时出错: {str(e)}")
                continue

        # 4. 总结分析
        html_result += "<h3>4. 分析总结</h3>"

        if len(labels) >= 2:
            try:
                # 计算均值差异
                mean_diff = max(mean_values) - min(mean_values)
                # 计算标准差的均值
                avg_std = np.mean(std_values)

                # 添加总结内容
                html_result += f"<p>所分析的 {len(labels)} 个光谱样本中，平均反射率差异为 {mean_diff:.2f}%，"
                html_result += f"标准差平均值为 {avg_std:.2f}。</p>"

                # 基于统计特征生成简要总结
                if mean_diff > 3 * avg_std:
                    html_result += "<p>分析结论：光谱反射率存在<strong>显著差异</strong>，不同样本之间的光谱特征差别较大。</p>"
                elif mean_diff > avg_std:
                    html_result += "<p>分析结论：光谱反射率存在<strong>一定差异</strong>，但总体趋势相似。</p>"
                else:
                    html_result += "<p>分析结论：光谱反射率<strong>差异较小</strong>，各样本光谱特征非常接近。</p>"

                # 添加波长范围特征分析
                regions = [
                    (350, 450, "紫外-蓝光"),
                    (450, 500, "蓝光"),
                    (500, 570, "绿光"),
                    (570, 620, "黄光"),
                    (620, 750, "红光"),
                    (750, 1050, "近红外")
                ]

                # 计算各区域的平均差异
                html_result += "<p>不同波长区域的差异特征：</p>"
                html_result += "<ul>"

                for wl_min, wl_max, region_name in regions:
                    try:
                        # 检查是否有数据在此区域
                        region_diffs = []

                        # 遍历每对光谱
                        for i in range(len(labels)):
                            for j in range(i+1, len(labels)):
                                label1, label2 = labels[i], labels[j]
                                wl1, ref1 = spectra_dict[label1]
                                wl2, ref2 = spectra_dict[label2]

                                # 获取特定波长范围的数据
                                mask1 = (wl1 >= wl_min) & (wl1 <= wl_max)
                                mask2 = (wl2 >= wl_min) & (wl2 <= wl_max)

                                if np.any(mask1) and np.any(mask2) and np.sum(mask1) > 1 and np.sum(mask2) > 1:
                                    # 确保有足够的数据点
                                    wl1_region = wl1[mask1]
                                    ref1_region = ref1[mask1]
                                    wl2_region = wl2[mask2]
                                    ref2_region = ref2[mask2]

                                    # 计算公共波长范围
                                    common_min = max(np.min(wl1_region), np.min(wl2_region))
                                    common_max = min(np.max(wl1_region), np.max(wl2_region))

                                    if common_min < common_max:
                                        # 重采样到相同波长点进行比较
                                        common_wl = np.linspace(common_min, common_max, 50)

                                        ref1_interp = np.interp(common_wl, wl1_region, ref1_region)
                                        ref2_interp = np.interp(common_wl, wl2_region, ref2_region)

                                        # 计算均方根差异
                                        rmse = np.sqrt(np.mean((ref1_interp - ref2_interp) ** 2))
                                        region_diffs.append(rmse)

                        if region_diffs:
                            avg_region_diff = np.mean(region_diffs)
                            if avg_region_diff > 2 * avg_std:
                                html_result += f"<li><strong>{region_name} ({wl_min}-{wl_max}nm)</strong>: 差异显著 (均方根差异: {avg_region_diff:.2f}%)</li>"
                            elif avg_region_diff > avg_std:
                                html_result += f"<li><strong>{region_name} ({wl_min}-{wl_max}nm)</strong>: 存在一定差异 (均方根差异: {avg_region_diff:.2f}%)</li>"
                            else:
                                html_result += f"<li><strong>{region_name} ({wl_min}-{wl_max}nm)</strong>: 差异较小 (均方根差异: {avg_region_diff:.2f}%)</li>"
                    except Exception as e:
                        html_result += f"<li><strong>{region_name} ({wl_min}-{wl_max}nm)</strong>: 区域分析时出错: {str(e)}</li>"

                html_result += "</ul>"
            except Exception as e:
                html_result += f"<p>总结分析时出错: {str(e)}</p>"
        else:
            html_result += "<p>需要至少两个样本才能进行差异分析对比。</p>"

        # 5. 方法说明
        html_result += "<h3>5. 分析方法说明</h3>"
        html_result += """
        <p>本分析采用以下方法：</p>
        <ul>
            <li><strong>基本统计特征</strong>：计算每个光谱的最大值、最小值、平均值和标准差</li>
            <li><strong>差异显著性分析</strong>：使用滑动窗口法检测光谱之间的显著差异波段</li>
            <li><strong>峰谷分析</strong>：使用信号处理方法检测光谱曲线中的峰值和谷值</li>
            <li><strong>区域分析</strong>：基于常见波长区域划分，分析不同区域的差异特征</li>
        </ul>
        <p>※注：图表中已标记重要的特征点，包括最大值、最小值、显著差异点、峰值和谷值。</p>
        """

        return html_result, markers_dict

    def analyze_single_curve(self):
        """分析单条光谱曲线的详细特征"""
        selected_item = self.file_list.selectedItems()[0]
        curve_name = selected_item.text()

        if curve_name not in self.spectra_data:
            QMessageBox.warning(self, "无法分析", f"未找到选中曲线 '{curve_name}' 的数据。")
            return

        # 获取当前波长范围
        wl_min = float(self.wl_min.currentText())
        wl_max = float(self.wl_max.currentText())

        # 获取曲线数据并筛选波长范围
        wavelength, reflectance = self.spectra_data[curve_name]
        mask = (wavelength >= wl_min) & (wavelength <= wl_max)

        filtered_data = (wavelength[mask], reflectance[mask])

        # 查找曲线在当前显示中的索引
        curve_index = -1
        for i, label in enumerate(self.canvas.axes.get_legend().get_texts()):
            if label.get_text() == curve_name:
                curve_index = i
                break

        if np.sum(mask) < 5:
            QMessageBox.warning(self, "无法分析", f"当前波长范围内的数据点不足，无法进行分析。")
            return

        # 创建并显示单条曲线分析对话框
        dialog = SingleCurveAnalysisDialog(self, filtered_data, curve_name, curve_index)
        dialog.exec_()

    def load_csv_file(self):
        """加载单个CSV文件并解析数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择CSV文件",
            QDir.currentPath(),
            "CSV文件 (*.csv);;所有文件 (*)"
        )

        if not file_path:
            return

        try:
            # 读取文件
            wavelength, reflectance = self.read_spectral_data(file_path)

            # 生成显示标签（使用文件名）
            file_name = os.path.basename(file_path)
            display_label = file_name

            # 检查是否已经加载相同名称的文件
            count = 0
            base_label = display_label
            while display_label in self.spectra_data:
                count += 1
                display_label = f"{base_label} ({count})"

            # 存储数据
            self.spectra_data[display_label] = (wavelength, reflectance)

            # 添加到列表显示
            item = QListWidgetItem(display_label)
            self.file_list.addItem(item)
            
            # 选中新添加的条目
            self.file_list.setCurrentItem(item)

            # 启用按钮
            self.reload_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)

            # 显示成功消息
            QMessageBox.information(self, "数据加载完成",
                                  f"已加载CSV文件: {file_name}")

            # 自动更新图表
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "加载错误",
                                f"无法加载文件 {file_path}:\n{str(e)}")

    def set_reference_curve(self, curve_name):
        """设置指定曲线为标准曲线

        Args:
            curve_name: 标准曲线的名称
        """
        # 如果已有标准曲线且不是当前选择的曲线，则先取消原标准曲线
        if self.reference_curve is not None and self.reference_curve != curve_name:
            # 更新文件列表项显示
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.text() == self.reference_curve:
                    # 移除之前项目的标准曲线标记
                    item.setText(item.text().replace(" [标准]", ""))
                    break

        # 设置新的标准曲线
        self.reference_curve = curve_name

        # 更新文件列表项显示
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == curve_name:
                # 添加标准曲线标记
                if not " [标准]" in item.text():
                    item.setText(f"{item.text()} [标准]")
                item.setForeground(QBrush(QColor("red")))  # 设置文本为红色
                break

        # 更新图表以突出显示标准曲线
        self.update_plot()

        QMessageBox.information(self, "设置标准曲线", f"已将 '{curve_name}' 设为标准曲线")

    def remove_reference_curve(self):
        """取消当前标准曲线的设置"""
        if self.reference_curve is None:
            return

        # 更新文件列表项显示
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if self.reference_curve in item.text():
                # 移除标准曲线标记
                item.setText(item.text().replace(" [标准]", ""))
                item.setForeground(QBrush(QColor("black")))  # 恢复默认文本颜色
                break

        # 清空标准曲线设置
        self.reference_curve = None

        # 更新图表
        self.update_plot()

    def add_wavelength(self):
        """添加波长到列表"""
        text = self.wavelength_input.text().strip()
        try:
            # 尝试将输入解析为波长值
            wavelength = float(text)

            # 检查是否在有效范围内
            wl_min = float(self.wl_min.currentText())
            wl_max = float(self.wl_max.currentText())

            if wavelength < wl_min or wavelength > wl_max:
                QMessageBox.warning(self, "无效波长",
                                 f"波长必须在当前显示范围 ({wl_min}nm - {wl_max}nm) 内")
                return

            # 检查是否已存在
            for i in range(self.wavelength_list.count()):
                if abs(float(self.wavelength_list.item(i).text()) - wavelength) < 0.1:
                    QMessageBox.information(self, "波长已存在",
                                         f"波长 {wavelength}nm 已在列表中")
                    return

            # 添加到列表
            self.wavelength_list.addItem(f"{wavelength:.1f}")

            # 清除输入框
            self.wavelength_input.clear()

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的波长数值")

    def show_wavelength_context_menu(self, position):
        """显示波长列表的右键菜单"""
        if self.wavelength_list.count() == 0:
            return

        context_menu = QMenu(self)

        # 删除选中项
        remove_action = QAction("删除选中波长", self)
        remove_action.triggered.connect(self.remove_selected_wavelength)
        context_menu.addAction(remove_action)

        # 清空所有
        if self.wavelength_list.count() > 1:
            clear_all_action = QAction("清空所有波长", self)
            clear_all_action.triggered.connect(self.clear_all_wavelengths)
            context_menu.addAction(clear_all_action)

        context_menu.exec_(self.wavelength_list.mapToGlobal(position))

    def remove_selected_wavelength(self):
        """移除选中的波长"""
        for item in self.wavelength_list.selectedItems():
            self.wavelength_list.takeItem(self.wavelength_list.row(item))

    def clear_all_wavelengths(self):
        """清空所有波长"""
        self.wavelength_list.clear()

    def mark_selected_wavelengths(self):
        """标记所有选择的波长，并分析不同曲线在这些波长点的反射率值"""
        if not self.spectra_data:
            QMessageBox.warning(self, "无数据", "请先加载光谱数据")
            return

        # 获取所有波长
        wavelengths = []
        for i in range(self.wavelength_list.count()):
            wavelengths.append(float(self.wavelength_list.item(i).text()))

        if not wavelengths:
            QMessageBox.information(self, "未选择波长", "请先添加要标记的波长")
            return

        # 先清除现有标记
        self.canvas.clear_wavelength_markers()

        # 添加新标记
        self.canvas.add_wavelength_markers(wavelengths)

        # 获取当前可见的曲线
        visible_lines = []
        visible_labels = []
        for i, line in enumerate(self.canvas.lines):
            if line.get_visible():
                visible_lines.append(line)
                visible_labels.append(line.get_label())

        if len(visible_lines) < 1:
            self.wavelength_analysis_result.setHtml("<p style='color:red'>无可见曲线，无法分析</p>")
            return

        # 执行波长点反射率分析
        analysis_results = self.analyze_wavelength_points(wavelengths, visible_lines, visible_labels)

        # 创建简化版结果显示在小窗口中
        simplified_results = self.get_simplified_analysis(wavelengths, visible_lines, visible_labels)
        self.wavelength_analysis_result.setHtml(simplified_results)

        # 添加查看详细结果按钮
        if not hasattr(self, 'detail_btn') or self.detail_btn is None:
            self.detail_btn = QPushButton("查看详细分析结果")
            layout = self.wavelength_analysis_result.parentWidget().layout()
            layout.addWidget(self.detail_btn)

        # 连接查看详细结果事件
        self.detail_btn.clicked.disconnect() if hasattr(self.detail_btn, 'clicked') else None
        self.detail_btn.clicked.connect(lambda: self.show_detailed_analysis(analysis_results))

    def analyze_wavelength_points(self, wavelengths, lines, labels):
        """分析不同曲线在指定波长点的反射率值差异

        Args:
            wavelengths: 波长点列表
            lines: 曲线对象列表
            labels: 曲线标签列表

        Returns:
            HTML格式的分析结果
        """
        # 创建分析结果HTML
        html_result = ""

        # 获取每条曲线在每个波长点的反射率值
        wavelength_values = {}

        for wl in wavelengths:
            wavelength_values[wl] = {}

            for i, line in enumerate(lines):
                label = labels[i]

                # 获取曲线数据
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                # 找到最接近波长点的值
                idx = np.abs(x_data - wl).argmin()
                closest_wl = x_data[idx]
                value = y_data[idx]

                # 判断找到的点与目标波长的接近程度
                if abs(closest_wl - wl) <= 10:  # 允许10nm以内的误差
                    wavelength_values[wl][label] = (value, closest_wl)

        # 生成波长点分析表格
        html_result += "<style>table {width: 100%; border-collapse: collapse;} "
        html_result += "th, td {border: 1px solid #ddd; padding: 4px; text-align: right;} "
        html_result += "th {background-color: #f2f2f2;} "
        html_result += ".diff-positive {color: green;} .diff-negative {color: red;}</style>"

        # 为每个波长点创建一个表格
        for wl in wavelengths:
            html_result += f"<h4>波长: {wl:.1f}nm 反射率比对</h4>"

            # 检查是否有有效数据
            if not wavelength_values[wl]:
                html_result += "<p>在此波长附近无有效数据点</p>"
                continue

            # 创建表格
            html_result += "<table>"
            html_result += "<tr><th>曲线名称</th><th>实际波长(nm)</th><th>反射率(%)</th>"

            # 如果有多条曲线，添加差值列
            if len(wavelength_values[wl]) > 1:
                html_result += "<th>与平均值差异</th>"

                # 计算平均反射率
                avg_value = np.mean([value for value, _ in wavelength_values[wl].values()])

                # 添加表头行结束标签
                html_result += "</tr>"

                # 添加每条曲线的数据行
                for label, (value, actual_wl) in wavelength_values[wl].items():
                    diff = value - avg_value
                    diff_class = "diff-positive" if diff >= 0 else "diff-negative"

                    html_result += f"<tr><td>{label}</td><td>{actual_wl:.2f}</td><td>{value:.2f}</td>"
                    html_result += f"<td class='{diff_class}'>{diff:+.2f}</td></tr>"

                # 添加平均值行
                html_result += f"<tr><td colspan='2'><b>平均值</b></td><td><b>{avg_value:.2f}</b></td><td>-</td></tr>"
            else:
                # 只有一条曲线，不显示差异
                html_result += "</tr>"

                # 添加单条曲线的数据行
                label, (value, actual_wl) = list(wavelength_values[wl].items())[0]
                html_result += f"<tr><td>{label}</td><td>{actual_wl:.2f}</td><td>{value:.2f}</td></tr>"

            html_result += "</table>"

            # 如果有多条曲线，添加曲线间两两比对
            if len(wavelength_values[wl]) > 1:
                labels_list = list(wavelength_values[wl].keys())

                html_result += "<h5>曲线间差异比对</h5>"
                html_result += "<table><tr><th>曲线对比</th><th>反射率差异(%)</th><th>相对差异(%)</th></tr>"

                for i in range(len(labels_list)):
                    for j in range(i+1, len(labels_list)):
                        label1 = labels_list[i]
                        label2 = labels_list[j]

                        value1, _ = wavelength_values[wl][label1]
                        value2, _ = wavelength_values[wl][label2]

                        abs_diff = value1 - value2
                        rel_diff = abs_diff / value2 * 100 if value2 != 0 else 0

                        diff_class = "diff-positive" if abs_diff >= 0 else "diff-negative"

                        html_result += f"<tr><td>{label1} vs {label2}</td>"
                        html_result += f"<td class='{diff_class}'>{abs_diff:+.2f}</td>"
                        html_result += f"<td class='{diff_class}'>{rel_diff:+.2f}%</td></tr>"

                html_result += "</table>"

        return html_result

    def clear_wavelength_marks(self):
        """清除所有波长标记和分析结果"""
        self.canvas.clear_wavelength_markers()
        # 清空分析结果显示
        self.wavelength_analysis_result.clear()

    def get_simplified_analysis(self, wavelengths, lines, labels):
        """获取简化版的分析结果，适合在小空间显示

        Args:
            wavelengths: 波长点列表
            lines: 曲线对象列表
            labels: 曲线标签列表

        Returns:
            HTML格式的简化分析结果
        """
        # 创建分析结果HTML
        html_result = ""

        # 获取每条曲线在每个波长点的反射率值
        wavelength_values = {}

        for wl in wavelengths:
            wavelength_values[wl] = {}

            for i, line in enumerate(lines):
                label = labels[i]

                # 获取曲线数据
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                # 找到最接近波长点的值
                idx = np.abs(x_data - wl).argmin()
                closest_wl = x_data[idx]
                value = y_data[idx]

                # 判断找到的点与目标波长的接近程度
                if abs(closest_wl - wl) <= 10:  # 允许10nm以内的误差
                    wavelength_values[wl][label] = (value, closest_wl)

        # 生成波长点分析表格
        html_result += "<style>table {width: 100%; border-collapse: collapse;} "
        html_result += "th, td {border: 1px solid #ddd; padding: 2px; text-align: right; font-size: 9pt;} "
        html_result += "th {background-color: #f2f2f2;} "
        html_result += ".diff-positive {color: green;} .diff-negative {color: red;}</style>"

        # 添加简要说明
        html_result += "<p style='font-size: 9pt;'><b>波长点反射率分析 (点击\"查看详细结果\"获取完整报告)</b></p>"

        # 为每个波长创建简化表格
        for wl in wavelengths:
            if not wavelength_values[wl]:
                continue

            html_result += f"<p style='font-size: 9pt; margin: 2px 0;'><b>λ={wl:.1f}nm</b></p>"
            html_result += "<table>"
            html_result += "<tr><th>曲线</th><th>反射率(%)</th></tr>"

            # 只显示曲线名和反射率值
            for label, (value, _) in wavelength_values[wl].items():
                short_label = label[:15] + "..." if len(label) > 15 else label
                html_result += f"<tr><td>{short_label}</td><td>{value:.2f}</td></tr>"

            html_result += "</table>"

        return html_result

    def show_detailed_analysis(self, analysis_html):
        """显示详细分析结果对话框

        Args:
            analysis_html: 完整的HTML分析结果
        """
        dialog = WavelengthAnalysisDialog(self)
        dialog.set_html_content(analysis_html)
        dialog.exec_()

    def on_file_selection_changed(self):
        """处理文件列表选择变更事件"""
        selected_items = self.file_list.selectedItems()
        
        # 如果没有选中项，或者没有绘图画布，清除高亮并返回
        if not selected_items or not hasattr(self, 'canvas') or self.canvas is None:
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.update_highlight(None)  # 清除高亮
            return
            
        # 获取选中的文件名（如果是多选，只取第一个）
        selected_curve = selected_items[0].text()
        clean_name = selected_curve.replace(" [标准]", "")  # 移除可能存在的标记
        
        print(f"选中文件: {clean_name}")  # 调试信息
        
        # 查找对应的曲线并高亮显示
        curve_found = False
        line_to_highlight = None
        
        if hasattr(self.canvas, 'lines') and self.canvas.lines:
            # 打印当前所有曲线标签以便调试
            print(f"当前曲线标签: {[line.get_label() for line in self.canvas.lines]}")
            
            # 查找匹配的曲线
            for line in self.canvas.lines:
                print(f"比较: '{line.get_label()}' 与 '{clean_name}'")  # 调试信息
                if line.get_label() == clean_name:
                    line_to_highlight = line
                    curve_found = True
                    print(f"找到匹配曲线: {clean_name}")  # 调试信息
                    break
            
            # 如果没有找到精确匹配，尝试模糊匹配（检查是否包含）
            if not curve_found:
                for line in self.canvas.lines:
                    if clean_name in line.get_label() or line.get_label() in clean_name:
                        line_to_highlight = line
                        curve_found = True
                        print(f"找到模糊匹配曲线: {line.get_label()} <-> {clean_name}")  # 调试信息
                        break
            
            # 高亮显示曲线或清除高亮
            if curve_found and line_to_highlight:
                self.canvas.update_highlight(line_to_highlight)
            else:
                print(f"未找到匹配曲线: {clean_name}")  # 调试信息
                self.canvas.update_highlight(None)
        else:
            print("画布上没有曲线或lines属性不存在")  # 调试信息


class WavelengthAnalysisDialog(BaseAnalysisDialog):
    """波长点反射率分析结果对话框"""

    def __init__(self, parent=None, title="波长点反射率分析结果"):
        super().__init__(parent, title)

        # 创建文本显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Microsoft YaHei", 10))

        # 添加导出按钮
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.export_btn)
        button_layout.addStretch(1)

        # 将组件添加到内容布局
        self.content_layout.addWidget(self.result_text)
        self.content_layout.addLayout(button_layout)

        # 设置窗口大小
        self.resize(800, 600)

    def set_html_content(self, html):
        """设置HTML内容"""
        self.result_text.setHtml(html)

    def export_results(self):
        """导出分析结果"""
        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存分析结果",
                QDir.currentPath() + "/波长点分析结果.html",
                "HTML文件 (*.html);;文本文件 (*.txt);;所有文件 (*)"
            )

            if not file_path:
                return

            # 根据文件扩展名选择格式
            if file_path.lower().endswith('.html'):
                # 保存为HTML
                content = self.result_text.toHtml()
            else:
                # 保存为纯文本
                content = self.result_text.toPlainText()

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            QMessageBox.information(self, "导出成功", f"分析结果已成功导出到 {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出分析结果时出错：{str(e)}")

    def mark_selected_wavelengths(self):
        """标记所有选择的波长，并分析不同曲线在这些波长点的反射率值"""
        if not self.spectra_data:
            QMessageBox.warning(self, "无数据", "请先加载光谱数据")
            return

        # 获取所有波长
        wavelengths = []
        for i in range(self.wavelength_list.count()):
            wavelengths.append(float(self.wavelength_list.item(i).text()))

        if not wavelengths:
            QMessageBox.information(self, "未选择波长", "请先添加要标记的波长")
            return

        # 先清除现有标记
        self.canvas.clear_wavelength_markers()

        # 添加新标记
        self.canvas.add_wavelength_markers(wavelengths)

        # 获取当前可见的曲线
        visible_lines = []
        visible_labels = []
        for i, line in enumerate(self.canvas.lines):
            if line.get_visible():
                visible_lines.append(line)
                visible_labels.append(line.get_label())

        if len(visible_lines) < 1:
            self.wavelength_analysis_result.setHtml("<p style='color:red'>无可见曲线，无法分析</p>")
            return

        # 执行波长点反射率分析
        analysis_results = self.analyze_wavelength_points(wavelengths, visible_lines, visible_labels)

        # 在小结果框中显示简化版分析结果
        simplified_results = self.get_simplified_analysis(wavelengths, visible_lines, visible_labels)
        self.wavelength_analysis_result.setHtml(simplified_results)

        # 添加"查看详细结果"按钮到结果区域下方
        result_area = self.wavelength_analysis_result.parentWidget()

        # 检查是否已存在按钮，避免重复添加
        detail_btn_found = False
        for child in result_area.findChildren(QPushButton):
            if child.text() == "查看详细结果":
                detail_btn_found = True
                self.detail_btn = child
                break

        if not detail_btn_found:
            # 创建新按钮并添加到布局
            self.detail_btn = QPushButton("查看详细结果")
            self.detail_btn.clicked.connect(lambda: self.show_detailed_analysis(analysis_results))
            # 获取结果区域的父布局
            parent_layout = None
            for layout in result_area.parent().findChildren(QVBoxLayout):
                # 找到包含结果区域的布局
                for i in range(layout.count()):
                    if layout.itemAt(i).widget() == result_area:
                        parent_layout = layout
                        break

            if parent_layout:
                # 在结果区域之后添加按钮
                idx = parent_layout.indexOf(result_area) + 1
                parent_layout.insertWidget(idx, self.detail_btn)
        else:
            # 更新现有按钮的点击事件
            try:
                self.detail_btn.clicked.disconnect()
            except:
                pass
            self.detail_btn.clicked.connect(lambda: self.show_detailed_analysis(analysis_results))

    def get_simplified_analysis(self, wavelengths, lines, labels):
        """获取简化版的分析结果，适合在小空间显示

        Args:
            wavelengths: 波长点列表
            lines: 曲线对象列表
            labels: 曲线标签列表

        Returns:
            HTML格式的简化分析结果
        """
        # 创建分析结果HTML
        html_result = ""

        # 获取每条曲线在每个波长点的反射率值
        wavelength_values = {}

        for wl in wavelengths:
            wavelength_values[wl] = {}

            for i, line in enumerate(lines):
                label = labels[i]

                # 获取曲线数据
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                # 找到最接近波长点的值
                idx = np.abs(x_data - wl).argmin()
                closest_wl = x_data[idx]
                value = y_data[idx]

                # 判断找到的点与目标波长的接近程度
                if abs(closest_wl - wl) <= 10:  # 允许10nm以内的误差
                    wavelength_values[wl][label] = (value, closest_wl)

        # 生成波长点分析表格
        html_result += "<style>table {width: 100%; border-collapse: collapse;} "
        html_result += "th, td {border: 1px solid #ddd; padding: 2px; text-align: right; font-size: 9pt;} "
        html_result += "th {background-color: #f2f2f2;} "
        html_result += ".diff-positive {color: green;} .diff-negative {color: red;}</style>"

        # 添加简要说明
        html_result += "<p style='font-size: 9pt;'><b>波长点反射率分析 (点击\"查看详细结果\"获取完整报告)</b></p>"

        # 为每个波长创建简化表格
        for wl in wavelengths:
            if not wavelength_values[wl]:
                continue

            html_result += f"<p style='font-size: 9pt; margin: 2px 0;'><b>λ={wl:.1f}nm</b></p>"
            html_result += "<table>"
            html_result += "<tr><th>曲线</th><th>反射率(%)</th></tr>"

            # 只显示曲线名和反射率值
            for label, (value, _) in wavelength_values[wl].items():
                short_label = label[:15] + "..." if len(label) > 15 else label
                html_result += f"<tr><td>{short_label}</td><td>{value:.2f}</td></tr>"

            html_result += "</table>"

        return html_result

    def show_detailed_analysis(self, analysis_html):
        """显示详细分析结果对话框

        Args:
            analysis_html: 完整的HTML分析结果
        """
        dialog = WavelengthAnalysisDialog(self)
        dialog.set_html_content(analysis_html)
        dialog.exec_()


if __name__ == '__main__':
    app = QApplication([])

    # 设置应用程序级别的字体
    font = QFont("Microsoft YaHei")
    app.setFont(font)

    window = SpectralViewer()
    window.show()

    sys.exit(app.exec())