#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QListWidget, QSplitter, QCheckBox, QComboBox, 
                            QGroupBox, QGridLayout, QMessageBox, QSizePolicy,
                            QMenu, QAction, QTextEdit, QDialog, QTableWidget,
                            QTableWidgetItem, QTabWidget, QHeaderView, QStyle)
from PyQt5.QtCore import Qt, QDir, QTextCodec, QPoint
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
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


class BaseAnalysisDialog(QDialog):
    """分析对话框的基类"""
    
    def __init__(self, parent=None, title="分析结果"):
        super().__init__(parent)
        self.setWindowTitle(title)
        
        # 获取屏幕尺寸并设置窗口为屏幕的80%大小
        screen = QApplication.desktop().screenGeometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry((screen.width() - width) // 2, (screen.height() - height) // 2, width, height)
        
        # 设置窗口标志，允许最大化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
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
        stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
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
            ax1.legend()
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
            
            ax1.legend()
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
        
        # 创建区域分析表格
        region_table = QTableWidget()
        region_table.setColumnCount(7)
        region_table.setHorizontalHeaderLabels([
            "波长区域", "平均反射率", "标准差", 
            "最大值", "最大值波长", "最小值", "最小值波长"
        ])
        region_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
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
                
                # 在图表上添加区域
                ax.axvspan(wl_min, wl_max, alpha=0.2, color=plt.cm.tab10(len(valid_regions) % 10))
                
                # 添加区域标签（位置调整到顶部）
                y_range = ax.get_ylim()
                y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.95
                ax.text((wl_min + wl_max) / 2, y_pos,
                        region_name, ha='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
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
        
        # 创建并添加导航工具栏
        toolbar = NavigationToolbar(canvas, tab)
        
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
        
        # 调用父窗口中的画布来高亮显示区域
        if hasattr(self.parent, 'canvas') and self.parent.canvas is not None:
            try:
                # 使用可视化的颜色
                self.parent.canvas.highlight_curve_segment(
                    self.curve_index, wl_min, wl_max, 
                    color='yellow', alpha=0.5
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
        """添加特征分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 过滤NaN值
        mask = ~np.isnan(self.reflectance)
        valid_wavelength = self.wavelength[mask]
        valid_reflectance = self.reflectance[mask]
        
        if len(valid_reflectance) < 10:
            layout.addWidget(QLabel("数据点不足，无法进行特征分析"))
            self.tab_widget.addTab(tab, "特征分析")
            return
        
        # 创建图表
        fig = Figure(figsize=(9, 9))
        
        # 反射率曲线及峰值检测
        ax1 = fig.add_subplot(211)
        ax1.plot(valid_wavelength, valid_reflectance, '-', linewidth=1.5)
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('反射率 (%)')
        ax1.set_title('特征峰值检测')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 确定窗口大小 - 必须是奇数且小于数据点数
        window_length = min(11, len(valid_reflectance) // 5 * 2 + 1)
        if window_length % 2 == 0 and window_length > 3:
            window_length -= 1
        
        feature_info = ""
        
        if window_length >= 3:
            # 平滑处理
            smoothed = savgol_filter(valid_reflectance, window_length, 2)
            ax1.plot(valid_wavelength, smoothed, '--', linewidth=1, alpha=0.7, label='平滑曲线')
            
            # 峰值检测参数
            prominence = (np.max(smoothed) - np.min(smoothed)) * 0.05  # 5% 的总范围
            width = max(3, len(smoothed) // 30)  # 至少3个点，或者数据点数的1/30
            distance = max(5, len(smoothed) // 20)  # 至少5个点，或者数据点数的1/20
            
            # 查找峰值
            peaks, peak_properties = find_peaks(smoothed, prominence=prominence, width=width, distance=distance)
            # 查找谷值
            troughs, trough_properties = find_peaks(-smoothed, prominence=prominence, width=width, distance=distance)
            
            # 提取峰值和谷值
            peak_x = valid_wavelength[peaks]
            peak_y = smoothed[peaks]
            trough_x = valid_wavelength[troughs]
            trough_y = smoothed[troughs]
            
            # 添加峰值和谷值标记
            ax1.plot(peak_x, peak_y, 'ro', label='峰值')
            ax1.plot(trough_x, trough_y, 'go', label='谷值')
            
            # 标记峰值和谷值
            for i, (x, y) in enumerate(zip(peak_x, peak_y)):
                ax1.annotate(f"P{i+1}", (x, y), xytext=(10, 10), 
                           textcoords='offset points', 
                           arrowprops=dict(arrowstyle='->'))
            
            for i, (x, y) in enumerate(zip(trough_x, trough_y)):
                ax1.annotate(f"T{i+1}", (x, y), xytext=(10, -15), 
                           textcoords='offset points', 
                           arrowprops=dict(arrowstyle='->'))
            
            # 添加特征信息
            if len(peaks) > 0:
                feature_info += "检测到的峰值特征:\n"
                for i, (x, y) in enumerate(zip(peak_x, peak_y)):
                    peak_prom = peak_properties['prominences'][i]
                    peak_width = peak_properties['widths'][i]
                    feature_info += f"  峰值 P{i+1}: 波长={x:.1f}nm, 反射率={y:.2f}%, "
                    feature_info += f"显著度={peak_prom:.2f}, 宽度={peak_width:.1f}\n"
            
            if len(troughs) > 0:
                feature_info += "\n检测到的谷值特征:\n"
                for i, (x, y) in enumerate(zip(trough_x, trough_y)):
                    trough_prom = trough_properties['prominences'][i]
                    trough_width = trough_properties['widths'][i]
                    feature_info += f"  谷值 T{i+1}: 波长={x:.1f}nm, 反射率={y:.2f}%, "
                    feature_info += f"显著度={trough_prom:.2f}, 宽度={trough_width:.1f}\n"
            
            # 添加可能的材料属性特征信息
            feature_info += "\n波段特征解读:\n"
            
            # 检查常见光谱特征
            common_features = [
                # (波长范围, 特征描述)
                ((450, 500), "蓝光范围强反射可能表明对象呈现蓝色外观"),
                ((500, 570), "绿光范围强反射可能表明对象呈现绿色外观"),
                ((620, 750), "红光范围强反射可能表明对象呈现红色外观"),
                ((780, 900), "近红外范围特征可能与植物健康状况或含水量相关"),
                ((900, 950), "此范围的吸收特征常与水分含量相关"),
                ((450, 520), "蓝色吸收带常见于含叶绿素的样本"),
                ((660, 680), "红色吸收带常见于含叶绿素的样本"),
                ((550, 560), "绿色反射峰常见于植物样本")
            ]
            
            # 检查每个特征
            for (wl_min, wl_max), description in common_features:
                # 计算此波长范围的平均反射率
                mask = (valid_wavelength >= wl_min) & (valid_wavelength <= wl_max)
                if np.sum(mask) > 5:  # 确保有足够多的点
                    region_avg = np.mean(valid_reflectance[mask])
                    global_avg = np.mean(valid_reflectance)
                    
                    # 检查是否有峰值在此范围内
                    peaks_in_range = np.sum((peak_x >= wl_min) & (peak_x <= wl_max))
                    troughs_in_range = np.sum((trough_x >= wl_min) & (trough_x <= wl_max))
                    
                    if peaks_in_range > 0 and region_avg > global_avg * 1.1:
                        feature_info += f"  • {wl_min}-{wl_max}nm: 检测到显著反射峰，{description}\n"
                    elif troughs_in_range > 0 and region_avg < global_avg * 0.9:
                        feature_info += f"  • {wl_min}-{wl_max}nm: 检测到显著吸收谷，与{description}相反\n"
            
            # 异常检测 (离群点)
            ax2 = fig.add_subplot(212)
            ax2.plot(valid_wavelength, valid_reflectance, '-', linewidth=1.5)
            ax2.set_xlabel('波长 (nm)')
            ax2.set_ylabel('反射率 (%)')
            ax2.set_title('异常点检测')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 使用移动平均线作为基准
            trend = gaussian_filter1d(valid_reflectance, sigma=len(valid_reflectance)//20)
            ax2.plot(valid_wavelength, trend, '--', linewidth=1, alpha=0.7, label='趋势线')
            
            # 计算残差
            residuals = valid_reflectance - trend
            # 计算残差的标准差
            residual_std = np.std(residuals)
            
            # 找出超过2个标准差的点
            outlier_mask = np.abs(residuals) > 2.5 * residual_std
            outlier_x = valid_wavelength[outlier_mask]
            outlier_y = valid_reflectance[outlier_mask]
            
            if len(outlier_x) > 0:
                ax2.plot(outlier_x, outlier_y, 'rx', markersize=8, label='异常点')
                
                # 添加异常点信息
                feature_info += "\n检测到的异常点:\n"
                for i, (x, y) in enumerate(zip(outlier_x, outlier_y)):
                    feature_info += f"  异常点 {i+1}: 波长={x:.1f}nm, 反射率={y:.2f}%\n"
            
            ax1.legend()
            ax2.legend()
        else:
            feature_info = "数据点不足，无法进行特征分析"
        
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        
        # 创建特征信息文本显示
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFont(QFont("monospace", 9))
        info_text.setText(feature_info)
        
        layout.addWidget(canvas, 2)
        layout.addWidget(info_text, 1)
        
        self.tab_widget.addTab(tab, "特征分析")


class SpectralCanvas(FigureCanvas):
    """用于显示光谱数据的画布类"""
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 存储线条和图例的引用
        self.lines = []
        self.legend_lines = []
        self.highlighted_line = None
        self.original_linewidths = {}
        
        # 连接事件
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # 网格显示状态 - 默认为True
        self._show_grid = True
        # 初始化显示网格
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
    @property
    def show_grid(self):
        """获取网格显示状态"""
        return self._show_grid
        
    @show_grid.setter
    def show_grid(self, value):
        """设置网格显示状态并立即应用"""
        self._show_grid = value
        self.axes.grid(value, linestyle='--', alpha=0.7)
        self.draw()
        
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
        self.draw()
        
    def plot_spectra(self, spectra_dict, title="反射率光谱对比图"):
        """绘制多条光谱曲线
        
        Args:
            spectra_dict: 包含多个光谱数据的字典 {显示标签: (波长数组, 反射率数组)}
            title: 图表标题
        """
        self.axes.clear()
        self.lines = []
        self.legend_lines = []
        self.highlighted_line = None
        self.original_linewidths = {}
        
        # 生成不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(spectra_dict)))
        
        for i, (label, (wavelength, reflectance)) in enumerate(spectra_dict.items()):
            # 移除NaN值
            mask = ~np.isnan(reflectance)
            
            # 绘制曲线，保存线条引用，并设置pickradius使曲线更容易被选中
            line, = self.axes.plot(wavelength[mask], reflectance[mask], 
                                  label=label, color=colors[i], 
                                  linewidth=1.5, picker=5)
            
            # 保存线条引用和原始线宽
            self.lines.append(line)
            self.original_linewidths[line] = line.get_linewidth()
            
        self.axes.set_xlabel('波长 (nm)')
        self.axes.set_ylabel('反射率 (%)')
        self.axes.set_title(title)
        
        # 应用当前的网格状态 - 使用属性而不是直接访问变量
        self.axes.grid(self._show_grid, linestyle='--', alpha=0.7)
        
        # 创建图例，并设置可点击
        if len(spectra_dict) > 10:
            # 对于大量曲线，将图例放在图外右侧
            legend = self.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                     fontsize='small', frameon=True)
            self.fig.subplots_adjust(right=0.75)  # 为图例留出空间
        else:
            legend = self.axes.legend(loc='best', fontsize='medium', frameon=True)
        
        # 设置图例可点击    
        for legline in legend.get_lines():
            legline.set_picker(10)  # 增大点击检测范围
            
        # 保存图例中的线条引用
        self.legend_lines = legend.get_lines()
        
        self.fig.tight_layout()
        self.draw()
    
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
                self.draw()
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
        """更新高亮显示的线条"""
        # 如果当前有高亮线条，恢复原始线宽
        if self.highlighted_line is not None and self.highlighted_line in self.original_linewidths:
            self.highlighted_line.set_linewidth(self.original_linewidths[self.highlighted_line])
            
        # 更新高亮线条引用
        self.highlighted_line = highlight_line
        
        # 如果有新的高亮线条，增加其线宽
        if self.highlighted_line is not None:
            orig_width = self.original_linewidths.get(self.highlighted_line, 1.5)
            self.highlighted_line.set_linewidth(orig_width * 2.5)
            
        # 重绘画布
        self.draw()

    def add_analysis_markers(self, markers_dict):
        """添加分析标记点到图表
        
        Args:
            markers_dict: 包含标记信息的字典 {线条索引: [(x坐标, y坐标, 标记类型, 标记文本), ...]}
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
        self.draw()

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
            
            # 绘制填充区域
            self.axes.axvspan(x_start, x_end, color=color, alpha=alpha, zorder=0)
            
            # 确保曲线在高亮区域的上层
            line.set_zorder(10)
            
            # 添加区域标记
            mid_x = (x_start + x_end) / 2
            self.axes.axvline(x_start, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
            self.axes.axvline(x_end, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
            
            # 添加波长标签
            self.axes.annotate(f"{x_start:.0f}nm", (x_start, y_min + (y_max-y_min)*0.05), 
                              ha='center', va='bottom', fontsize=8, rotation=90, alpha=0.7)
            self.axes.annotate(f"{x_end:.0f}nm", (x_end, y_min + (y_max-y_min)*0.05), 
                              ha='center', va='bottom', fontsize=8, rotation=90, alpha=0.7)
            
            self.draw()


class SpectralViewer(QMainWindow):
    """反射率光谱数据可视化应用程序"""
    
    def __init__(self):
        super().__init__()
        
        self.spectra_data = {}  # 用于存储所有已加载的光谱数据
        self.base_folder = ""   # 基础文件夹路径
        self.grid_visible = True  # 添加网格状态跟踪
        
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('反射率光谱对比分析工具')
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置中文字体
        app_font = QFont("Microsoft YaHei", 9)
        QApplication.setFont(app_font)
        
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
        
        # 浏览按钮
        self.load_btn = QPushButton('加载数据文件夹')
        self.load_btn.clicked.connect(self.load_data_folder)
        load_layout.addWidget(self.load_btn)
        
        # 添加加载单个CSV文件的按钮
        self.load_csv_btn = QPushButton('加载单个CSV文件')
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        load_layout.addWidget(self.load_csv_btn)
        
        # 添加刷新按钮
        self.reload_btn = QPushButton('刷新显示')
        self.reload_btn.clicked.connect(self.update_plot)
        self.reload_btn.setEnabled(False)
        load_layout.addWidget(self.reload_btn)
        
        # 清空按钮
        self.clear_btn = QPushButton('清空所有数据')
        self.clear_btn.clicked.connect(self.clear_all_data)
        self.clear_btn.setEnabled(False)
        load_layout.addWidget(self.clear_btn)
        
        # 添加分析差异按钮
        self.analyze_btn = QPushButton('分析差异')
        self.analyze_btn.clicked.connect(self.analyze_spectra)
        self.analyze_btn.setEnabled(False)
        load_layout.addWidget(self.analyze_btn)
        
        # 数据文件列表（用于显示已加载的文件，支持右键菜单删除）
        self.file_list = QListWidget()
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        self.file_list.setToolTip("右键点击列表项可以删除文件")
        load_layout.addWidget(QLabel("已加载的数据文件 (右键可删除):"))
        load_layout.addWidget(self.file_list)
        
        load_group.setLayout(load_layout)
        control_layout.addWidget(load_group)
        
        # 绘图控制区域
        plot_control_group = QGroupBox("绘图控制")
        plot_control_layout = QVBoxLayout()
        
        # 添加交互提示
        interaction_label = QLabel("交互提示：\n· 点击图例可显示/隐藏对应曲线\n· 鼠标移动到曲线上将高亮显示\n· 右键点击文件名可移除数据")
        plot_control_layout.addWidget(interaction_label)
        
        # 添加其他控制选项
        self.grid_cb = QCheckBox("显示网格")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(self.on_grid_changed)
        plot_control_layout.addWidget(self.grid_cb)
        
        # 添加波长范围设置
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("波长范围:"), 0, 0)
        range_layout.addWidget(QLabel("从"), 1, 0)
        self.wl_min = QComboBox()
        self.wl_min.addItems(["350", "400", "450", "500", "550", "600", "650", "700", "750", "800"])
        range_layout.addWidget(self.wl_min, 1, 1)
        range_layout.addWidget(QLabel("到"), 1, 2)
        self.wl_max = QComboBox()
        self.wl_max.addItems(["1020", "950", "900", "850", "800", "750", "700", "650", "600"])
        range_layout.addWidget(self.wl_max, 1, 3)
        self.wl_min.currentTextChanged.connect(self.update_plot)
        self.wl_max.currentTextChanged.connect(self.update_plot)
        
        plot_control_layout.addLayout(range_layout)
        
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
        self.canvas = SpectralCanvas(plot_panel)
        self.toolbar = NavigationToolbar(self.canvas, plot_panel)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # 创建分割器，允许调整两侧面板的大小
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(plot_panel)
        splitter.setSizes([300, 900])  # 设置初始大小
        
        main_layout.addWidget(splitter)
    
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
            if label in self.spectra_data:
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
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
            
        # 清空数据
        self.spectra_data.clear()
        self.file_list.clear()
        self.canvas.clear_plot()
        
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
                        self.file_list.addItem(display_label)
                        files_loaded += 1
                    except Exception as e:
                        print(f"无法读取文件 {file_path}: {e}")
        
        # 启用按钮
        if files_loaded > 0:
            self.reload_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)  # 启用分析按钮
            
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
                self.canvas.plot_spectra(filtered_data)
        except Exception as e:
            QMessageBox.warning(self, "绘图错误", f"无法绘制图表: {str(e)}")
    
    def on_grid_changed(self, state):
        """处理网格显示状态改变"""
        is_checked = state == Qt.Checked
        
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
                ax.axvspan(effective_wl_min, effective_wl_max, alpha=0.2, color=plt.cm.Pastel1(i % 9))
                
                # 添加区域标签
                ax.text((effective_wl_min + effective_wl_max) / 2, ax.get_ylim()[1] * 0.95,
                       region_name, ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
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
        
        # 设置图例
        if len(spectra_dict) > 10:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
            fig.tight_layout(rect=[0, 0, 0.8, 1])  # 为图例留出空间
        else:
            ax.legend(loc='best', fontsize='large')
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
            self.file_list.addItem(display_label)
            
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用程序级别的字体
    font = QFont("Microsoft YaHei")
    app.setFont(font)
    
    # 中文编码设置
    QTextCodec.setCodecForLocale(QTextCodec.codecForName("UTF-8"))
    
    app.setStyle('Fusion')  # 使用Fusion风格，在不同平台上看起来一致
    
    window = SpectralViewer()
    window.show()
    
    sys.exit(app.exec_()) 