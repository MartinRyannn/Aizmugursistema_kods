import os
import sys
import pandas as pd
import mplfinance as mpf
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QHBoxLayout, QLabel, QSlider, QStyleFactory, QSpacerItem, 
    QSizePolicy, QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon, QPainter, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

def CustomMplStyle():
    return mpf.make_mpf_style(
        base_mpl_style="dark_background",
        marketcolors={
            'candle': {'up': '#1C9963', 'down': '#E94F64'},
            'edge': {'up': '#1C9963', 'down': '#E94F64'},
            'wick': {'up': '#1C9963', 'down': '#E94F64'}
        },
        rc={
            'figure.facecolor': '#121212',
            'axes.facecolor': '#121212',
            'grid.color': '#1C1C1C',
            'axes.labelcolor': '#ECF0F1',
            'text.color': '#ECF0F1',
            'xtick.color': '#95A5A6',
            'ytick.color': '#95A5A6'
        }
    )

class ModernButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #34495E;
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background-color: #2980B9;
                transform: translateY(0);
            }
            QPushButton:disabled {
                background-color: #2C3E50;
                color: #7F8C8D;
            }
        """)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

class ModernSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal):
        super().__init__(orientation)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2C3E50;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980B9;
            }
            QSlider::sub-page:horizontal {
                background: #3498DB;
                border-radius: 4px;
            }
        """)

class AdvancedCandlestickApp(QMainWindow):
    def __init__(self, data_files):
        super().__init__()
        
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(18, 18, 18))
        dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Button, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        
        QApplication.setPalette(dark_palette)

        self.data_files = data_files
        self.current_file_index = 0
        self.data = self.load_data(self.data_files[self.current_file_index])
        self.current_index = 0
        self.running = False
        self.x_limit = 50
        self.playback_speed = 1000

        self.analysis_markers = {
            'support_levels': [],
            'resistance_levels': [],
            'trend_lines': []
        }
        self.drawing_mode = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Advanced Candlestick Analyzer")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 13px;
            }
            QFrame {
                background-color: #1E1E1E;
                border-radius: 8px;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Top controls container
        top_controls = QFrame()
        top_controls.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        top_controls_layout = QHBoxLayout()
        top_controls_layout.setSpacing(15)
        top_controls_layout.setContentsMargins(15, 15, 15, 15)

        # Control buttons
        control_buttons = [
            ('▶ Play', self.start_playback),
            ('⏹ Stop', self.stop_playback),
            ('◀ Previous', self.load_previous_file),
            ('▶ Next', self.load_next_file),
            ('⬇ Draw Support', lambda: self.set_drawing_mode('support')),
            ('⬆ Draw Resistance', lambda: self.set_drawing_mode('resistance')),
            ('➖ Draw Trend', lambda: self.set_drawing_mode('trend'))
        ]

        for text, action in control_buttons:
            btn = ModernButton(text)
            btn.clicked.connect(action)
            top_controls_layout.addWidget(btn)

        # Speed control
        speed_container = QFrame()
        speed_container.setStyleSheet("""
            QFrame {
                background-color: #2C3E50;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        speed_layout = QVBoxLayout()
        speed_layout.setSpacing(5)
        
        speed_label = QLabel("Playback Speed")
        speed_label.setStyleSheet("color: #ECF0F1; font-weight: 600;")
        speed_label.setAlignment(Qt.AlignCenter)

        self.speed_slider = ModernSlider(Qt.Horizontal)
        self.speed_slider.setRange(100, 2000)
        self.speed_slider.setValue(1000)
        self.speed_slider.setFixedWidth(200)
        self.speed_slider.valueChanged.connect(self.set_speed)

        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_container.setLayout(speed_layout)
        top_controls_layout.addWidget(speed_container)

        top_controls.setLayout(top_controls_layout)
        main_layout.addWidget(top_controls)

        # Chart container
        chart_container = QFrame()
        chart_container.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='#121212')
        
        mpf_style = CustomMplStyle()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #121212; border-radius: 8px;")
        chart_layout.addWidget(self.canvas)
        chart_container.setLayout(chart_layout)
        main_layout.addWidget(chart_container)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.plot_candlestick()
        self.canvas.mpl_connect('button_press_event', self.on_chart_click)

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'volume': 'Volume'}, inplace=True)
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data

    def plot_candlestick(self):
        try:
            if len(self.data) == 0:
                return

            self.ax.clear()
            
            plot_data = self.data.iloc[:self.current_index + 1]
            
            mpf.plot(plot_data,
                    type='candle',
                    style='nightclouds',
                    ax=self.ax,
                    volume=False)

            self.plot_analysis_markers()

            self.ax.set_title('Candlestick Analyzer', color='#D0D0D0', fontsize=12, fontweight='bold')
            self.ax.grid(color='#1C1C1C', linestyle=':', alpha=0.3)
            
            self.update_view_limits()
            self.canvas.draw()
        except Exception as e:
            print(f"Error in plot_candlestick: {e}")

    def plot_analysis_markers(self):
        for level in self.analysis_markers['support_levels']:
            self.ax.axhline(y=level, color='#00FF00', linestyle='--', alpha=0.7)

        for level in self.analysis_markers['resistance_levels']:
            self.ax.axhline(y=level, color='#FF0000', linestyle='--', alpha=0.7)

        for line in self.analysis_markers['trend_lines']:
            self.ax.plot(line[0], line[1], color='#2A2A2A', linestyle='--', alpha=0.5)

    def set_drawing_mode(self, mode):
        self.drawing_mode = mode

    def on_chart_click(self, event):
        if event.inaxes == self.ax and self.drawing_mode:
            y_value = event.ydata
            
            if self.drawing_mode == 'support':
                self.analysis_markers['support_levels'].append(y_value)
            
            elif self.drawing_mode == 'resistance':
                self.analysis_markers['resistance_levels'].append(y_value)
            
            elif self.drawing_mode == 'trend':
                self.analysis_markers['trend_lines'].append(([event.xdata], [y_value]))
            
            self.plot_candlestick()

    def start_playback(self):
        if not self.running:
            self.running = True
            self.playback()

    def stop_playback(self):
        self.running = False

    def playback(self):
        if self.running and self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.plot_candlestick()
            QTimer.singleShot(self.playback_speed, self.playback)

    def set_speed(self, speed):
        self.playback_speed = speed

    def load_previous_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_data_and_plot()

    def load_next_file(self):
        if self.current_file_index < len(self.data_files) - 1:
            self.current_file_index += 1
            self.load_data_and_plot()

    def load_data_and_plot(self):
        self.stop_playback()
        self.data = self.load_data(self.data_files[self.current_file_index])
        self.current_index = 0
        self.analysis_markers = {
            'support_levels': [],
            'resistance_levels': [],
            'trend_lines': []
        }
        self.plot_candlestick()

    def update_view_limits(self):
        try:
            if len(self.data) == 0:
                return

            current_data = self.data.iloc[:self.current_index + 1]
            
            valid_low = current_data['Low'].dropna()
            valid_high = current_data['High'].dropna()
            
            if len(valid_low) == 0 or len(valid_high) == 0:
                return

            y_min = valid_low.min()
            y_max = valid_high.max()

            y_min -= 5
            y_max += 5

            if self.current_index >= self.x_limit:
                self.x_limit += 30
            
            self.ax.set_xlim(0, self.x_limit)
            self.ax.set_ylim(y_min, y_max)
        except Exception as e:
            print(f"Error in update_view_limits: {e}")

def main():
    data_folder = 'data'
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    app = QApplication(sys.argv)
    window = AdvancedCandlestickApp(data_files)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()