import os
import sys
import pandas as pd
import tpqoa
from datetime import datetime, timedelta
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
    QComboBox, QLabel, QDateEdit, QFormLayout, QMessageBox, QToolBar,
    QSplitter, QFrame, QSizePolicy
)
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QPalette, QColor, QFont
import mplfinance as mpf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Modern dark theme colors
DARK_BG = '#121212'
DARK_CARD_BG = '#1e1e1e'
DARK_FG = '#ffffff'
DARK_ACCENT = '#2979ff' 
DARK_GRID = '#333333'
DARK_TEXT = '#e0e0e0'
DARK_SECONDARY = '#757575'
UP_COLOR = '#00c853' 
DOWN_COLOR = '#ff1744'

class ModernToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.setStyleSheet("""
            QToolBar { 
                background-color: #1e1e1e; 
                border: none;
                spacing: 10px;
                padding: 5px;
            }
            QToolButton { 
                background-color: #2d2d2d;
                color: #e0e0e0;
                border-radius: 4px;
                padding: 4px;
                margin: 2px;
            }
            QToolButton:hover { 
                background-color: #3d3d3d;
            }
        """)

class HistoricalChartApp(QMainWindow):
    def __init__(self):
        super().__init__()
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oanda.cfg')
        self.api = tpqoa.tpqoa(config_path)
        
        self.setWindowTitle("Financial Chart Explorer")
        self.setMinimumSize(1600, 800) 
        
        self.apply_dark_theme()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        splitter = QSplitter(Qt.Horizontal)
        
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_CARD_BG};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        controls_card_layout = QVBoxLayout(controls_frame)
        controls_card_layout.setSpacing(15)
        
        title_label = QLabel("Chart Settings")
        title_label.setStyleSheet(f"color: {DARK_FG}; font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        controls_card_layout.addWidget(title_label)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        form_layout.setFormAlignment(Qt.AlignLeft)
        
        label_style = f"color: {DARK_TEXT}; font-size: 14px; font-weight: bold;"
        
        instrument_label = QLabel("Instrument:")
        instrument_label.setStyleSheet(label_style)
        self.instrument_dropdown = self.create_styled_combobox([
            "XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", 
            "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP", "EUR_JPY"
        ])
        self.instrument_dropdown.setFixedWidth(250)  # Set fixed width
        form_layout.addRow(instrument_label, self.instrument_dropdown)
        
        period_label = QLabel("Timeframe:")
        period_label.setStyleSheet(label_style)
        self.period_dropdown = self.create_styled_combobox(["1", "5", "10", "15", "30", "1h", "4h"])
        self.period_dropdown.setFixedWidth(250)  # Set fixed width
        form_layout.addRow(period_label, self.period_dropdown)

        chart_type_label = QLabel("Chart Type:")
        chart_type_label.setStyleSheet(label_style)
        self.chart_type_dropdown = self.create_styled_combobox(["Candlestick", "Line"])
        self.chart_type_dropdown.setFixedWidth(250)  # Set fixed width
        form_layout.addRow(chart_type_label, self.chart_type_dropdown)

        time_range_label = QLabel("Time Range:")
        time_range_label.setStyleSheet(label_style)
        self.days_back_dropdown = self.create_styled_combobox(
            ["1 day", "1 week", "1 month", "3 months", "6 months", "1 year"]
        )
        self.days_back_dropdown.setFixedWidth(250)  # Set fixed width
        self.days_back_dropdown.currentTextChanged.connect(self.update_date_range)
        form_layout.addRow(time_range_label, self.days_back_dropdown)

        start_date_label = QLabel("Start Date:")
        start_date_label.setStyleSheet(label_style)
        self.start_date_picker = QDateEdit(calendarPopup=True)
        self.start_date_picker.setReadOnly(True)
        self.start_date_picker.setStyleSheet(self.get_date_picker_style())
        self.start_date_picker.setFixedWidth(250)  # Set fixed width
        form_layout.addRow(start_date_label, self.start_date_picker)
        
        end_date_label = QLabel("End Date:")
        end_date_label.setStyleSheet(label_style)
        self.end_date_picker = QDateEdit(calendarPopup=True)
        self.end_date_picker.setReadOnly(True)
        self.end_date_picker.setStyleSheet(self.get_date_picker_style())
        self.end_date_picker.setFixedWidth(250)  # Set fixed width
        form_layout.addRow(end_date_label, self.end_date_picker)
        
        controls_card_layout.addLayout(form_layout)
        
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        
        self.submit_button = self.create_styled_button("Load Data", DARK_ACCENT)
        self.submit_button.setFixedWidth(250)  # Set fixed width
        self.submit_button.clicked.connect(self.load_and_plot_data)
        button_layout.addWidget(self.submit_button)

        
        controls_card_layout.addLayout(button_layout)
        
        controls_card_layout.addStretch()
        
        controls_layout.addWidget(controls_frame)
        
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.StyledPanel)
        chart_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DARK_CARD_BG};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        chart_card_layout = QVBoxLayout(chart_frame)
        
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.toolbar = ModernToolbar(self.canvas, self)
        
        chart_card_layout.addWidget(self.toolbar)
        chart_card_layout.addWidget(self.canvas)
        
        chart_layout.addWidget(chart_frame)
        
        splitter.addWidget(controls_widget)
        splitter.addWidget(chart_widget)
        
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        plt.style.use('dark_background')
        self.figure.patch.set_facecolor(DARK_BG)
        
        self.update_date_range()
        
        self.init_empty_chart()

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(DARK_BG))
        dark_palette.setColor(QPalette.WindowText, QColor(DARK_FG))
        dark_palette.setColor(QPalette.Base, QColor(DARK_CARD_BG))
        dark_palette.setColor(QPalette.AlternateBase, QColor(DARK_CARD_BG))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(DARK_FG))
        dark_palette.setColor(QPalette.ToolTipText, QColor(DARK_TEXT))
        dark_palette.setColor(QPalette.Text, QColor(DARK_FG))
        dark_palette.setColor(QPalette.Button, QColor(DARK_CARD_BG))
        dark_palette.setColor(QPalette.ButtonText, QColor(DARK_FG))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(DARK_ACCENT))
        dark_palette.setColor(QPalette.Highlight, QColor(DARK_ACCENT))
        dark_palette.setColor(QPalette.HighlightedText, QColor(DARK_BG))
        
        self.setPalette(dark_palette)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {DARK_BG};
                color: {DARK_FG};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QSplitter::handle {{
                background-color: #333333;
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
        """)

    def create_styled_combobox(self, items):
        combo = QComboBox()
        combo.addItems(items)
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: #2d2d2d;
                color: {DARK_TEXT};
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                min-height: 30px;
                font-size: 13px;
            }}
            QComboBox:hover {{
                border: 1px solid {DARK_ACCENT};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3d3d3d;
            }}
            QComboBox QAbstractItemView {{
                background-color: #2d2d2d;
                color: {DARK_TEXT};
                selection-background-color: {DARK_ACCENT};
                selection-color: {DARK_BG};
                border: 1px solid #3d3d3d;
            }}
        """)
        return combo

    def get_date_picker_style(self):
        return f"""
            QDateEdit {{
                background-color: #2d2d2d;
                color: {DARK_TEXT};
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                min-height: 30px;
                font-size: 13px;
            }}
            QDateEdit:hover {{
                border: 1px solid {DARK_ACCENT};
            }}
            QDateEdit::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3d3d3d;
            }}
            QCalendarWidget {{
                background-color: #2d2d2d;
                color: {DARK_TEXT};
            }}
            QCalendarWidget QAbstractItemView:enabled {{
                background-color: #2d2d2d;
                color: {DARK_TEXT};
                selection-background-color: {DARK_ACCENT};
                selection-color: {DARK_BG};
            }}
        """

    def create_styled_button(self, text, bg_color):
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {DARK_FG};
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(bg_color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(bg_color)};
            }}
        """)
        return button

    def lighten_color(self, hex_color, amount=20):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r = min(255, r + amount)
        g = min(255, g + amount)
        b = min(255, b + amount)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def darken_color(self, hex_color, amount=20):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r = max(0, r - amount)
        g = max(0, g - amount)
        b = max(0, b - amount)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def init_empty_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(DARK_CARD_BG)
        
        ax.text(0.5, 0.5, 'Select parameters and click "Load Data" to view chart', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color=DARK_TEXT, transform=ax.transAxes)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        self.canvas.draw()

    def update_date_range(self):
        today = QDate.currentDate()
        time_range = self.days_back_dropdown.currentText()
        
        if time_range == "1 day":
            days_back = 1
        elif time_range == "1 week":
            days_back = 7
        elif time_range == "1 month":
            days_back = 30
        elif time_range == "3 months":
            days_back = 90
        elif time_range == "6 months":
            days_back = 180
        else:  # 1 year
            days_back = 365

        end_date = today.addDays(-1)
        start_date = end_date.addDays(-days_back)
        
        self.end_date_picker.setDate(end_date)
        self.start_date_picker.setDate(start_date)

    def load_and_plot_data(self):
        instrument = self.instrument_dropdown.currentText()
        start_date = self.start_date_picker.date().toString("yyyy-MM-dd")
        end_date = self.end_date_picker.date().toString("yyyy-MM-dd")
        period = self.period_dropdown.currentText()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(DARK_CARD_BG)
        ax.text(0.5, 0.5, 'Loading data...', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color=DARK_TEXT, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

        try:
            if period.endswith('h'):
                granularity = f'H{period[:-1]}'
            else:
                granularity = f'M{period}'

            data = self.api.get_history(
                instrument=instrument,
                start=start_date + "T00:00:00",
                end=end_date + "T23:59:59",
                granularity=granularity,
                price='M',
                localize=False
            )

            if data.empty:
                self.show_error_message("No data available for the selected date range.")
                self.init_empty_chart()
            else:
                data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}, inplace=True)
                
                if isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                else:
                    data.index = pd.to_datetime(data.index)
                
                self.plot_data(data)

        except Exception as e:
            self.show_error_message(f"Error downloading data: {e}")
            self.init_empty_chart()

    def plot_data(self, data):
        self.figure.clear()
        
        mc = mpf.make_marketcolors(
            up=UP_COLOR,
            down=DOWN_COLOR,
            edge='inherit',
            wick='inherit',
            volume='in',
            ohlc='inherit'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor=DARK_CARD_BG,
            edgecolor=DARK_TEXT,
            figcolor=DARK_CARD_BG,
            gridcolor=DARK_GRID,
            gridstyle='--',
            y_on_right=False,
            rc={
                'font.size': 12,
                'font.family': 'Segoe UI',
                'axes.labelcolor': DARK_TEXT,
                'axes.titlecolor': DARK_TEXT,
                'xtick.color': DARK_TEXT,
                'ytick.color': DARK_TEXT
            }
        )
        
        ax = self.figure.add_subplot(111)
        
        instrument = self.instrument_dropdown.currentText()
        period = self.period_dropdown.currentText()
        title = f"{instrument} - {period} Chart"
        self.figure.suptitle(title, color=DARK_FG, fontsize=14, fontweight='bold', y=0.98)  # Reduced font size
        
        start_date = self.start_date_picker.date().toString("yyyy-MM-dd")
        end_date = self.end_date_picker.date().toString("yyyy-MM-dd")
        subtitle = f"Period: {start_date} to {end_date}"
        self.figure.text(0.5, 0.94, subtitle, color=DARK_SECONDARY, fontsize=10,  # Reduced font size
                        ha='center', va='center')
        
        chart_type = self.chart_type_dropdown.currentText().lower()
        
        mpf.plot(
            data,
            type=chart_type,
            style=s,
            ax=ax,
            volume=False,
            ylabel='Price',
            warn_too_much_data=len(data) + 1
        )
        
        ax.grid(True, color=DARK_GRID, linestyle='--', alpha=0.3)
        
        ax.set_ylabel('Price', fontsize=12, fontweight='bold', color=DARK_TEXT)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', color=DARK_TEXT)
        
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        self.canvas.draw()

    def save_chart(self):
        try:
            filename = f"chart_{self.instrument_dropdown.currentText()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.figure.savefig(filename, dpi=300, facecolor=DARK_CARD_BG)
            QMessageBox.information(self, "Success", f"Chart saved as {filename}")
        except Exception as e:
            self.show_error_message(f"Error saving chart: {e}")

    def show_error_message(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {DARK_CARD_BG};
                color: {DARK_FG};
            }}
            QLabel {{
                color: {DARK_FG};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {DARK_ACCENT};
                color: {DARK_FG};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(DARK_ACCENT)};
            }}
        """)
        
        msg.exec_()

    def closeEvent(self, event):
        try:
            response = requests.post("http://localhost:3001/history-app-closed")
            if response.status_code == 200:
                print("Successfully notified Flask app that the trading app is closing.")
            else:
                print(f"Failed to notify Flask app. Status code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Error notifying Flask app: {e}")

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = HistoricalChartApp()
    window.show()
    sys.exit(app.exec_())