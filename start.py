import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, 
    QWidget, QLabel, QGraphicsDropShadowEffect, QFrame
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from main import AdvancedCandlestickApp  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ModernButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                text-align: left;
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
        """)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(18, 18, 18))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(dark_palette)
        self.setAutoFillBackground(True)

        self.setWindowTitle("Timeframe Selector")
        self.setGeometry(200, 200, 400, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #ECF0F1;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Select Timeframe")
        title.setStyleSheet("""
            color: #ECF0F1;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            qproperty-alignment: AlignCenter;
        """)
        main_layout.addWidget(title)

        subtitle = QLabel("Choose your preferred trading timeframe")
        subtitle.setStyleSheet("""
            color: #95A5A6;
            font-size: 14px;
            margin-bottom: 30px;
            qproperty-alignment: AlignCenter;
        """)
        main_layout.addWidget(subtitle)

        self.folders = {
            "1 Minute": os.path.join(SCRIPT_DIR, "data/1min"),
            "5 Minute": os.path.join(SCRIPT_DIR, "data/5min"), 
            "10 Minute": os.path.join(SCRIPT_DIR, "data/10min"),
            "15 Minute": os.path.join(SCRIPT_DIR, "data/15min"),
            "30 Minute": os.path.join(SCRIPT_DIR, "data/30min")
        }

        for label, folder in self.folders.items():
            button = ModernButton(label)
            button.clicked.connect(lambda checked, f=folder: self.open_candlestick_app(f))
            main_layout.addWidget(button)

        main_layout.addStretch()

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                border-radius: 10px;
            }
            QWidget {
                background-color: transparent;
            }
        """)

    def open_candlestick_app(self, folder):
        csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

        if not csv_files:
            print(f"No CSV files found in {folder}")
            return

        self.candlestick_window = AdvancedCandlestickApp(csv_files)
        self.candlestick_window.show()

    def closeEvent(self, event):
        try:
            requests.post("http://localhost:3001/trading-app-closed")
        except Exception as e:
            print(f"Error notifying Flask app: {e}")

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())