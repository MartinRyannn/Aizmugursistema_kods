import sys
import logging
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtMultimedia import QSoundEffect
import re
import os
import configparser
import subprocess
import tpqoa
import pandas as pd
from datetime import datetime

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
        print(f"Using PyInstaller base path: {base_path}")
    except Exception:
        base_path = os.path.abspath(".")
        print(f"Using development base path: {base_path}")

    if getattr(sys, 'frozen', False):
        base_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Resources')
        print(f"Frozen app base path: {base_path}")
        
        if relative_path.startswith('xautron_frontend'):
            relative_path = os.path.join('xautron_frontend', *relative_path.split('/')[1:])
        elif relative_path.startswith('xautron_backend'):
            if 'start_stream.py' in relative_path:
                return os.path.join(base_path, 'xautron_backend', 'start_stream.py')
            relative_path = os.path.join('xautron_backend', *relative_path.split('/')[1:])
        elif relative_path.startswith('tpqoa'):
            relative_path = os.path.join('tpqoa', *relative_path.split('/')[1:])
        else:
            base_path = os.path.join(base_path, 'launcher')
    else:
        if relative_path.startswith('xautron_frontend'):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        elif relative_path.startswith('xautron_backend'):
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elif relative_path.startswith('tpqoa'):
            try:
                import tpqoa
                base_path = os.path.dirname(tpqoa.__file__)
                print(f"Found tpqoa at: {base_path}")
            except ImportError:
                print("tpqoa not found in Python path")
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(base_path, relative_path)
    
    print(f"Resolving path for {relative_path}:")
    print(f"  Base path: {base_path}")
    print(f"  Relative path: {relative_path}")
    print(f"  Full path: {full_path}")
    print(f"  Path exists: {os.path.exists(full_path)}")
    if os.path.exists(full_path):
        print(f"  File size: {os.path.getsize(full_path)} bytes")
    else:
        print(f"  Directory contents: {os.listdir(os.path.dirname(full_path)) if os.path.exists(os.path.dirname(full_path)) else 'Directory does not exist'}")
    
    return full_path

def get_backend_path():
    logger = logging.getLogger('ModernLauncher')
    
    if getattr(sys, 'frozen', False):
        base_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Resources')
        backend_dir = os.path.join(base_path, 'xautron_backend')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
    
    logger.info(f"Backend directory resolved to: {backend_dir}")
    logger.info(f"Directory exists: {os.path.exists(backend_dir)}")
    if os.path.exists(backend_dir):
        logger.info(f"Directory contents: {os.listdir(backend_dir)}")
        start_stream_path = os.path.join(backend_dir, "start_stream.py")
        logger.info(f"start_stream.py exists: {os.path.exists(start_stream_path)}")
    
    return backend_dir

class LogWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                border: none;
                font-family: 'Courier New';
                font-size: 12px;
            }
        """)

class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setFixedHeight(100)
        self.widget.setVisible(False)
        
    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)


class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 550)
        
        self.container = QWidget(self)
        self.container.setFixedSize(400, 550)
        self.container.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border-radius: 20px;
            }
        """)
        
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: transparent;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.loader = QLabel()
        self.loader.setFixedSize(100, 100)
        self.loader.setStyleSheet("background: transparent;")
        self.content_layout.addWidget(self.loader, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("Validating Configuration...")
        self.status_label.setStyleSheet("""
            color: #C9D1D9;
            font-size: 16px;
            font-weight: bold;
            background: transparent;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.status_label)
        
        self.layout.addWidget(self.content_widget)
        
        self.loading_movie = QMovie(resource_path("loading.gif"))
        self.loading_movie.setScaledSize(QSize(100, 100))
        self.loader.setMovie(self.loading_movie)
        self.loading_movie.start()

    def show_success(self):
        self.loading_movie.stop()
        success_icon = resource_path("checkmark.png")
        self.loader.setPixmap(QPixmap(success_icon).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
        self.status_label.setText("Configuration Valid!")
        self.status_label.setStyleSheet("""
            color: #3FB950;
            font-size: 16px;
            font-weight: bold;
            background: transparent;
        """)
        
    def show_error(self):
        self.loading_movie.stop()
        error_icon = resource_path("error.png")
        self.loader.setPixmap(QPixmap(error_icon).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
        self.status_label.setText("Invalid Configuration")
        self.status_label.setStyleSheet("""
            color: #F85149;
            font-size: 16px;
            font-weight: bold;
            background: transparent;
        """)

class ModernLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend_process = None
        self.frontend_process = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(400, 550)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.apply_rounded_corners()
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

        self.setup_logging()
        self.logger.info("Startins Launcher")
        
        screen = QApplication.primaryScreen().geometry()
        self.move(int((screen.width() - self.width()) / 2),
                 int((screen.height() - self.height()) / 2))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)
        
        self.initial_page = QWidget()
        self.initial_page.setStyleSheet("background-color: #000000;")
        self.initial_layout = QVBoxLayout(self.initial_page)
        self.initial_layout.setContentsMargins(0, 0, 0, 0)
        
        self.logo_label = QLabel(self.initial_page)
        self.logo_label.setFixedSize(300, 400)
        pixmap = QPixmap(resource_path("xlogo.png"))
        scaled_pixmap = pixmap.scaled(300, 400, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.logo_label.setPixmap(scaled_pixmap)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.move(50, 50)
        
        self.logo_opacity = QGraphicsOpacityEffect(self.logo_label)
        self.logo_opacity.setOpacity(0)
        self.logo_label.setGraphicsEffect(self.logo_opacity)
        
        self.form_page = QWidget()
        self.loading_screen = LoadingScreen()
        self.launch_options_page = self.setup_launch_options()
        
        self.setup_form()
        
        self.stack.addWidget(self.initial_page)
        self.stack.addWidget(self.form_page)
        self.stack.addWidget(self.loading_screen)
        self.stack.addWidget(self.launch_options_page)
        
        self.setup_close_button()
        self.setup_animations()
        self.setup_sound()
        
        self.layout.addWidget(self.log_widget)
        
        self.close_btn.setVisible(False)
        QTimer.singleShot(3500, self.start_logo_fade_in)
        self.logger.info("Launcher started successfully")
        
        self.process = None

    def setup_form(self):
        self.form_page.setStyleSheet("background-color: #000000;")
        self.form_layout = QVBoxLayout(self.form_page)
        self.form_layout.setSpacing(25)
        self.form_layout.setContentsMargins(40, 40, 40, 40)

        self.small_logo = QLabel()
        self.small_logo.setFixedSize(100, 100)
        logo_path = resource_path("xlogo.png")
        pixmap = QPixmap(logo_path)
        scaled_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.small_logo.setPixmap(scaled_pixmap)
        self.small_logo.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.form_layout.addWidget(self.small_logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.form_layout.addStretch()

        self.account_id_input = QLineEdit()
        self.access_token_input = QLineEdit()
        self.account_type_combo = QComboBox()

        form_fields = [
            ("Account ID", self.account_id_input),
            ("Access Token", self.access_token_input),
            ("Account Type", self.account_type_combo)
        ]

        for label_text, widget in form_fields:
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)

            label = QLabel(label_text)
            label.setStyleSheet("color: #8B949E; font-size: 14px; font-weight: 500;")

            if label_text == "Access Token":
                widget.setEchoMode(QLineEdit.EchoMode.Password)
                widget.setStyleSheet("""
                    QLineEdit {
                        padding: 15px;
                        background-color: #161B22;
                        border: 1px solid #30363D;
                        border-radius: 8px;
                        color: #C9D1D9;
                        font-size: 14px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #58A6FF;
                        background-color: #1F2937;
                    }
                """)
            elif isinstance(widget, QComboBox):
                widget.addItems(["Live", "Practice"])
                widget.setStyleSheet("""
                    QComboBox {
                        padding: 15px;
                        background-color: #000000;
                        border: 1px solid #30363D;
                        border-radius: 8px;
                        color: #C9D1D9;
                        font-size: 14px;
                    }
                    QComboBox:focus {
                        border: 1px solid #58A6FF;
                        background-color: #000000;
                    }
                    QComboBox::drop-down {
                        border: none;
                        padding-right: 10px;
                    }
                    QComboBox::down-arrow {
                        image: url(arrow.png);
                        width: 12px;
                        height: 12px;
                    }
                    QComboBox QAbstractItemView {
                        background-color: #000000;
                        border: 1px solid #30363D;
                        color: #C9D1D9;
                        selection-background-color: #1F2937;
                    }
                """)
            else:
                widget.setStyleSheet("""
                    QLineEdit {
                        padding: 15px;
                        background-color: #161B22;
                        border: 1px solid #30363D;
                        border-radius: 8px;
                        color: #C9D1D9;
                        font-size: 14px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #58A6FF;
                        background-color: #1F2937;
                    }
                """)

            container_layout.addWidget(label)
            container_layout.addWidget(widget)
            self.form_layout.addWidget(container)

        self.form_layout.addStretch()

        self.launch_btn = QPushButton("Launch Platform")
        self.launch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 40px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #238636, stop:1 #2EA043);
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-size: 16px;
                font-weight: 600;
                min-width: 200px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2EA043, stop:1 #3FB950);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #238636, stop:1 #2EA043);
            }
            QPushButton:disabled {
                background: #21262D;
                color: #8B949E;
            }
        """)
        self.launch_btn.clicked.connect(self.handle_launch)
        self.form_layout.addWidget(self.launch_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        existing_config = self.read_existing_config()
        if existing_config:
            self.account_id_input.setText(existing_config["account_id"])
            self.access_token_input.setText(existing_config["access_token"])
            self.account_type_combo.setCurrentText(existing_config["account_type"])

    def setup_launch_options(self):
        options_page = QWidget()
        options_page.setStyleSheet("background-color: #000000;")
        options_layout = QVBoxLayout(options_page)
        options_layout.setSpacing(25)
        options_layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("Launch Options")
        title.setStyleSheet("color: #C9D1D9; font-size: 24px; font-weight: bold; text-align: center;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        options_layout.addWidget(title)
        
        options_layout.addSpacing(30)
        
        self.launch_algo_btn = QPushButton("Launch Platform")
        self.launch_algo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_algo_btn.setStyleSheet("""
            QPushButton {
                padding: 20px;
                background-color: #238636;
                border: 1px solid #30363D;
                border-radius: 8px;
                color: #FFFFFF;
                font-size: 16px;
                font-weight: 500;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #2EA043;
                border-color: #3FB950;
            }
            QPushButton:pressed {
                background-color: #1B7430;
            }
        """)
        self.launch_algo_btn.clicked.connect(self.launch_with_terminal)
        options_layout.addWidget(self.launch_algo_btn)
        
        self.stop_algo_btn = QPushButton("Stop Platform")
        self.stop_algo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_algo_btn.setStyleSheet("""
            QPushButton {
                padding: 20px;
                background-color: #F85149;
                border: 1px solid #30363D;
                border-radius: 8px;
                color: #FFFFFF;
                font-size: 16px;
                font-weight: 500;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #FF6A65;
                border-color: #FF6A65;
            }
            QPushButton:pressed {
                background-color: #DA4844;
            }
            QPushButton:disabled {
                background-color: #21262D;
                color: #8B949E;
                border-color: #30363D;
            }
        """)
        self.stop_algo_btn.clicked.connect(self.stop_algo)
        self.stop_algo_btn.setEnabled(False)
        options_layout.addWidget(self.stop_algo_btn)
        
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #8B949E; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(self.status_label)
        options_layout.addWidget(status_container)
        
        options_layout.addStretch()
        
        return options_page

    def setup_close_button(self):
        self.close_btn = QPushButton("×", self)
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.move(self.width() - 50, 10)
        self.close_btn.clicked.connect(self.close_application)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #8B949E;
                font-size: 24px;
                border: none;
            }
            QPushButton:hover {
                color: #F85149;
            }
        """)
        
        self.help_btn = QPushButton("?", self)
        self.help_btn.setFixedSize(40, 40)
        self.help_btn.move(10, 10)
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #8B949E;
                font-size: 20px;
                border: none;
            }
            QPushButton:hover {
                color: #58A6FF;
            }
        """)
        self.help_btn.setVisible(False)

    def kill_existing_process(self):
        current_pid = os.getpid()
        self.logger.info(f"Current process ID: {current_pid}")
        
        if sys.platform == 'darwin':  # macOS
            try:
                self.logger.info("Checking for existing processes")
                cmd = f"pgrep -f 'python.*start_stream.py' | grep -v {current_pid}"
                process_ids = subprocess.check_output(cmd, shell=True, text=True).strip()
                
                if process_ids:
                    for pid in process_ids.split('\n'):
                        if pid.strip():
                            subprocess.run(["kill", "-9", pid.strip()], stderr=subprocess.DEVNULL)
                    self.logger.info("Killed existing Python stream processes")
            except subprocess.SubprocessError:
                self.logger.info("No existing processes to kill")
        else:  # Windows
            try:
                cmd = f'wmic process where "commandline like \'%start_stream.py%\' and not processid={current_pid}" get processid'
                output = subprocess.check_output(cmd, shell=True, text=True)
                pids = [line.strip() for line in output.split('\n') if line.strip().isdigit()]
                
                for pid in pids:
                    subprocess.run(["taskkill", "/F", "/PID", pid], stderr=subprocess.DEVNULL)
                
                self.logger.info("Killed existing Python stream processes")
            except subprocess.SubprocessError:
                self.logger.info("No existing processes to kill")

    def launch_with_terminal(self):
        self.logger.info("Starting launch process...")
        self.kill_existing_process()
        
        backend_path = get_backend_path()
        self.logger.info(f"Backend path: {backend_path}")
        
        stream_script = os.path.join(backend_path, "start_stream.py")
        self.logger.info(f"Looking for start_stream.py at: {stream_script}")
        self.logger.info(f"File exists: {os.path.exists(stream_script)}")
        
        if not os.path.exists(stream_script):
            error_msg = f"start_stream.py not found at: {stream_script}"
            self.logger.error(error_msg)
            self.logger.error("Current working directory: " + os.getcwd())
            if os.path.exists(backend_path):
                self.logger.error("Backend directory contents: " + str(os.listdir(backend_path)))
            self.status_label.setText("Error: start_stream.py not found!")
            self.status_label.setStyleSheet("color: #F85149; font-size: 14px;")
            return
        
        frontend_path = resource_path("xautron_frontend")
        if not os.path.exists(frontend_path):
            self.logger.error(f"Frontend directory not found at {frontend_path}")
            return False
        
        self.logger.info(f"Frontend path: {frontend_path}")
        
        try:
            if sys.platform == 'darwin':  # macOS
                self.logger.info("Running on macOS, creating shell scripts...")
                
                temp_backend_script = os.path.join(backend_path, "_temp_run_stream.command")
                self.logger.info(f"Creating backend script at: {temp_backend_script}")
                
                with open(temp_backend_script, "w") as f:
                    f.write(f"""#!/bin/bash
cd "{backend_path}"
echo "Starting Xautron stream processor..."
echo "Working directory: $(pwd)"
echo "Using Python path: $(which python3)"
echo "Python version: $(python3 --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Contents of current directory:"
ls -la
echo "Contents of xautron_backend directory:"
ls -la xautron_backend
echo "Contents of tpqoa directory:"
ls -la tpqoa
python3 "start_stream.py"
echo "Process completed or terminated."
echo "Press Enter to close this terminal..."
read
""")
                os.chmod(temp_backend_script, 0o755)
                self.logger.info("Backend script created and permissions set")
                
                temp_frontend_script = os.path.join(backend_path, "_temp_run_frontend.command")
                self.logger.info(f"Creating frontend script at: {temp_frontend_script}")
                
                with open(temp_frontend_script, "w") as f:
                    f.write(f"""#!/bin/bash
cd "{frontend_path}/design"
echo "Starting Xautron frontend..."
echo "Working directory: $(pwd)"
echo "Installing dependencies..."
rm -rf node_modules
npm install
echo "Installing react-scripts..."
npm install react-scripts@5.0.1 --save-dev
echo "Starting frontend server..."
export NODE_OPTIONS=--openssl-legacy-provider
npm start
echo "Frontend server stopped."
echo "Press Enter to close this terminal..."
read
""")
                os.chmod(temp_frontend_script, 0o755)
                self.logger.info("Frontend script created and permissions set")
                
                self.logger.info("Starting backend process...")
                self.backend_process = subprocess.Popen(["open", temp_backend_script])
                self.logger.info("Starting frontend process...")
                self.frontend_process = subprocess.Popen(["open", temp_frontend_script])
                
            
            self.status_label.setText("Status: Algorithm & Frontend Running")
            self.status_label.setStyleSheet("color: #3FB950; font-size: 14px;")
            self.logger.info("Successfully started both backend and frontend processes")
            
            self.launch_algo_btn.setEnabled(False)
            self.stop_algo_btn.setEnabled(True)
        except Exception as e:
            error_msg = f"Failed to start algorithm or frontend: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error("Exception details:", exc_info=True)
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #F85149; font-size: 14px;")

    def stop_algo(self):
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(["lsof", "-ti", ":3001", "|", "xargs", "kill", "-9"], shell=True, stderr=subprocess.DEVNULL)
                subprocess.run(["lsof", "-ti", ":3000", "|", "xargs", "kill", "-9"], shell=True, stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.run(["powershell", "-Command", "Get-NetTCPConnection -LocalPort 3001 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"], stderr=subprocess.DEVNULL)
                subprocess.run(["powershell", "-Command", "Get-NetTCPConnection -LocalPort 3000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"], stderr=subprocess.DEVNULL)
            
            if sys.platform == 'darwin':  # macOS
                subprocess.run(["pkill", "-f", "start_stream.py"], stderr=subprocess.DEVNULL)
                subprocess.run(["pkill", "-f", "npm start"], stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq *start_stream.py*"], stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/FI", "IMAGENAME eq cmd.exe", "/FI", "WINDOWTITLE eq *start_stream.py*"], stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/FI", "IMAGENAME eq node.exe"], stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/FI", "IMAGENAME eq npm.cmd"], stderr=subprocess.DEVNULL)
            
            if hasattr(self, 'backend_process') and self.backend_process and self.backend_process.poll() is None:
                self.backend_process.terminate()
                try:
                    self.backend_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.backend_process.kill()
            
            if hasattr(self, 'frontend_process') and self.frontend_process and self.frontend_process.poll() is None:
                self.frontend_process.terminate()
                try:
                    self.frontend_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.frontend_process.kill()
            
            self.status_label.setText("Status: Algorithm & Frontend Stopped")
            self.status_label.setStyleSheet("color: #8B949E; font-size: 14px;")
            self.logger.info("Algorithm and frontend stopped successfully")
            
            self.launch_algo_btn.setEnabled(True)
            self.stop_algo_btn.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f"Error stopping: {str(e)}")
            self.status_label.setStyleSheet("color: #F85149; font-size: 14px;")
            self.logger.error(f"Failed to stop algorithm or frontend: {str(e)}")

    def close_application(self):
        if hasattr(self, 'backend_process') and self.backend_process and self.backend_process.poll() is None:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if hasattr(self, 'frontend_process') and self.frontend_process and self.frontend_process.poll() is None:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(["pkill", "-f", "npm"], stderr=subprocess.DEVNULL)
                subprocess.run(["pkill", "-f", "node"], stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.run(["taskkill", "/F", "/IM", "node.exe"], stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/IM", "npm.cmd"], stderr=subprocess.DEVNULL)
        except:
            pass
            
        self.close()


    def show_form_elements(self):
        self.logger.info("Showing form elements")
        self.close_btn.setVisible(True)
        self.help_btn.setVisible(True)
        self.stack.setCurrentIndex(1)

    def apply_rounded_corners(self):
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 20, 20)
        mask = QRegion(path.toFillPolygon(QTransform()).toPolygon())
        self.setMask(mask)

    def setup_logging(self):
        self.log_widget = LogWidget(self)
        self.log_handler = QTextEditLogger(self.log_widget)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger = logging.getLogger('ModernLauncher')
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)

    def setup_animations(self):
        self.fade_in_anim = QPropertyAnimation(self.logo_opacity, b"opacity")
        self.fade_in_anim.setDuration(1500)
        self.fade_in_anim.setStartValue(0)
        self.fade_in_anim.setEndValue(1)
        self.fade_in_anim.finished.connect(self.start_logo_fade_out)
        
        self.fade_out_anim = QPropertyAnimation(self.logo_opacity, b"opacity")
        self.fade_out_anim.setDuration(1500)
        self.fade_out_anim.setStartValue(1)
        self.fade_out_anim.setEndValue(0)
        self.fade_out_anim.finished.connect(self.show_form_elements)

    def setup_sound(self):
        self.sound = QSoundEffect()
        sound_path = resource_path("welcome.wav")
        self.sound.setSource(QUrl.fromLocalFile(sound_path))
        self.sound.setLoopCount(1)
        self.sound.setVolume(0.5)
        QTimer.singleShot(100, self.sound.play)

    def validate_api_connection(self):
        try:
            config_path = os.path.join(get_backend_path(), "oanda.cfg")
            api = tpqoa.tpqoa(config_path)
            data = api.get_history(
                instrument="XAU_USD",
                start="2024-12-12",
                end="2024-12-13",
                granularity="D",
                price='M',
                localize=False
            )
            return True
        except Exception as e:
            self.logger.error(f"API validation error: {str(e)}")
            return False

    def handle_launch(self):
        account_id = self.account_id_input.text().strip()
        access_token = self.access_token_input.text().strip()
        account_type = self.account_type_combo.currentText()

        if not account_id or not access_token:
            QMessageBox.warning(self, "Error", "All fields are required.")
            return

        if not self.is_valid_account_id(account_id):
            QMessageBox.warning(self, "Error", "Invalid Account ID format.")
            return

        if not self.is_valid_access_token(access_token):
            QMessageBox.warning(self, "Error", "Invalid Access Token format.")
            return

        self.write_config(account_id, access_token, account_type)
        self.stack.setCurrentWidget(self.loading_screen)
        
        QTimer.singleShot(1500, self.validate_configuration)

    def validate_configuration(self):
        if self.validate_api_connection():
            self.loading_screen.show_success()
            QTimer.singleShot(1500, self.show_launch_options)
        else:
            self.loading_screen.show_error()
            QTimer.singleShot(2000, lambda: self.stack.setCurrentWidget(self.form_page))

    def show_launch_options(self):
        self.stack.setCurrentWidget(self.launch_options_page)

    def start_logo_fade_in(self):
        self.logger.info("Starting logo fade in animation")
        self.fade_in_anim.start()

    def start_logo_fade_out(self):
        self.logger.info("Starting logo fade out animation")
        QTimer.singleShot(2000, self.fade_out_anim.start)

    def is_valid_account_id(self, account_id):
        pattern = r"^\d{3}-\d{3}-\d{8}-\d{3}$"
        return bool(re.fullmatch(pattern, account_id))

    def is_valid_access_token(self, access_token):
        pattern = r"^[a-f0-9]{32}-[a-f0-9]{32}$"
        return bool(re.fullmatch(pattern, access_token))

    def read_existing_config(self):
        config_path = os.path.join(get_backend_path(), "oanda.cfg")
        
        if not os.path.exists(config_path):
            return None
        
        config = configparser.ConfigParser()
        config.read(config_path)

        if "oanda" in config:
            return {
                "account_id": config["oanda"].get("account_id", ""),
                "access_token": config["oanda"].get("access_token", ""),
                "account_type": config["oanda"].get("account_type", ""),
            }
        return None
    
    def write_config(self, account_id, access_token, account_type):
        config_path = os.path.join(get_backend_path(), "oanda.cfg")
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        config = configparser.ConfigParser()
        config["oanda"] = {
            "account_id": account_id,
            "access_token": access_token,
            "account_type": account_type,
        }

        with open(config_path, "w") as configfile:
            config.write(configfile)
        
        self.logger.info(f"Configuration saved to {config_path}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(self.pos() + event.globalPosition().toPoint() - self.dragPos)
            self.dragPos = event.globalPosition().toPoint()

    def closeEvent(self, event):
        if hasattr(self, 'backend_process') and self.backend_process and self.backend_process.poll() is None:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if hasattr(self, 'frontend_process') and self.frontend_process and self.frontend_process.poll() is None:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    
    window = ModernLauncher()
    window.show()
    sys.exit(app.exec())