import sys
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QListWidget, QMessageBox,
    QTabWidget, QProgressBar, QGridLayout, QFrame, QSpinBox,
    QScrollArea, QSizePolicy, QSystemTrayIcon
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor
from ultralytics import YOLO
import numpy as np

# Цветовая палитра
COLORS = {
    "brown_beige": "#A27B5C",
    "soft_cream": "#DCD0C0",
    "light_bg": "#F1F1F1",
    "pine_green": "#3F4B3B",
    "sage": "#7B9E89"
}

class StyleableButton(QPushButton):
    def __init__(self, text="", parent=None, color=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color if color else COLORS['sage']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['pine_green']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['brown_beige']};
            }}
        """)

class ClassCard(QFrame):
    def __init__(self, class_name, on_delete, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.on_delete = on_delete
        
        self.setStyleSheet(f"""
            ClassCard {{
                background-color: {COLORS['soft_cream']};
                border-radius: 6px;
                margin: 4px;
                padding: 10px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Иконка класса (заглушка)
        icon_label = QLabel()
        icon_label.setFixedSize(32, 32)
        icon_label.setStyleSheet(f"background-color: {COLORS['sage']}; border-radius: 16px;")
        
        # Название класса
        name_label = QLabel(class_name)
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #3F4B3B;")
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Если текст длинный, включаем перенос строк
        if len(class_name) > 30:
            name_label.setWordWrap(True)
        
        # Кнопка удаления
        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(24, 24)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['brown_beige']};
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #8B6B4F;
            }}
        """)
        delete_btn.clicked.connect(lambda: self.on_delete(class_name))
        
        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        layout.addWidget(delete_btn)
        
        self.setLayout(layout)

class VideoWidget(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.selected_classes = ["__placeholder__"]
        self.model.set_classes(self.selected_classes)

        # Время, когда видели объект в последний раз
        self.last_seen = {cls: 0 for cls in self.selected_classes}
        
        # Максимальное время, которое может отсутствовать объект
        self.max_absence_time = 30
        
        # Статус-бары отсутствующих объектов
        self.status_bars = {}
        
        # Затригеренные объекты
        self.notified_objects = set()
        
        # Для уведомления
        self.setup_tray_icon()

        # Таймер для видео
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Установка стилей
        self.apply_styles()

        # Установка UI
        self.init_ui()

        # Переменная для хранения объекта захвата камеры
        self.cap = None

    def setup_tray_icon(self):
        # Создание значка в системном трее
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon.fromTheme("camera-video"))
        self.tray_icon.setVisible(True)
        
        # Запасная иконка, если иконка темы недоступна
        if self.tray_icon.icon().isNull():
            # Создание простой иконки программно
            pixmap = QPixmap(32, 32)
            pixmap.fill(QColor(COLORS["sage"]))
            self.tray_icon.setIcon(QIcon(pixmap))
            
    def apply_styles(self):
        # Установка глобальной таблицы стилей с более конкретными правилами
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['light_bg']};
                color: {COLORS['pine_green']};
                font-size: 13px;
            }}
            QLabel {{
                color: {COLORS['pine_green']};
                padding: 2px;
            }}
            QLineEdit {{
                border: 2px solid {COLORS['sage']};
                border-radius: 4px;
                padding: 8px;
                background-color: white;
                selection-background-color: {COLORS['sage']};
                color: {COLORS['pine_green']};
            }}
            QLineEdit:focus {{
                border-color: {COLORS['brown_beige']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['sage']};
                background-color: {COLORS['light_bg']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {COLORS['soft_cream']};
                color: {COLORS['pine_green']};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 15px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['sage']};
                color: white;
            }}
            QSpinBox {{
                border: 2px solid {COLORS['sage']};
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }}
            QProgressBar {{
                border: 1px solid {COLORS['soft_cream']};
                border-radius: 4px;
                text-align: center;
                height: 20px;
                margin: 2px;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
            }}
            QScrollArea {{
                border: 1px solid {COLORS['soft_cream']};
                border-radius: 4px;
                background-color: {COLORS['light_bg']};
            }}
            QFrame#separatorLine {{
                background-color: {COLORS['soft_cream']};
                max-height: 1px;
                min-height: 1px;
            }}
        """)
        
        # Установка шрифта приложения для улучшения четкости
        font = QFont("Arial", 10)
        QApplication.setFont(font)

    def init_ui(self):
        self.tabs = QTabWidget()

        # Вкладка 1: Управление классами
        tab1 = QWidget()
        tab1.setContentsMargins(0, 0, 0, 0)
        
        header_label = QLabel("Добавление и управление классами объектов")
        header_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['pine_green']}; margin: 10px 0; padding: 8px;")
        header_label.setAlignment(Qt.AlignCenter)
        
        # Секция ввода с описанием
        input_section = QWidget()
        input_section.setStyleSheet(f"background-color: white; border-radius: 8px; border: 1px solid {COLORS['soft_cream']};")
        input_layout = QVBoxLayout(input_section)
        input_layout.setContentsMargins(15, 15, 15, 15)
        
        input_description = QLabel("Введите название или описание объекта для отслеживания:")
        input_description.setStyleSheet("font-weight: bold;")
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Любой объект")
        self.input_line.setMinimumHeight(36)
        
        # Добавим описание функциональности
        help_text = QLabel("Вы можете вводить как короткие названия, так и подробные описания объектов. (на английском)")
        help_text.setStyleSheet("font-size: 12px; color: #777; font-style: italic;")
        help_text.setWordWrap(True)
        
        input_btn_layout = QHBoxLayout()
        self.add_button = StyleableButton("Добавить", color=COLORS['sage'])
        self.add_button.setMinimumWidth(120)
        self.add_button.clicked.connect(self.add_class)
        input_btn_layout.addStretch()
        input_btn_layout.addWidget(self.add_button)
        
        input_layout.addWidget(input_description)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(help_text)
        input_layout.addLayout(input_btn_layout)
        
        # Настройка максимального времени отслеживания
        time_setting = QWidget()
        time_setting.setStyleSheet(f"background-color: white; border-radius: 8px; border: 1px solid {COLORS['soft_cream']};")
        time_layout = QHBoxLayout(time_setting)
        time_layout.setContentsMargins(15, 15, 15, 15)
        
        time_label = QLabel("Максимальное время отслеживания отсутствия (сек):")
        time_label.setStyleSheet("font-weight: bold;")
        self.max_time_spinner = QSpinBox()
        self.max_time_spinner.setRange(5, 300)
        self.max_time_spinner.setValue(self.max_absence_time)
        self.max_time_spinner.valueChanged.connect(self.update_max_time)
        self.max_time_spinner.setMinimumHeight(30)
        self.max_time_spinner.setFixedWidth(80)
        
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.max_time_spinner)
        time_layout.addStretch()
        
        # Контейнер для карточек классов
        class_container_wrapper = QWidget()
        class_container_wrapper.setStyleSheet(f"background-color: white; border-radius: 8px; border: 1px solid {COLORS['soft_cream']};")
        class_container_layout = QVBoxLayout(class_container_wrapper)
        class_container_layout.setContentsMargins(15, 15, 15, 15)
        
        class_label = QLabel("Добавленные объекты:")
        class_label.setStyleSheet("font-weight: bold;")
        class_container_layout.addWidget(class_label)
        
        class_container = QWidget()
        class_container.setStyleSheet(f"background-color: {COLORS['light_bg']}; border-radius: 6px;")
        
        self.class_layout = QVBoxLayout(class_container)
        self.class_layout.setSpacing(10)
        self.class_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(class_container)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {COLORS['soft_cream']};
                border-radius: 4px;
                background-color: {COLORS['light_bg']};
            }}
        """)
        
        class_container_layout.addWidget(scroll_area)
        
        self.update_class_cards()
        
        # Добавить разделительный элемент
        spacer1 = QFrame()
        spacer1.setFixedHeight(20)
        spacer1.setStyleSheet(f"background-color: {COLORS['light_bg']};")
        
        spacer2 = QFrame()
        spacer2.setFixedHeight(20)
        spacer2.setStyleSheet(f"background-color: {COLORS['light_bg']};")
        
        layout1 = QVBoxLayout()
        layout1.addWidget(header_label)
        layout1.addWidget(input_section)
        layout1.addWidget(spacer1)
        layout1.addWidget(time_setting)
        layout1.addWidget(spacer2)
        layout1.addWidget(class_container_wrapper, 1)  # Даем коэффициент растяжения
        layout1.setContentsMargins(20, 20, 20, 20)
        tab1.setLayout(layout1)

        # Вкладка 2: Видео
        tab2 = QWidget()
        tab2.setContentsMargins(0, 0, 0, 0)
        
        video_header = QLabel("Отслеживание объектов")
        video_header.setAlignment(Qt.AlignCenter)
        video_header.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['pine_green']}; margin: 10px 0; padding: 8px;")
        
        # Создаем горизонтальный макет для камеры и панели отслеживания
        camera_tracking_layout = QHBoxLayout()
        camera_tracking_layout.setSpacing(15)
        
        # Левая часть - Секция камеры
        camera_section = QWidget()
        camera_section.setStyleSheet(f"background-color: white; border-radius: 8px; border: 1px solid {COLORS['soft_cream']};")
        camera_layout = QVBoxLayout(camera_section)
        camera_layout.setContentsMargins(15, 15, 15, 15)
        
        camera_label = QLabel("Видеокамера:")
        camera_label.setStyleSheet("font-weight: bold;")
        camera_layout.addWidget(camera_label)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet(f"border: 2px solid {COLORS['sage']}; border-radius: 8px; background-color: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        camera_controls = QWidget()
        camera_controls_layout = QHBoxLayout(camera_controls)
        camera_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_button = StyleableButton("Запустить камеру", color=COLORS['sage'])
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self.start_camera)
        
        self.stop_button = StyleableButton("Остановить камеру", color=COLORS['brown_beige'])
        self.stop_button.setMinimumWidth(150)
        self.stop_button.clicked.connect(self.stop_camera)
        
        camera_controls_layout.addStretch()
        camera_controls_layout.addWidget(self.start_button)
        camera_controls_layout.addSpacing(10)
        camera_controls_layout.addWidget(self.stop_button)
        camera_controls_layout.addStretch()
        
        camera_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        camera_layout.addWidget(camera_controls)
        
        # Правая часть - Секция отслеживания статуса
        status_section = QWidget()
        status_section.setStyleSheet(f"""
            background-color: white;
            border-radius: 8px;
            border: 1px solid {COLORS['soft_cream']};
        """)
        status_section.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        status_layout = QVBoxLayout(status_section)
        status_layout.setContentsMargins(15, 15, 15, 15)
        
        status_header = QLabel("Статус отслеживаемых объектов")
        status_header.setStyleSheet("font-weight: bold;")
        
        self.status_grid = QGridLayout()
        self.status_grid.setColumnStretch(1, 1)
        self.status_grid.setSpacing(10)
        self.update_status_bars()
        
        status_layout.addWidget(status_header)
        status_layout.addLayout(self.status_grid)
        status_layout.addStretch()
        
        # Добавляем обе секции в основной горизонтальный макет
        camera_tracking_layout.addWidget(camera_section, 7)  # 70% ширины
        camera_tracking_layout.addWidget(status_section, 3)  # 30% ширины
        
        layout2 = QVBoxLayout()
        layout2.addWidget(video_header)
        layout2.addLayout(camera_tracking_layout, 1)  # Даем коэффициент растяжения
        layout2.setContentsMargins(20, 20, 20, 20)
        tab2.setLayout(layout2)

        self.tabs.addTab(tab1, "Управление объектами")
        self.tabs.addTab(tab2, "Мониторинг")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
        # Устанавливаем разумный размер
        self.setMinimumSize(1080, 720)
    
    def update_max_time(self, value):
        self.max_absence_time = value
        self.update_status_bars()
        # Сбрасываем состояние уведомлений при изменении максимального времени
        self.notified_objects = set()

    def update_class_cards(self):
        # Очищаем существующие карточки
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Добавляем новые карточки
        display_classes = [cls for cls in self.selected_classes if cls != "__placeholder__"]
        
        if not display_classes:
            empty_label = QLabel("Список пуст. Добавьте объекты для отслеживания.")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet(f"color: {COLORS['brown_beige']}; padding: 20px;")
            self.class_layout.addWidget(empty_label)
        else:
            for cls in display_classes:
                card = ClassCard(cls, self.remove_class_by_name)
                self.class_layout.addWidget(card)
            
            # Добавляем растяжку в конце
            self.class_layout.addStretch()
    
    def remove_class_by_name(self, class_name):
        if class_name in self.selected_classes:
            self.selected_classes.remove(class_name)
            self.last_seen.pop(class_name, None)
            
            # Добавляем заполнитель, если пользователь удалил все классы
            # чтобы предотвратить сбой YOLO с пустым массивом
            if not self.selected_classes:
                # Используем специальный заполнитель, который пользователи не увидят в интерфейсе
                self.selected_classes = ["__placeholder__"]
                self.last_seen["__placeholder__"] = 0
                
            self.model.set_classes(self.selected_classes)
            self.update_class_cards()
            self.update_status_bars()

    def update_status_bars(self):
        # Очищаем существующую сетку
        while self.status_grid.count():
            item = self.status_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Создаем индикаторы статуса для каждого класса
        self.status_bars = {}
        
        # Фильтруем заполнители классов для отображения в интерфейсе
        display_classes = [cls for cls in self.selected_classes if cls != "__placeholder__"]
        
        if not display_classes:
            empty_label = QLabel("Нет объектов для отслеживания. Добавьте их во вкладке «Управление объектами»")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet(f"color: {COLORS['brown_beige']}; padding: 20px;")
            self.status_grid.addWidget(empty_label, 0, 0, 1, 2)
            return
        
        for i, cls in enumerate(display_classes):
            # Метка имени класса с иконкой
            name_container = QWidget()
            name_container.setStyleSheet(f"background-color: white;")
            name_layout = QHBoxLayout(name_container)
            name_layout.setContentsMargins(0, 0, 0, 0)
            
            icon_label = QLabel()
            icon_label.setFixedSize(16, 16)
            icon_label.setStyleSheet(f"background-color: {COLORS['sage']}; border-radius: 8px;")
            
            # Сокращаем длинные имена/описания для отображения в статусной панели
            display_name = cls
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."
                
            class_label = QLabel(display_name)
            class_label.setStyleSheet("font-weight: bold;")
            class_label.setToolTip(cls)  # Полный текст при наведении
            
            name_layout.addWidget(icon_label)
            name_layout.addWidget(class_label)
            name_layout.addStretch()
            
            # Индикатор прогресса для времени отсутствия
            progress_bar = QProgressBar()
            progress_bar.setRange(0, self.max_absence_time)
            progress_bar.setValue(0)
            progress_bar.setFormat("%v сек / %m сек")
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {COLORS['soft_cream']};
                    border-radius: 4px;
                    text-align: center;
                    height: 20px;
                    margin: 2px;
                }}
                QProgressBar::chunk {{ 
                    background-color: {COLORS['sage']};
                    border-radius: 4px;
                }}
            """)
            self.status_bars[cls] = progress_bar
            
            # Добавляем в сетку
            self.status_grid.addWidget(name_container, i*2, 0)
            self.status_grid.addWidget(progress_bar, i*2, 1)
            
            # Добавляем разделитель, если это не последний элемент
            if i < len(display_classes) - 1:
                line = QFrame()
                line.setObjectName("separatorLine")
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.status_grid.addWidget(line, i*2+1, 0, 1, 2)

    def start_camera(self):
        # Проверяем, есть ли какие-либо реальные классы для отслеживания (исключая заполнитель)
        display_classes = [cls for cls in self.selected_classes if cls != "__placeholder__"]
        if not display_classes:
            QMessageBox.warning(self, "Внимание", "Нет объектов для отслеживания. Добавьте хотя бы один объект.")
            return
            
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось получить доступ к веб-камере.")
            return
            
        # Сбрасываем состояние уведомлений при запуске камеры
        self.notified_objects = set()
        self.timer.start(30)

    def stop_camera(self):
        """Остановка камеры и сброс состояния отслеживания"""
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            
        # Сбрасываем индикаторы прогресса на ноль
        for cls, progress_bar in self.status_bars.items():
            progress_bar.setValue(0)
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {COLORS['soft_cream']};
                    border-radius: 4px;
                    text-align: center;
                    height: 20px;
                    margin: 2px;
                }}
                QProgressBar::chunk {{ 
                    background-color: {COLORS['sage']};
                    border-radius: 4px;
                }}
            """)
            
        blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        self.display_frame(blank)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()  # Правильно останавливаем камеру, если не можем получить кадр
            return

        # Проверяем, есть ли какие-либо реальные классы для отслеживания (исключая заполнитель)
        display_classes = [cls for cls in self.selected_classes if cls != "__placeholder__"]
        if not display_classes:
            # Отображаем сообщение на кадре
            blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Нет объектов для отслеживания", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            self.display_frame(blank)
            return

        now = time.time()
        results = self.model(frame)[0]

        # Обновляем время обнаружения
        detected = set()
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            if cls_idx < len(self.selected_classes):
                detected.add(self.selected_classes[cls_idx])

        for cls in detected:
            self.last_seen[cls] = now

        # Отображаем обнаружения
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.selected_classes[cls_idx] if cls_idx < len(self.selected_classes) else str(cls_idx)
            
            # Пропускаем отображение класса-заполнителя
            if class_name == "__placeholder__":
                continue
            
            # Для отображения на экране используем сокращенное имя, если текст слишком длинный
            display_name = class_name
            if len(display_name) > 20:  # На видео нужно совсем короткое название
                display_name = display_name[:17] + "..."
                
            # Убираем отображение процентов, оставляем только название объекта
            label = display_name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Проверяем объекты, отсутствующие максимальное время
        max_time_exceeded = False
        missing_objects = []
        
        # Обновляем индикаторы статуса - только для реальных классов, не заполнителей
        for cls in display_classes:
            if cls in self.status_bars:
                absence_time = int(now - self.last_seen.get(cls, 0)) if self.last_seen.get(cls, 0) > 0 else 0
                # Ограничиваем максимальным значением
                absence_time = min(absence_time, self.max_absence_time)
                self.status_bars[cls].setValue(absence_time)
                
                # Проверяем, отсутствует ли объект максимальное время
                if absence_time >= self.max_absence_time and cls not in self.notified_objects:
                    missing_objects.append(cls)
                    self.notified_objects.add(cls)
                    max_time_exceeded = True
                
                # Обновляем цвет в зависимости от времени отсутствия
                if absence_time > self.max_absence_time * 0.75:
                    self.status_bars[cls].setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #DCD0C0;
                            border-radius: 4px;
                            text-align: center;
                            height: 20px;
                        }
                        QProgressBar::chunk { 
                            background-color: #D9534F;
                            border-radius: 4px;
                        }
                    """)
                elif absence_time > self.max_absence_time * 0.5:
                    self.status_bars[cls].setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #DCD0C0;
                            border-radius: 4px;
                            text-align: center;
                            height: 20px;
                        }
                        QProgressBar::chunk { 
                            background-color: #F0AD4E;
                            border-radius: 4px;
                        }
                    """)
                else:
                    self.status_bars[cls].setStyleSheet(f"""
                        QProgressBar {{
                            border: 1px solid #DCD0C0;
                            border-radius: 4px;
                            text-align: center;
                            height: 20px;
                        }}
                        QProgressBar::chunk {{ 
                            background-color: {COLORS['sage']};
                            border-radius: 4px;
                        }}
                    """)
        
        # Показываем уведомление, если превышено максимальное время
        if max_time_exceeded:
            self.show_notification(missing_objects)
            # Останавливаем камеру после уведомления
            self.stop_camera()
            return

        self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def add_class(self):
        new_cls = self.input_line.text().strip()
        if not new_cls:
            QMessageBox.warning(self, "Внимание", "Введите название или описание объекта для отслеживания.")
            return
        if new_cls in self.selected_classes:
            QMessageBox.warning(self, "Внимание", "Этот объект уже добавлен.")
            return
            
        # Если у нас был только класс-заполнитель, полностью заменяем его
        if len(self.selected_classes) == 1 and self.selected_classes[0] == "__placeholder__":
            self.selected_classes = [new_cls]
        else:
            self.selected_classes.append(new_cls)
            
        self.model.set_classes(self.selected_classes)
        
        # Новый класс пока не обнаружен -> считаем отсутствующим
        self.last_seen[new_cls] = 0
        self.update_class_cards()
        self.update_status_bars()
        self.input_line.clear()

    def show_notification(self, missing_objects):
        """Показать уведомление об отсутствующих объектах"""
        if not missing_objects:
            return
            
        # Для уведомления также используем короткие имена
        shortened_objects = []
        for obj in missing_objects:
            if len(obj) > 30:
                shortened_objects.append(obj[:27] + "...")
            else:
                shortened_objects.append(obj)
                
        object_list = ", ".join(shortened_objects)
        title = "Внимание! Объекты отсутствуют"
        message = f"Следующие объекты отсутствуют более {self.max_absence_time} секунд: {object_list}"
        
        # Показываем уведомление в системном трее
        self.tray_icon.showMessage(title, message, QSystemTrayIcon.Warning, 5000)
        
        # Также показываем диалоговое окно для большей заметности
        QMessageBox.warning(self, title, message)
        
        # Сбрасываем время последнего обнаружения для всех объектов, чтобы разрешить перезапуск камеры
        for obj in self.selected_classes:
            self.last_seen[obj] = 0
        
        # Сбрасываем состояние уведомлений, чтобы разрешить новые уведомления
        self.notified_objects = set()

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    model = YOLO('/Users/pavelstarostin/Source/course_work/LVIS.pt')
    app = QApplication(sys.argv)
    # Установка шрифта для всего приложения для улучшения четкости
    app.setFont(QFont("Arial", 10))
    window = VideoWidget(model)
    window.setWindowTitle('Мониторинг объектов')
    window.show()
    sys.exit(app.exec_())
