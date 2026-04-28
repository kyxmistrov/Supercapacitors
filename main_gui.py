import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QTabWidget, QComboBox, QCheckBox,
    QLineEdit, QFormLayout, QScrollArea, QProgressDialog
)

from visualization.plot_utils import (
    clear_output_dirs, plot_histograms, plot_correlation_heatmap,
    plot_scan_speed_vs_capacity, plot_thickness_vs_energy_power,
    plot_electrolyte_material_influence, plot_3d_energy_power_thickness,
    plot_capacity_vs_scan_speed_and_current_density, plot_capacity_vs_surface_area,
    animate_charge_discharge, clear_dir
)
from PyQt6.QtGui import QKeySequence
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer
from core.data_loader import load_input_data
from core.physics import calculate_all
from core.ml.predict_models import is_enough_data_for_prediction, predict_on_raw_data, clear_output_dirs as clear_ml_outputs
from optimization.genetic_optimizer import optimize_parameters
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QThread, pyqtSignal
from fpdf import FPDF
from textwrap import shorten
from datetime import datetime
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QStyledItemDelegate
from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import QSize

DEFAULT_COLUMNS = [
    "Тип материала", "Площадь поверхности (м²/г)", "Размер пор (нм)", "Гетероатомы", "ID/IG",
    "Толщина слоя (мкм)", "PSD", "Пористость (%)", "Уд. поверхность (м²/см³)", "Тип электролита",
    "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)", "Температура (°C)",
    "Скорость скан. (В/с)", "Диапазон EIS (Гц)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
]

GOAL_MAPPING = {
    "Удельная ёмкость": "capacity",
    "Срок службы": "lifetime",
    "Эффективность хранения": "efficiency"
}

MIN_REQUIRED_ROWS = 10
ALLOW_SMALL_DATA = True


class OptimizationReportThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, df_input, df_results, graph_dir, save_path, goal_text, constraints, include_secondary):
        super().__init__()
        self.df_input = df_input
        self.df_results = df_results
        self.graph_dir = graph_dir
        self.save_path = save_path
        self.goal_text = goal_text
        self.constraints = constraints
        self.include_secondary = include_secondary
        

    def run(self):
        try:
            generate_optimization_report(
                df_input=self.df_input,
                df_results=self.df_results,
                graph_dir=self.graph_dir,
                output_path=self.save_path,
                goal_text=self.goal_text,
                constraints=self.constraints,
                include_secondary=self.include_secondary
            )
            self.finished.emit(self.save_path)
        except Exception as e:
            self.error.emit(str(e))


class PhysicsCalculationThread(QThread):
    finished = pyqtSignal(pd.DataFrame, pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, df_input):
        super().__init__()
        self.df_input = df_input

    def run(self):
        try:
            df_results = calculate_all(self.df_input)
            self.finished.emit(self.df_input, df_results)
        except Exception as e:
            self.error.emit(str(e))




class PredictionCalculationThread(QThread):
    finished = pyqtSignal(pd.DataFrame, pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, df_input):
        super().__init__()
        self.df_input = df_input

    def run(self):
        try:
            df_phys = calculate_all(self.df_input)
            df_results = predict_on_raw_data(self.df_input, df_phys)
            self.finished.emit(self.df_input, df_results)
        except Exception as e:
            self.error.emit(str(e))


class ReportGenerationThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, df_input, df_results, graph_dir, save_path):
        super().__init__()
        self.df_input = df_input
        self.df_results = df_results
        self.graph_dir = graph_dir
        self.save_path = save_path

    def run(self):
        try:
            generate_physics_report(self.df_input, self.df_results, self.graph_dir, self.save_path)
            self.finished.emit(self.save_path)
        except Exception as e:
            self.error.emit(str(e))
            


class PredictionReportThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, df_input, df_results, graph_dir, save_path):
        super().__init__()
        self.df_input = df_input
        self.df_results = df_results
        self.graph_dir = graph_dir
        self.save_path = save_path

    def run(self):
        try:
            generate_prediction_report(self.df_input, self.df_results, self.graph_dir, self.save_path)
            self.finished.emit(self.save_path)
        except Exception as e:
            self.error.emit(str(e))

class PlottingThread(QThread):
    finished = pyqtSignal(str)  
    error = pyqtSignal(str)

    def __init__(self, df_result: pd.DataFrame, base_path: str):
        super().__init__()
        self.df_result = df_result
        self.base_path = base_path

    def run(self):
        try:
            hist_path = os.path.join(self.base_path, "histograms")
            graph_path = os.path.join(self.base_path, "graphics")
            os.makedirs(hist_path, exist_ok=True)
            os.makedirs(graph_path, exist_ok=True)
            clear_dir(hist_path)
            clear_dir(graph_path)

            if len(self.df_result.dropna()) < 3:
                plot_histograms(self.df_result, out_dir=hist_path, show_mean=True, show_median=True)
            else:
                plot_histograms(self.df_result, out_dir=hist_path, show_mean=True, show_median=True)
                plot_correlation_heatmap(self.df_result, out_dir=graph_path)
                plot_scan_speed_vs_capacity(self.df_result, out_dir=graph_path)
                plot_thickness_vs_energy_power(self.df_result, out_dir=graph_path)
                plot_electrolyte_material_influence(self.df_result, out_dir=graph_path)
                plot_3d_energy_power_thickness(self.df_result, out_dir=graph_path)
                plot_capacity_vs_scan_speed_and_current_density(self.df_result, out_dir=graph_path)
                plot_capacity_vs_surface_area(self.df_result, out_dir=graph_path)

            self.finished.emit(self.base_path)
        except Exception as e:
            self.error.emit(str(e))


class ProgressDialog(QDialog):
    def __init__(self, text="Выполняется оптимизация...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Обработка данных")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()
        self.label = QLabel(text)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # бесконечная анимация

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

class OptimizationWorkerThread(QThread):
    finished = pyqtSignal(object, object)

    def __init__(self, df_input, goal, constraints, include_secondary):
        super().__init__()
        self.df_input = df_input
        self.goal = goal
        self.constraints = constraints
        self.include_secondary = include_secondary

    def run(self):
        try:
            df_result = optimize_parameters(
                df_start=self.df_input,
                optimization_goal=self.goal,
                custom_constraints=self.constraints,
                include_secondary_metrics=self.include_secondary
            )
            self.finished.emit(self.df_input, df_result)
        except Exception as e:
            self.finished.emit(None, str(e))




class SharedCalculationThread(QThread):
    finished = pyqtSignal(str, object, object, QTableWidget)

    def __init__(self, mode, table):
        super().__init__()
        self.mode = mode
        self.table = table

    def run(self):
        try:
            data = []
            for row in range(self.table.rowCount()):
                row_data = []
                empty_row = True
                for col in range(len(DEFAULT_COLUMNS)):
                    item = self.table.item(row, col)
                    value = item.text() if item else ""
                    if value.strip():
                        empty_row = False
                    row_data.append(value.strip())
                if not empty_row:
                    data.append(row_data)

            df_input = pd.DataFrame(data, columns=DEFAULT_COLUMNS)
            numeric_cols = [
                "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "Пористость (%)",
                "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
                "Температура (°C)", "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
            ]
            for col in numeric_cols:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            df_input.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
            df_input.dropna(subset=numeric_cols, inplace=True)
            for col in ["Толщина слоя (мкм)", "Площадь поверхности (м²/г)", "Площадь электрода (см²)", "ESR (Ом)"]:
                df_input = df_input[df_input[col] != 0]

            if df_input.empty:
                raise ValueError("Недостаточно корректных данных после фильтрации.")

          
            if self.mode == "physics":
                df_results = calculate_all(df_input)
            elif self.mode == "prediction":
                df_phys = calculate_all(df_input)
                df_results = predict_on_raw_data(df_input, df_phys)
            else:
                raise ValueError("Неверный режим обработки.")

            self.finished.emit(self.mode, df_input, df_results, self.table)

        except Exception as e:
            self.finished.emit(self.mode, None, str(e), self.table)

class UndoDelegate(QStyledItemDelegate):
    def __init__(self, parent, table, undo_stack_ref, is_programmatic_edit_ref):
        super().__init__(parent)
        self.table = table
        self.undo_stack_ref = undo_stack_ref
        self.is_programmatic_edit_ref = is_programmatic_edit_ref
        

    def createEditor(self, parent, option, index):
        self._editing_row = index.row()
        self._editing_col = index.column()
        item = self.table.item(self._editing_row, self._editing_col)
        self._prev_value = item.text() if item else ""
        return super().createEditor(parent, option, index)

    def setModelData(self, editor, model, index):
        if not self.is_programmatic_edit_ref():
            row = index.row()
            col = index.column()
            new_value = editor.text()
            prev_value = model.data(index)
            if new_value != prev_value:
                self.undo_stack_ref().append([(row, col, prev_value)])
        super().setModelData(editor, model, index)
  



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Суперконденсаторы")
        self.setWindowIcon(QIcon("gui/logo.png"))
        self.resize(1400, 700)
        self.tabs = QTabWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.tabs.addTab(self.create_shared_tab("physics"), "Физическое моделирование")
        self.tabs.addTab(self.create_shared_tab("prediction"), "Предсказание параметров")
        self.tabs.addTab(self.create_optimization_tab(), "Оптимизация")
        self.calculations_done = {"physics": False, "prediction": False}
        self.tabs.currentChanged.connect(self.attach_key_event)
        
        self.scale_physics = 1000
        self.scale_prediction = 1000
        self.scale_optimization = 1000
        self.scale_labels = {} 
        self.original_pixmaps = {}
        self.scroll_areas = {}
        



    def attach_key_event(self):
        current_tab = self.tabs.currentWidget()
        table = current_tab.findChild(QTableWidget)
        current_tab.keyPressEvent = lambda event: self.handle_key_event(event, table)

    def _invalidate_report_flag(self, mode):
        self.calculations_done[mode] = False

    def on_report_success(self, path):
        self.progress_dialog.close()
        QMessageBox.information(self, "Готово", f"Отчёт успешно сохранён:\n{path}")

    def on_report_error(self, error_msg):
        self.progress_dialog.close()
        self.show_error(f"Ошибка при формировании отчёта: {error_msg}")

    def parse_constraint(self, val, col_name=None):
        val = val.strip()
        if not val:
            return None
        if val.lower() in ["нет", "-"]:
            return val
        if ";" in val:
            return [p.strip() for p in val.split(";") if p.strip()]
        if val.startswith("(") and val.endswith(")"):
            inner = val[1:-1].split(',')
            try:
                return tuple(map(float, inner))
            except:
                return tuple(part.strip() for part in inner)
        if col_name != "Диапазон EIS (Гц)" and '-' in val and not any(c.isalpha() for c in val):
            parts = val.split('-')
            try:
                return (float(parts[0]), float(parts[1]))
            except:
                return val
        if ',' in val:
            parts = [p.strip() for p in val.split(',')]
            try:
                return [float(p) for p in parts]
            except:
                return parts
        try:
            return float(val)
        except:
            return val

    
    def generate_report_for_physics(self, mode):
        if not self.calculations_done.get(mode):
            self.show_error("Сначала выполните расчёты перед формированием отчёта.")
            return

        df_results = getattr(self, "df_results_storage", None)
        if df_results is None or df_results.empty:
            self.show_error("Нет данных для отчёта.")
            return

        table = self.tables[mode]
        df_input = self._extract_df_from_table(table)
        graph_dir = os.path.join("visualization", "plots", "physics", "graphics")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PDF отчёт", "physics_report.pdf", "PDF файлы (*.pdf)"
        )
        if not save_path:
            return

        # Показываем прогресс
        self.progress_dialog = ProgressDialog("Формирование PDF отчёта...", self)
        self.progress_dialog.show()
        QApplication.processEvents()

        # Поток отчёта
        self.report_thread = ReportGenerationThread(df_input, df_results, graph_dir, save_path)
        self.report_thread.finished.connect(self.on_report_success)
        self.report_thread.error.connect(self.on_report_error)
        self.report_thread.start()


    def on_visualization_finished(self, base_path, mode):
        try:
            hist_path = os.path.join(base_path, "histograms")
            graph_path = os.path.join(base_path, "graphics")

            graphics_widget = self.graphics_tabs.get(mode)
            if graphics_widget is None:
                return

            layout = graphics_widget.layout()
            if layout is None:
                layout = QVBoxLayout()
                graphics_widget.setLayout(layout)

            if mode in self.scroll_areas:
                old_scroll = self.scroll_areas[mode]
                layout.removeWidget(old_scroll)
                old_scroll.deleteLater()


            # Контейнер с графиками
            graph_container = QWidget()
            graph_layout = QVBoxLayout()

            self.original_pixmaps[mode] = {}

            for folder in [hist_path, graph_path]:
                for file in sorted(os.listdir(folder)):
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                        path = os.path.join(folder, file)
                        pixmap = QPixmap(path)
                        if not pixmap.isNull():
                            label = QLabel()
                            scaled_width = getattr(self, f"scale_{mode}", 1000)
                            label.setPixmap(pixmap.scaledToWidth(scaled_width, Qt.TransformationMode.SmoothTransformation))
                            self.original_pixmaps[mode][label] = pixmap

                            hlayout = QHBoxLayout()
                            hlayout.addStretch(1)
                            hlayout.addWidget(label)
                            hlayout.addStretch(1)
                            graph_layout.addLayout(hlayout)

            graph_container.setLayout(graph_layout)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(graph_container)

            # Получаем виджет графиков
            graphics_widget = self.graphics_tabs.get(mode)
            if graphics_widget is None:
                return

            layout = graphics_widget.layout()
            if layout is None:
                layout = QVBoxLayout()
                graphics_widget.setLayout(layout)

            # Удаляем всё кроме scale_slider и scale_label
            items_to_keep = set()
            if mode in self.scale_labels:
                items_to_keep.add(self.scale_labels[mode])
            if hasattr(self, f"scale_slider_{mode}"):
                items_to_keep.add(getattr(self, f"scale_slider_{mode}"))

            # Удаляем всё, что не ползунок и не подпись
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget and widget not in items_to_keep:
                    widget.setParent(None)

            # Если слайдера и метки масштаба ещё не было — создаём
            if mode not in self.scale_labels:
                scale_slider = QSlider(Qt.Orientation.Horizontal)
                scale_slider.setRange(600, 1600)
                scale_slider.setValue(getattr(self, f"scale_{mode}", 1000))
                scale_slider.setTickInterval(100)
                scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                setattr(self, f"scale_slider_{mode}", scale_slider)

                scale_slider_label = QLabel(f"Масштаб: {scale_slider.value()} px")
                self.scale_labels[mode] = scale_slider_label

                scale_slider.valueChanged.connect(lambda val, m=mode: self.update_graphics_scale(m, val))

                scale_panel = QHBoxLayout()
                scale_panel.addWidget(scale_slider_label)
                scale_panel.addWidget(scale_slider)
                scale_panel.addStretch()
                layout.addLayout(scale_panel)

            self.scale_labels[mode].setText(f"Масштаб: {getattr(self, f'scale_{mode}', 1000)} px")
            
            if mode in self.scale_labels:
                self.scale_labels[mode].setVisible(True)
            if hasattr(self, f"scale_slider_{mode}"):
                getattr(self, f"scale_slider_{mode}").setVisible(True)
            layout.addWidget(scroll)

            self.scroll_areas[mode] = scroll
        finally:
            if hasattr(self, "progress_dialog"):
                self.progress_dialog.close()




    def update_graphics_scale(self, mode, scale_value):
        scroll_widget = self.scroll_areas.get(mode)
        if not scroll_widget:
            return
        setattr(self, f"scale_{mode}", scale_value)

        if mode in self.scale_labels:
            self.scale_labels[mode].setText(f"Масштаб: {scale_value} px")

        container = scroll_widget.widget()
        if not container:
            return

        layout = container.layout()
        if not layout:
            return

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    sub_item = item.itemAt(j)
                    if sub_item:
                        widget = sub_item.widget()
                        if isinstance(widget, QLabel) and mode in self.original_pixmaps:
                            orig_pixmap = self.original_pixmaps[mode].get(widget)
                            if orig_pixmap:
                                widget.setPixmap(orig_pixmap.scaledToWidth(scale_value, Qt.TransformationMode.SmoothTransformation))
                                

    def handle_finished(self, df_input, result):
        self.progress_dialog.close()

        if isinstance(result, str):
            self.show_error(result)
            return

        if isinstance(result, pd.DataFrame):
            self.display_table(result, self.optim_table)
            self.visual(result, "visualization/plots/optimization", mode="optimization")
            self.df_results_storage = result
            self.calculations_done["optimization"] = True
            if hasattr(self, "report_buttons") and "optimization" in self.report_buttons:
                self.report_buttons["optimization"].setEnabled(True)

            self.progress_dialog.close()

    def export_table_to_excel_or_csv(self, table: QTableWidget):
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл",
            "",
            "Excel (*.xlsx);;CSV (*.csv)"
        )
        if not path:
            return

        df = pd.DataFrame(columns=[table.horizontalHeaderItem(i).text() for i in range(table.columnCount())])
        for row in range(table.rowCount()):
            row_data = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                row_data.append(item.text() if item else "")
            df.loc[row] = row_data

        try:
            if selected_filter == "CSV (*.csv)" or path.endswith(".csv"):
                if not path.endswith(".csv"):
                    path += ".csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                if not path.endswith(".xlsx"):
                    path += ".xlsx"
                df.to_excel(path, index=False)

            QMessageBox.information(self, "Успех", f"Данные успешно сохранены в файл:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")



    def handle_finished_shared(self, mode, df_input, result, table):
        if isinstance(result, str):
            self.progress_dialog.close()
            self.show_error(result)
            return

        try:
            if mode == "physics":
                self.display_table(result, table)
                self.df_results_storage = result
                self.calculations_done[mode] = True
                if hasattr(self, "report_buttons") and mode in self.report_buttons:
                    self.report_buttons[mode].setEnabled(True)
                self.visual(result, "visualization/plots/physics", on_complete_callback=self.progress_dialog.close)
                return

            elif mode == "prediction":
                enough = is_enough_data_for_prediction(result, MIN_REQUIRED_ROWS)
                if enough or ALLOW_SMALL_DATA:
                    if not enough:
                        print("⚠️ Недостаточно данных для точного предсказания")
                    df_results = predict_on_raw_data(df_input, result)
                    self.df_results_storage = df_results
                    ml_cols = [col for col in df_results.columns if "ML" in col]
                    combined = pd.concat([df_input.reset_index(drop=True), df_results[ml_cols].reset_index(drop=True)], axis=1)
                    self.display_table(combined, table)

                    graph_dir = os.path.join("visualization", "plots", "models")
                    prediction_graphics_widget = self.graphics_tabs.get("prediction")
                    if prediction_graphics_widget:
                        container = QWidget()
                        layout = QVBoxLayout()
                        self.original_pixmaps["prediction"] = {}

                        if os.path.exists(graph_dir):
                            for file in sorted(os.listdir(graph_dir)):
                                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                                    path = os.path.join(graph_dir, file)
                                    pixmap = QPixmap(path)
                                    if not pixmap.isNull():
                                        label = QLabel()
                                        scale = getattr(self, "scale_prediction", 1000)
                                        label.setPixmap(pixmap.scaledToWidth(scale, Qt.TransformationMode.SmoothTransformation))
                                        self.original_pixmaps["prediction"][label] = pixmap

                                        hlayout = QHBoxLayout()
                                        hlayout.addStretch(1)
                                        hlayout.addWidget(label)
                                        hlayout.addStretch(1)
                                        layout.addLayout(hlayout)

                        container.setLayout(layout)
                        scroll = QScrollArea()
                        scroll.setWidgetResizable(True)
                        scroll.setWidget(container)
                        self.scroll_areas["prediction"] = scroll

                        pred_layout = prediction_graphics_widget.layout()
                        if pred_layout is None:
                            pred_layout = QVBoxLayout()
                            prediction_graphics_widget.setLayout(pred_layout)

                        for i in reversed(range(pred_layout.count())):
                            item = pred_layout.itemAt(i)
                            widget = item.widget() if item else None
                            if widget:
                                widget.setParent(None)

                        if "prediction" not in self.scale_labels:
                            scale_slider = QSlider(Qt.Orientation.Horizontal)
                            scale_slider.setRange(600, 1600)
                            scale_slider.setValue(getattr(self, "scale_prediction", 1000))
                            scale_slider.setTickInterval(100)
                            scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
                            setattr(self, "scale_slider_prediction", scale_slider)

                            scale_label = QLabel(f"Масштаб: {scale_slider.value()} px")
                            self.scale_labels["prediction"] = scale_label

                            scale_slider.valueChanged.connect(lambda val: self.update_graphics_scale("prediction", val))

                            scale_panel = QHBoxLayout()
                            scale_panel.addWidget(scale_label)
                            scale_panel.addWidget(scale_slider)
                            scale_panel.addStretch()
                            pred_layout.addLayout(scale_panel)

                        self.scale_labels["prediction"].setText(f"Масштаб: {getattr(self, 'scale_prediction', 1000)} px")
                        pred_layout.addWidget(scroll)

                    self.calculations_done[mode] = True
                    if hasattr(self, "report_buttons") and mode in self.report_buttons:
                        self.report_buttons[mode].setEnabled(True)
                    return
                else:
                    clear_ml_outputs()
                    self.show_error("Недостаточно данных для предсказания. Выполнены только физические расчёты.")
                    return

        except Exception as e:
            self.show_error(str(e))
        finally:
            self.progress_dialog.close()


    def visual(self, result: pd.DataFrame, base_path: str, mode: str = "physics", on_complete_callback=None):
        self.progress_dialog = ProgressDialog("Генерация графиков...", self)
        self.progress_dialog.show()

        self.plot_thread = PlottingThread(result, base_path)

        def finalize_visualization(path):
            self.on_visualization_finished(path, mode)
            self.progress_dialog.close()
            if on_complete_callback:
                on_complete_callback()


        def handle_error(err):
            self.show_error(f"Ошибка визуализации: {err}")
            self.progress_dialog.close()

        self.plot_thread.finished.connect(finalize_visualization)
        self.plot_thread.error.connect(handle_error)
        self.plot_thread.start()

    def load_model_graphics(self):
        graph_dir = os.path.join("visualization", "plots", "models")
        prediction_graphics_widget = self.graphics_tabs.get("prediction")
        if not prediction_graphics_widget:
            return

        container = QWidget()
        layout = QVBoxLayout()

        if os.path.exists(graph_dir):
            for file in sorted(os.listdir(graph_dir)):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(graph_dir, file)
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        label = QLabel()
                        label.setPixmap(pixmap.scaledToWidth(1000, Qt.TransformationMode.SmoothTransformation))
                        hlayout = QHBoxLayout()
                        hlayout.addStretch(1)
                        hlayout.addWidget(label)
                        hlayout.addStretch(1)
                        layout.addLayout(hlayout)

        container.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        for i in reversed(range(prediction_graphics_widget.layout().count())):
            item = prediction_graphics_widget.layout().itemAt(i)
            if item:
                w = item.widget()
                if w:
                    w.setParent(None)
        prediction_graphics_widget.layout().addWidget(scroll)



    def create_graphics_tab(self, image_dir: str) -> QWidget:
        widget = QWidget()
        scroll = QScrollArea()
        layout = QVBoxLayout()

        for file in sorted(os.listdir(image_dir)):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(image_dir, file)
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    label = QLabel()
                    label.setPixmap(pixmap.scaledToWidth(1000, Qt.TransformationMode.SmoothTransformation))
                    hlayout = QHBoxLayout()
                    hlayout.addStretch(1)
                    hlayout.addWidget(label)
                    hlayout.addStretch(1)
                    layout.addLayout(hlayout)


        container = QWidget()
        container.setLayout(layout)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        return widget
    

    def handle_table_edit_operations(self, table: QTableWidget):
        self._undo_stack = []
        self._redo_stack = []
        self._is_programmatic_edit = False  

        def copy_selection():
            selected_ranges = table.selectedRanges()
            if not selected_ranges:
                return
            clipboard_data = []
            for selection in selected_ranges:
                for row in range(selection.topRow(), selection.bottomRow() + 1):
                    row_data = []
                    for col in range(selection.leftColumn(), selection.rightColumn() + 1):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    clipboard_data.append("\t".join(row_data))
            QApplication.clipboard().setText("\n".join(clipboard_data))

        def paste_selection():
            clipboard = QApplication.clipboard().text()
            if not clipboard:
                return
            selected_ranges = table.selectedRanges()
            start_row = selected_ranges[0].topRow() if selected_ranges else 0
            start_col = selected_ranges[0].leftColumn() if selected_ranges else 0
            rows = clipboard.split("\n")
            change_block = []
            self._is_programmatic_edit = True  # ⛔ отключаем on_manual_edit временно
            for i, row_data in enumerate(rows):
                cells = row_data.split("\t")
                row = start_row + i
                if row >= table.rowCount():
                    table.insertRow(row)
                for j, cell in enumerate(cells):
                    col = start_col + j
                    if col < table.columnCount():
                        prev_item = table.item(row, col)
                        prev_value = prev_item.text() if prev_item else ""
                        change_block.append((row, col, prev_value))
                        item = QTableWidgetItem(cell)
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                        table.setItem(row, col, item)
            self._is_programmatic_edit = False  # ✅ возвращаем
            if change_block:
                self._undo_stack.append(change_block)

        def delete_selection():
            selected_ranges = table.selectedRanges()
            change_block = []
            self._is_programmatic_edit = True
            for selection in selected_ranges:
                for row in range(selection.topRow(), selection.bottomRow() + 1):
                    for col in range(selection.leftColumn(), selection.rightColumn() + 1):
                        prev_item = table.item(row, col)
                        prev_value = prev_item.text() if prev_item else ""
                        change_block.append((row, col, prev_value))
                        table.setItem(row, col, QTableWidgetItem(""))
            self._is_programmatic_edit = False
            if change_block:
                self._undo_stack.append(change_block)

        def undo_last_change():
            if not self._undo_stack:
                return
            change_block = self._undo_stack.pop()
            redo_block = []
            self._is_programmatic_edit = True
            for row, col, old_value in change_block:
                current_item = table.item(row, col)
                current_value = current_item.text() if current_item else ""
                redo_block.append((row, col, current_value))
                table.blockSignals(True)
                table.setItem(row, col, QTableWidgetItem(old_value))
                table.blockSignals(False)
            self._is_programmatic_edit = False
            self._redo_stack.append(redo_block)

        def redo_last_change():
            if not self._redo_stack:
                return
            change_block = self._redo_stack.pop()
            undo_block = []
            self._is_programmatic_edit = True
            for row, col, value in change_block:
                current_item = table.item(row, col)
                current_value = current_item.text() if current_item else ""
                undo_block.append((row, col, current_value))
                table.blockSignals(True)
                table.setItem(row, col, QTableWidgetItem(value))
                table.blockSignals(False)
            self._is_programmatic_edit = False
            self._undo_stack.append(undo_block)

        def on_manual_edit(row, col):
            if self._is_programmatic_edit:
                return  # не сохраняем правку
            prev_item = table.item(row, col)
            prev_value = prev_item.text() if prev_item else ""
            self._undo_stack.append([(row, col, prev_value)])

        #table.itemChanged.connect(lambda item: on_manual_edit(item.row(), item.column()))

        def keyPressEvent(event):
            if event.matches(QKeySequence.StandardKey.Copy):
                copy_selection()
            elif event.matches(QKeySequence.StandardKey.Paste):
                paste_selection()
            elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                delete_selection()
            elif event.matches(QKeySequence.StandardKey.Undo):
                undo_last_change()
            elif event.matches(QKeySequence.StandardKey.Redo):  # Ctrl+Y
                redo_last_change()
            else:
                QWidget.keyPressEvent(table, event)


        table.setItemDelegate(
            UndoDelegate(
                parent=table,
                table=table,
                undo_stack_ref=lambda: self._undo_stack,
                is_programmatic_edit_ref=lambda: self._is_programmatic_edit
            )
        )

        table.keyPressEvent = keyPressEvent







    def create_shared_tab(self, mode):
        tab_widget = QTabWidget()

        # Таб "Таблица"
        table_tab = QWidget()
        layout = QVBoxLayout()
        label = QLabel("Файл не выбран")
        info_button = QPushButton()
        info_button.setIcon(QIcon("gui/icons/bulb.svg"))  # относительный путь
        info_button.setIconSize(QSize(15, 15))
        info_button.setFixedSize(25, 25)
        info_button.setStyleSheet("border: none; color: white;")
        info_button.setFixedWidth(25)
        info_button.setToolTip("Инструкция по режиму работы")

        # Горизонтальное выравнивание метки и кнопки
        top_info_layout = QHBoxLayout()
        top_info_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        top_info_layout.addWidget(info_button)
        top_info_layout.addWidget(label)


        layout.addLayout(top_info_layout)


        table = QTableWidget()
        self.handle_table_edit_operations(table)
        table.setColumnCount(len(DEFAULT_COLUMNS))
        table.setHorizontalHeaderLabels(DEFAULT_COLUMNS)
        table.setRowCount(10)
        layout.addWidget(table)

        btn_layout = QHBoxLayout()
        load_button = QPushButton("Импорт")
        clear_button = QPushButton("Очистить")
        add_row_button = QPushButton("Новая строка")
        process_button = QPushButton("Рассчитать")
        export_button = QPushButton("Экспорт")
        report_button = QPushButton("Отчет (PDF)")
        


        if mode == "prediction":
            report_button.clicked.connect(lambda: self.generate_report_for_prediction(mode))
        else:
            report_button.clicked.connect(lambda: self.generate_report_for_physics(mode))

        report_button.setEnabled(False)

        if not hasattr(self, "report_buttons"):
            self.report_buttons = {}
        self.report_buttons[mode] = report_button

        btn_layout.addWidget(load_button)
        btn_layout.addWidget(export_button)
        btn_layout.addWidget(clear_button)
        btn_layout.addWidget(add_row_button)
        btn_layout.addWidget(process_button)
        btn_layout.addWidget(report_button)
        btn_layout.addWidget(info_button)

        
        layout.addLayout(btn_layout)
        table_tab.setLayout(layout)

        # Таб "Графики"
        graphics_tab = QWidget()
        graphics_layout = QVBoxLayout()
        graphics_label = QLabel("Графики появятся после расчёта")
        graphics_layout.addWidget(graphics_label)
        graphics_tab.setLayout(graphics_layout)

        # Добавляем в QTabWidget
        tab_widget.addTab(table_tab, "Таблица")
        tab_widget.addTab(graphics_tab, "Графики")

        # Сохраняем ссылки
        if not hasattr(self, "graphics_tabs"):
            self.graphics_tabs = {}
        if not hasattr(self, "tables"):
            self.tables = {}
        if not hasattr(self, "labels"):
            self.labels = {}

        self.graphics_tabs[mode] = graphics_tab
        self.tables[mode] = table
        self.labels[mode] = label

        # Обработка событий
        def load():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Импорт",
                "",
                "Файлы Excel/CSV (*.xlsx *.xls *.csv);"
            )
            if file_path:
                try:
                    if file_path.endswith(".csv"):
                        df = pd.read_csv(file_path, encoding="utf-8-sig")
                    else:
                        df = pd.read_excel(file_path)

                    missing = [col for col in DEFAULT_COLUMNS if col not in df.columns]
                    if missing:
                        raise KeyError(f"В файле отсутствуют столбцы: {', '.join(missing)}")
                    df = df[DEFAULT_COLUMNS]
                    self.display_table(df, table)
                    label.setText(f"Загружен файл: {file_path}")
                    self.calculations_done[mode] = False
                except Exception as e:
                    self.show_error(str(e))

        def clear():
            table.clearContents()
            table.setRowCount(10)
            table.setHorizontalHeaderLabels(DEFAULT_COLUMNS)
            label.setText("Файл не выбран")
            self.calculations_done[mode] = False

            graphics_widget = self.graphics_tabs.get(mode)
            if graphics_widget:
                for i in reversed(range(graphics_widget.layout().count())):
                    item = graphics_widget.layout().itemAt(i)
                    if item:
                        w = item.widget()
                        if w:
                            w.setParent(None)

                # Удаляем ползунок и метку масштаба полностью
                if mode in self.scale_labels:
                    scale_label = self.scale_labels.pop(mode)
                    scale_label.setParent(None)
                slider_attr = f"scale_slider_{mode}"
                if hasattr(self, slider_attr):
                    slider = getattr(self, slider_attr)
                    slider.setParent(None)
                    delattr(self, slider_attr)


                graphics_widget.layout().addWidget(QLabel("Графики появятся после расчёта"))

            if hasattr(self, "report_buttons") and mode in self.report_buttons:
                self.report_buttons[mode].setEnabled(False)

        def show_info():
            if mode == "physics":
                message = (
                    "<b>Инструкция по модулю физического моделирования:</b><br><br>"
                    "Режим <b>«Физическое моделирование»</b> позволяет рассчитать ключевые характеристики суперконденсаторов на основе введённых параметров.<br><br>"
                    "<u>Что нужно сделать:</u><br>"
                    "• Заполните таблицу значениями<br>"
                    "• Нажмите кнопку <b>«Рассчитать»</b><br><br>"
                    "<u>Что будет выполнено:</u><br>"
                    "• Расчёт удельной ёмкости, энергии, саморазряда, потерь и других параметров<br>"
                    "• Автоматическая генерация графиков<br><br>"
                    "После этого можно сформировать PDF-отчёт"
                )
            elif mode == "prediction":
                message = (
                    "<b>Инструкция по модулю предсказания параметров:</b><br><br>"
                    "Режим <b>«Предсказание параметров»</b> использует машинное обучение для оценки характеристик суперконденсаторов на основе введённых данных.<br><br>"
                    "<u>Что нужно сделать:</u><br>"
                    "• Заполните таблицу параметрами устройства<br>"
                    "• Нажмите кнопку <b>«Рассчитать»</b><br><br>"
                    "<u>Что будет выполнено:</u><br>"
                    "• Применяются обученные ML-модели для предсказания таких параметров, как:<br>"
                    "  – Удельная ёмкость (ML)<br>"
                    "  – Срок службы (ML)<br>"
                    "  – Эффективность хранения (ML)<br>"
                    "• Построение графиков с визуализацией предсказаний и точности моделей<br><br>"
                    "После этого можно сформировать PDF-отчёт"
                )
            
            msg = QMessageBox()
            msg.setWindowTitle("Инструкция")
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(message)
            msg.exec()


        def add():
            table.blockSignals(True)  # 🔒 отключаем сигналы временно
            row = table.rowCount()
            table.insertRow(row)
            for j in range(table.columnCount()):
                table.setItem(row, j, QTableWidgetItem(""))
            table.blockSignals(False)  # 🔓 включаем обратно
            


        def process():
            try:
                has_data = any(table.item(r, c) and table.item(r, c).text().strip()
                            for r in range(table.rowCount())
                            for c in range(table.columnCount()))
                if not has_data:
                    self.show_error("Таблица пуста. Введите данные перед выполнением расчёта.")
                    return

                required_numeric = [
                    "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "Пористость (%)",
                    "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
                    "Температура (°C)", "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
                ]
                if self.highlight_invalid_cells(table, required_numeric):
                    self.show_error("Некоторые числовые ячейки пусты или содержат ошибки.")
                    return

                self.progress_dialog = ProgressDialog("Выполняется расчёт...", self)
                self.progress_dialog.show()

                df_input = self._extract_df_from_table(table)
                if mode == "physics":
                    self.worker = PhysicsCalculationThread(df_input)
                else:
                    self.worker = PredictionCalculationThread(df_input)

                self.worker.finished.connect(lambda inp, res: self.handle_finished_shared(mode, inp, res, table))
                self.worker.error.connect(lambda err: self.handle_finished_shared(mode, None, err, table))
                self.worker.start()



            except Exception as e:
                self.show_error(f"Ошибка при выполнении расчётов: {str(e)}")

        def on_table_edited():
            self._invalidate_report_flag(mode)
            if hasattr(self, "report_buttons") and mode in self.report_buttons:
                self.report_buttons[mode].setEnabled(False)

        load_button.clicked.connect(load)
        clear_button.clicked.connect(clear)
        add_row_button.clicked.connect(add)
        process_button.clicked.connect(process)
        export_button.clicked.connect(lambda: self.export_table_to_excel_or_csv(table))
        tab_widget.keyPressEvent = lambda event: self.handle_key_event(event, table, label)
        info_button.clicked.connect(show_info)

        

        table.itemChanged.connect(on_table_edited)

        return tab_widget
    

    def _extract_df_from_table(self, table):
        data = []
        for row in range(table.rowCount()):
            row_data = []
            empty_row = True
            for col in range(len(DEFAULT_COLUMNS)):
                item = table.item(row, col)
                value = item.text() if item else ""
                if value.strip():
                    empty_row = False
                row_data.append(value.strip())
            if not empty_row:
                data.append(row_data)

        df = pd.DataFrame(data, columns=DEFAULT_COLUMNS)
        numeric_cols = [
            "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "Пористость (%)",
            "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
            "Температура (°C)", "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        df.dropna(subset=numeric_cols, inplace=True)
        for col in ["Толщина слоя (мкм)", "Площадь поверхности (м²/г)", "Площадь электрода (см²)", "ESR (Ом)"]:
            df = df[df[col] != 0]

        return df


    def highlight_invalid_cells(self, table, required_columns):
        """
        Подсвечивает:
        - Все пустые ячейки в непустых строках.
        - Все некорректные числовые ячейки (если столбец числовой).
        Возвращает True, если есть ошибки.
        """
        error_found = False

        for row in range(table.rowCount()):
            row_is_not_empty = any(
                table.item(row, col) and table.item(row, col).text().strip()
                for col in range(table.columnCount())
            )

            if not row_is_not_empty:
                # Пропускаем полностью пустую строку
                continue

            for col_index, col_name in enumerate(DEFAULT_COLUMNS):
                item = table.item(row, col_index)
                text = item.text().strip() if item else ""

                if not item:
                    item = QTableWidgetItem()
                    table.setItem(row, col_index, item)

                if not text:
                    item.setBackground(QColor(255, 102, 102))  # Пустое поле
                    error_found = True
                    continue

                if col_name in required_columns:
                    try:
                        float(text)
                    except:
                        item.setBackground(QColor(255, 102, 102))  # Некорректное число
                        error_found = True

        return error_found



    def create_optimization_tab(self):
        tab_widget = QTabWidget()
        # ======= Таб "Таблица" =======
        table_tab = QWidget()
        layout = QHBoxLayout()

        left_panel = QVBoxLayout()

        info_button = QPushButton()
        info_button.setIcon(QIcon("gui/icons/bulb.svg"))  # относительный путь
        info_button.setIconSize(QSize(15, 15))
        info_button.setFixedSize(25, 25)
        info_button.setStyleSheet("border: none; color: white;")
        info_button.setFixedWidth(25)
        info_button.setToolTip("Инструкция по вводу ограничений")
        left_panel.addWidget(info_button)

        self.goal_combo = QComboBox()
        self.goal_combo.addItems(list(GOAL_MAPPING.keys()))
        self.secondary_checkbox = QCheckBox("Учитывать сопутствующие характеристики")
        self.secondary_checkbox.setChecked(True)

        goal_layout = QVBoxLayout()
        goal_layout.addWidget(QLabel("Цель оптимизации:"))
        goal_layout.addWidget(self.goal_combo)
        goal_layout.addWidget(self.secondary_checkbox)
        goal_layout.addWidget(QLabel("Ограничения по параметрам:"))

        
        self.constraint_inputs = {}
        form_layout = QFormLayout()
        for col in DEFAULT_COLUMNS:
            input_field = QLineEdit()
            input_field.setMaximumWidth(150)
            self.constraint_inputs[col] = input_field
            form_layout.addRow(col, input_field)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_widget.setLayout(form_layout)
        scroll.setWidget(form_widget)

        left_panel.addLayout(goal_layout)
        left_panel.addWidget(scroll)

        bottom_buttons = QHBoxLayout()
    
        load_button = QPushButton("Импорт")
        clear_button = QPushButton("Очистить")
        add_row_button = QPushButton("Новая строка")
        process_button = QPushButton("Рассчитать")
        export_button = QPushButton("Экспорт")

    
    
        bottom_buttons.addWidget(load_button)
        bottom_buttons.addWidget(export_button)
        bottom_buttons.addWidget(clear_button)
        bottom_buttons.addWidget(add_row_button)
        bottom_buttons.addWidget(process_button)
        left_panel.addLayout(bottom_buttons)

        report_button = QPushButton("Отчет (PDF)")
        report_button.clicked.connect(lambda: self.generate_report_for_optimization("optimization"))
        report_button.setEnabled(False)
        self.report_buttons["optimization"] = report_button
        bottom_buttons.addWidget(report_button)

        right_panel = QVBoxLayout()
        self.optim_table = QTableWidget()
        self.handle_table_edit_operations(self.optim_table)

        self.optim_table.setColumnCount(len(DEFAULT_COLUMNS))
        self.optim_table.setHorizontalHeaderLabels(DEFAULT_COLUMNS)
        self.optim_table.setRowCount(10)
        right_panel.addWidget(self.optim_table)

        # Создаем левую панель как виджет с фиксированной шириной
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_widget.setMaximumWidth(700)  # Подберите подходящее значение

        # Добавляем в layout с разными stretch-факторами
        layout.addWidget(left_panel_widget, 1)  # Левая панель — 25% ширины
        layout.addLayout(right_panel, 2)        # Таблица — 75% ширины
        table_tab.setLayout(layout)

        # ======= Таб "Графики" =======
        graphics_tab = QWidget()
        graphics_layout = QVBoxLayout()
        graphics_label = QLabel("Графики появятся после расчёта")
        graphics_layout.addWidget(graphics_label)
        graphics_tab.setLayout(graphics_layout)

        # Добавляем в QTabWidget
        tab_widget.addTab(table_tab, "Таблица")
        tab_widget.addTab(graphics_tab, "Графики")

        # Сохраняем графическую вкладку для дальнейшего обновления
        if not hasattr(self, "graphics_tabs"):
            self.graphics_tabs = {}
        self.graphics_tabs["optimization"] = graphics_tab

        def load():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Импорт",
                "",
                "Файлы Excel/CSV (*.xlsx *.xls *.csv);"
            )
            if file_path:
                try:
                    if file_path.endswith(".csv"):
                        df = pd.read_csv(file_path, encoding="utf-8-sig")
                    else:
                        df = pd.read_excel(file_path)

                    df = df[DEFAULT_COLUMNS]
                    self.display_table(df, self.optim_table)
                except Exception as e:
                    self.show_error(str(e))

        def clear():
            self.optim_table.clearContents()
            self.optim_table.setRowCount(10)
            self.optim_table.setHorizontalHeaderLabels(DEFAULT_COLUMNS)

             # Очистка графиков
            graphics_widget = self.graphics_tabs.get("optimization")
            if graphics_widget:
                for i in reversed(range(graphics_widget.layout().count())):
                    item = graphics_widget.layout().itemAt(i)
                    if item:
                        w = item.widget()
                        if w:
                            w.setParent(None)
                if "optimization" in self.scale_labels:
                    label = self.scale_labels.pop("optimization")
                    label.setParent(None)

                if hasattr(self, "scale_slider_optimization"):
                    slider = getattr(self, "scale_slider_optimization")
                    slider.setParent(None)
                    delattr(self, "scale_slider_optimization")
                graphics_widget.layout().addWidget(QLabel("Графики появятся после расчёта"))

        def add():
            row = self.optim_table.rowCount()
            self.optim_table.insertRow(row)
            for j in range(self.optim_table.columnCount()):
                self.optim_table.setItem(row, j, QTableWidgetItem(""))

        def process():
            try:
                has_data = any(self.optim_table.item(r, c) and self.optim_table.item(r, c).text().strip()
                            for r in range(self.optim_table.rowCount())
                            for c in range(self.optim_table.columnCount()))
                if not has_data:
                    self.show_error("Таблица пуста. Введите данные перед выполнением расчёта.")
                    return

                required_numeric = [
                    "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "Пористость (%)",
                    "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
                    "Температура (°C)", "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
                ]
                if self.highlight_invalid_cells(self.optim_table, required_numeric):
                    self.show_error("Некоторые числовые ячейки пусты или содержат ошибки.")
                    return

                df_input = self._extract_df_from_table(self.optim_table)
                if df_input.empty:
                    self.show_error("Нет корректных данных для оптимизации.")
                    return

                self.progress_dialog = ProgressDialog("Оптимизация в процессе...", self)
                self.progress_dialog.show()

                goal = GOAL_MAPPING.get(self.goal_combo.currentText())
                constraints = {}
                invalid_inputs = []
                numeric_constraint_cols = required_numeric + ["PSD"]

                for col, widget in self.constraint_inputs.items():
                    parsed = self.parse_constraint(widget.text(), col)
                    if col in numeric_constraint_cols:
                        def is_numeric(val):
                            if isinstance(val, (float, int)):
                                return True
                            if isinstance(val, (list, tuple)):
                                return all(isinstance(v, (float, int)) for v in val)
                            return False

                        if parsed is not None and not is_numeric(parsed):
                            invalid_inputs.append(f"• {col}: {widget.text()}")
                    constraints[col] = parsed

                if invalid_inputs:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setWindowTitle("Некорректные ограничения")
                    msg.setText("Обнаружены некорректные значения в числовых ограничениях.")
                    msg.setInformativeText("Проверьте следующие поля:\n" + "\n".join(invalid_inputs))
                    msg.exec()
                    self.progress_dialog.close()
                    return

                include_secondary = self.secondary_checkbox.isChecked()

                self.opt_thread = OptimizationWorkerThread(df_input, goal, constraints, include_secondary)
                self.opt_thread.finished.connect(self.handle_finished)
                self.opt_thread.start()

            except Exception as e:
                self.show_error(f"Ошибка при запуске оптимизации: {str(e)}")

        def show_info():
            message = ("<b>Инструкция по модулю оптимизации:</b><br><br>"
                       "<u>Цель:</u> максимизация одного из параметров: удельной ёмкости, срока службы или эффективности хранения.<br><br>"
                       "<u>Результат:</u> первая строка - лучшее значение.<br><br>"
                       "<u>Формат ввода ограничений:</u><br>"
                       "Можно задавать числовые или строковые ограничения. Поддерживаются следующие форматы:<br><br>"
                       "• <b>12.5</b> — фиксированное значение<br>"
                       "• <b>10-30</b> — интервал от 10 до 30<br>"
                       "• <b>(10, 30)</b> — то же самое, альтернативный формат<br>"
                       "• <b>10, 20, 30</b> — список чисел<br><br>"
                       "<u>Особенность для <b>Строковых</b> полей:</u><br>"
                       "В отличие от других, здесь можно указать <u>список строк</u> через точку с запятой:<br>"
                       "• <b>N,S,B; S; нет</b> → будет интерпретировано как список: ['N,S,B','S','нет']<br>"
                       "• <b>N,S,B или N;S;B</b> → будет интерпретировано как список: ['N','S','B']<br><br>"
                       "<u>Примеры по конкретным полям:</u><br>"
                       "• <b>Площадь поверхности (м²/г):</b> 100-300<br>"
                       "• <b>ESR (Ом):</b> 0.01, 0.05, 0.1<br>"
                       "• <b>Гетероатомы:</b> N,S,B; S; -; нет<br>"
                       "• <b>Тип материала:</b> Carbon<br>"
                       "• <b>Тип электролита:</b> H2SO4<br><br>"
                       "<i>Важно: пробелы влияют на распознавание.</i>")

            msg = QMessageBox()
            msg.setWindowTitle("Инструкция")
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(message)
            msg.exec()

        def on_table_edited():
            self._invalidate_report_flag("optimization")
            if hasattr(self, "report_buttons") and "optimization" in self.report_buttons:
                self.report_buttons["optimization"].setEnabled(False)

        self.optim_table.itemChanged.connect(on_table_edited)


        info_button.clicked.connect(show_info)
        load_button.clicked.connect(load)
        clear_button.clicked.connect(clear)
        add_row_button.clicked.connect(add)
        process_button.clicked.connect(process)
        export_button.clicked.connect(lambda: self.export_table_to_excel_or_csv(self.optim_table))
        return tab_widget

    def generate_report_for_prediction(self, mode="prediction"):
        if not self.calculations_done.get(mode):
            self.show_error("Сначала выполните расчёты перед формированием отчёта.")
            return

        table = self.tables[mode]
        df_input = self._extract_df_from_table(table)
        df_results = self.df_results_storage
        graph_dir = os.path.join("visualization", "plots", "models")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PDF отчёт", "prediction_report.pdf", "PDF файлы (*.pdf)"
        )
        if not save_path:
            return

        self.progress_dialog = ProgressDialog("Формирование PDF отчёта...", self)
        self.progress_dialog.show()
        QApplication.processEvents()

        self.prediction_report_thread = PredictionReportThread(df_input, df_results, graph_dir, save_path)
        self.prediction_report_thread.finished.connect(self.on_report_success)
        self.prediction_report_thread.error.connect(self.on_report_error)
        self.prediction_report_thread.start()

    def generate_report_for_optimization(self, mode="optimization"):
        if not self.calculations_done.get(mode):
            self.show_error("Сначала выполните расчёты перед формированием отчёта.")
            return

        table = self.optim_table
        df_input = self._extract_df_from_table(table)
        df_results = self.df_results_storage
        graph_dir = os.path.join("visualization", "plots", "optimization", "graphics")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PDF отчёт", "optimization_report.pdf", "PDF файлы (*.pdf)"
        )
        if not save_path:
            return

     
        goal_text = self.goal_combo.currentText()
        constraints = {
            k: v.text().strip()
            for k, v in self.constraint_inputs.items()
            if v.text().strip()
        }
        include_secondary = self.secondary_checkbox.isChecked()

        self.progress_dialog = ProgressDialog("Формирование PDF отчёта...", self)
        self.progress_dialog.show()
        QApplication.processEvents()

        
        try:
            self.optimization_report_thread = OptimizationReportThread(
                df_input, df_results, graph_dir, save_path,
                goal_text, constraints, include_secondary
            )
            self.optimization_report_thread.finished.connect(self.on_report_success)
            self.optimization_report_thread.error.connect(self.on_report_error)
            self.optimization_report_thread.start()
        except Exception as e:
            self.progress_dialog.close()
            self.show_error(f"Ошибка при формировании отчёта: {str(e)}")



    def _run_process(self, mode, table, label, progress):
        try:
            data = []
            for row in range(table.rowCount()):
                row_data = []
                empty_row = True
                for col in range(len(DEFAULT_COLUMNS)):
                    item = table.item(row, col)
                    value = item.text() if item else ""
                    if value.strip():
                        empty_row = False
                    row_data.append(value.strip())
                if not empty_row:
                    data.append(row_data)

            df_input = pd.DataFrame(data, columns=DEFAULT_COLUMNS)
            numeric_cols = [
                "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "Пористость (%)",
                "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)",
                "Температура (°C)", "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"
            ]
            for col in numeric_cols:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            df_input.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
            df_input.dropna(subset=numeric_cols, inplace=True)
            for col in ["Толщина слоя (мкм)", "Площадь поверхности (м²/г)", "Площадь электрода (см²)", "ESR (Ом)"]:
                df_input = df_input[df_input[col] != 0]

            if df_input.empty:
                raise ValueError("Недостаточно корректных данных после фильтрации.")

            df_results = calculate_all(df_input)

            if mode == "physics":
                self.display_table(df_results, table)

            elif mode == "prediction":
                enough = is_enough_data_for_prediction(df_results, MIN_REQUIRED_ROWS)
                if enough or ALLOW_SMALL_DATA:
                    if not enough:
                        print("⚠️ Недостаточно данных для точного предсказания")
                    df_results = predict_on_raw_data(df_input, df_results)
                    ml_cols = [col for col in df_results.columns if "ML" in col]
                    combined = pd.concat([df_input.reset_index(drop=True), df_results[ml_cols].reset_index(drop=True)], axis=1)
                    self.display_table(combined, table)
                else:
                    clear_ml_outputs()
                    self.show_error("Недостаточно данных для предсказания. Выполнены только физические расчёты.")

            elif mode == "optimization":
                goal_rus = self.goal_combo.currentText()
                goal = GOAL_MAPPING.get(goal_rus)
                if not goal:
                    raise ValueError("Не удалось распознать цель оптимизации.")
                constraints = {}
                invalid_inputs = []
                numeric_constraint_cols = [
                    "Площадь поверхности (м²/г)", "Размер пор (нм)", "ID/IG", "Толщина слоя (мкм)", "PSD", "Пористость (%)",
                    "Уд. поверхность (м²/см³)", "Концентрация (моль/л)", "Напряжение (В)", "Ток (А)", "Температура (°C)",
                    "Скорость скан. (В/с)", "ESR (Ом)", "Циклы", "Площадь электрода (см²)"  # ❗ Исключили "Диапазон EIS (Гц)"
                ]

                for col, widget in self.constraint_inputs.items():
                    parsed = self.parse_constraint(widget.text(), col)
                    if col in numeric_constraint_cols:
                        def is_numeric(val):
                            if isinstance(val, (float, int)):
                                return True
                            if isinstance(val, (list, tuple)):
                                return all(isinstance(v, (float, int)) for v in val)
                            return False

                        if parsed is not None and not is_numeric(parsed):
                            invalid_inputs.append(f"• {col}: {widget.text()}")
                    constraints[col] = parsed

                if invalid_inputs:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setWindowTitle("Некорректные ограничения")
                    msg.setText("Обнаружены некорректные значения в числовых ограничениях.")
                    msg.setInformativeText("Проверьте следующие поля:\n" + "\n".join(invalid_inputs))
                    msg.exec()
                    progress.close()
                    return

                include_secondary = self.secondary_checkbox.isChecked()
                df_opt = optimize_parameters(
                    df_start=df_input,
                    optimization_goal=goal,
                    custom_constraints=constraints,
                    include_secondary_metrics=include_secondary
                )
                self.display_table(df_opt, table)
                self.visual(df_opt, "visualization/plots/optimization")

                


        except Exception as e:
            self.show_error(str(e))
        finally:
            progress.close()





    def handle_key_event(self, event, table, label=None):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected_row = table.currentRow()
            if selected_row != -1 and table.selectionModel().isRowSelected(selected_row, table.rootIndex()):
                row_count = table.rowCount()
                table.blockSignals(True)
                for i in range(selected_row, row_count - 1):
                    for j in range(table.columnCount()):
                        next_item = table.item(i + 1, j)
                        text = next_item.text() if next_item else ""
                        table.setItem(i, j, QTableWidgetItem(text))
                table.removeRow(row_count - 1)
                table.blockSignals(False)
                if label:
                    label.setText(f"Удалена строка {selected_row + 1}")


    def display_table(self, df, table):
        table.clear()
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = str(df.iat[i, j])
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
        

    def show_error(self, message):
        detailed = ""

        if "not in index" in message:
            detailed = (
                "Одна или несколько колонок не найдены в таблице. "
                "Проверьте, что в файле Excel есть все обязательные столбцы:\n\n"
                + "\n".join(f"• {col}" for col in DEFAULT_COLUMNS)
            )
        elif "could not convert" in message or "could not convert string" in message:
            detailed = (
                "Некоторые значения невозможно преобразовать в числа. "
                "Проверьте, чтобы в числовых колонках не было текста или символов.\n\n"
                "Пример ошибки: 'ввв' в колонке 'Площадь поверхности (м²/г)'"
            )
        elif "float" in message and "argument must be a string" in message:
            detailed = (
                "Ожидалось числовое значение, но получено пустое или текстовое поле.\n"
                "Проверьте пустые ячейки и типы данных."
            )
        elif "NaN" in message or "dropna" in message:
            detailed = (
                "После очистки от некорректных значений таблица стала пустой.\n"
                "Проверьте, чтобы в числовых колонках были корректные значения."
            )

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Ошибка")
        msg.setText("Ошибка при обработке данных")
        msg.setInformativeText(message)
        if detailed:
            msg.setDetailedText(detailed)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()


def format_value(val, digits=4):
    if pd.isna(val):
        return "–"
    try:
        val_float = float(val)
        return f"{val_float:.{digits}g}"
    except (ValueError, TypeError):
        return str(val)


def generate_physics_report(df_input, df_results, graph_dir, output_path="physics_report.pdf"):
    from fpdf import FPDF
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt

    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", size=8)
            self.cell(0, 10, f"Страница {self.page_no()} из {{nb}}", align='C')

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=14)

    # --- Логотип ---
    logo_path = "gui/logo.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=160, y=10, w=30)
    pdf.cell(0, 10, txt="Отчёт по физическому моделированию", ln=True, align="C")
    pdf.ln(5)

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.set_font("DejaVu", size=10)
    pdf.cell(0, 8, txt=f"Дата генерации отчёта: {now}", ln=True)
    pdf.cell(0, 8, txt=f"Количество строк: {len(df_results)}", ln=True)
    pdf.ln(3)
    pdf.multi_cell(0, 6, "Отчёт содержит анализ параметров модели суперконденсаторов с статистикой и визуализациями.")
    pdf.ln(5)

    # --- Таблица статистики ---
    calculated_cols = [col for col in df_results.columns if col not in df_input.columns and pd.api.types.is_numeric_dtype(df_results[col])]
    if calculated_cols:
        pdf.set_font("DejaVu", size=9)
        pdf.cell(70, 8, "Показатель", border=1)
        pdf.cell(30, 8, "Среднее", border=1)
        pdf.cell(30, 8, "Стд", border=1)
        pdf.cell(30, 8, "Мин", border=1)
        pdf.cell(30, 8, "Макс", border=1)
        pdf.ln()
        for col in calculated_cols:
            col_data = df_results[col].dropna()
            if not col_data.empty:
                pdf.set_font("DejaVu", size=8)
                pdf.cell(70, 6, col[:50], border=1)
                pdf.cell(30, 6, format_value(col_data.mean()), border=1)
                pdf.cell(30, 6, format_value(col_data.std()), border=1)
                pdf.cell(30, 6, format_value(col_data.min()), border=1)
                pdf.cell(30, 6, format_value(col_data.max()), border=1)
                pdf.ln()

    # --- Подпись ---
    pdf.ln(10)
    pdf.set_font("DejaVu", size=10)
    pdf.cell(0, 10, txt="Автор отчёта: Кухмистров Игорь, студент гр. ВПР 41", ln=True)

    # --- Автоматическая гистограмма входных параметров ---
    try:
        numeric_input_cols = [col for col in df_input.columns if pd.api.types.is_numeric_dtype(df_input[col])]
        if numeric_input_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            df_input[numeric_input_cols].hist(ax=ax, bins=15, edgecolor='black')
            plt.tight_layout()
            input_hist_path = "auto_input_hist.png"
            plt.savefig(input_hist_path)
            plt.close()
            pdf.add_page()
            pdf.set_font("DejaVu", size=11)
            pdf.cell(0, 10, txt="Сводная гистограмма входных параметров", ln=True)
            pdf.image(input_hist_path, x=10, y=25, w=190)
            os.remove(input_hist_path)
    except Exception as e:
        pdf.cell(0, 10, txt=f"[Ошибка генерации гистограммы входных данных: {e}]", ln=True)

    # --- Графики результатов ---
    if os.path.exists(graph_dir):
        image_files = [f for f in sorted(os.listdir(graph_dir)) if f.lower().endswith(".png")]
        for idx, fname in enumerate(image_files):
            img_path = os.path.join(graph_dir, fname)
            title = os.path.splitext(fname)[0]
            pdf.add_page()
            pdf.set_font("DejaVu", size=11)
            pdf.cell(0, 10, txt=f"График {idx + 1}: {title}", ln=True)
            try:
                pdf.image(img_path, x=10, y=25, w=190)
            except:
                pdf.cell(0, 10, txt=f"[Ошибка загрузки {fname}]", ln=True)

    pdf.output(output_path)

    


def generate_prediction_report(df_input, df_results, graph_dir, output_path="prediction_report.pdf"):
        from fpdf import FPDF
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os

        class PDF(FPDF):
            def footer(self):
                self.set_y(-15)
                self.set_font("DejaVu", size=8)
                self.cell(0, 10, f"Страница {self.page_no()} из {{nb}}", align='C')

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        font_path = os.path.join("fonts", "DejaVuSans.ttf")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=14)

        logo_path = "gui/logo.png"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=160, y=10, w=30)
        pdf.cell(0, 10, txt="Отчёт по предсказанию параметров", ln=True, align="C")
        pdf.ln(5)

        now = datetime.now().strftime("%d.%m.%Y %H:%M")
        pdf.set_font("DejaVu", size=10)
        pdf.cell(0, 8, txt=f"Дата генерации отчёта: {now}", ln=True)
        pdf.cell(0, 8, txt=f"Количество строк: {len(df_results)}", ln=True)
        pdf.ln(3)
        pdf.multi_cell(0, 6, "Отчёт содержит результаты машинного предсказания характеристик суперконденсаторов и визуализации.")

        # Таблица предсказанных значений
        pred_cols = [col for col in df_results.columns if "ML" in col]
        if pred_cols:
            pdf.ln(5)
            pdf.set_font("DejaVu", size=9)
            pdf.cell(80, 8, "Параметр", border=1)
            pdf.cell(30, 8, "Среднее", border=1)
            pdf.cell(30, 8, "Стд", border=1)
            pdf.cell(30, 8, "Мин", border=1)
            pdf.cell(30, 8, "Макс", border=1)
            pdf.ln()

            for col in pred_cols:
                col_data = df_results[col].dropna()
                if not col_data.empty:
                    pdf.set_font("DejaVu", size=8)
                    pdf.cell(80, 6, col[:50], border=1)
                    pdf.cell(30, 6, format_value(col_data.mean()), border=1)
                    pdf.cell(30, 6, format_value(col_data.std()), border=1)
                    pdf.cell(30, 6, format_value(col_data.min()), border=1)
                    pdf.cell(30, 6, format_value(col_data.max()), border=1)
                    pdf.ln()

        # Графики из папки models
        if os.path.exists(graph_dir):
            image_files = [f for f in sorted(os.listdir(graph_dir)) if f.lower().endswith(".png")]
            for idx, fname in enumerate(image_files):
                img_path = os.path.join(graph_dir, fname)
                title = os.path.splitext(fname)[0]
                pdf.add_page()
                pdf.set_font("DejaVu", size=11)
                pdf.cell(0, 10, txt=f"График {idx + 1}: {title}", ln=True)
                try:
                    pdf.image(img_path, x=10, y=25, w=190)
                except:
                    pdf.cell(0, 10, txt=f"[Ошибка загрузки {fname}]", ln=True)

        pdf.output(output_path)

def generate_optimization_report(df_input, df_results, graph_dir, output_path="optimization_report.pdf", goal_text=None, constraints=None, include_secondary=False):

    

    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", size=8)
            self.cell(0, 10, f"Страница {self.page_no()} из {{nb}}", align='C')

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=14)

    logo_path = "gui/logo.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=160, y=10, w=30)
    pdf.cell(0, 10, txt="Отчёт по оптимизации параметров", ln=True, align="C")
    pdf.ln(5)

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.set_font("DejaVu", size=10)
    pdf.cell(0, 8, txt=f"Дата генерации отчёта: {now}", ln=True)
    pdf.cell(0, 8, txt=f"Количество строк: {len(df_results)}", ln=True)
    pdf.ln(3)
    pdf.multi_cell(0, 6, "В отчёте представлены результаты оптимизации параметров суперконденсатора с учетом выбранной цели и ограничений.")
    pdf.ln(5)

    pdf.cell(0, 8, txt=f"Цель оптимизации: {goal_text}", ln=True)
    pdf.cell(0, 8, txt=f"Учитывать сопутствующие характеристики: {'Да' if include_secondary else 'Нет'}", ln=True)
    pdf.ln(3)

    if constraints:
        pdf.set_font("DejaVu", size=9)
        pdf.cell(0, 8, txt="Указанные ограничения:", ln=True)
        for key, val in constraints.items():
            if val is not None and val != "":
                text_val = str(val)
                pdf.set_font("DejaVu", size=8)
                pdf.multi_cell(0, 6, f"• {key}: {text_val}")
        pdf.ln(3)

    # Таблица лучших значений
    if not df_results.empty:
        pdf.set_font("DejaVu", size=9)
        pdf.cell(80, 8, "Параметр", border=1)
        pdf.cell(40, 8, "Значение", border=1)
        pdf.ln()
        best_row = df_results.iloc[0]
        for col in df_results.columns:
            val = best_row[col]
            pdf.set_font("DejaVu", size=8)
            pdf.cell(80, 6, col[:50], border=1)
            pdf.cell(40, 6, format_value(val), border=1)
            pdf.ln()

    # Графики из папки
    if os.path.exists(graph_dir):
        image_files = [f for f in sorted(os.listdir(graph_dir)) if f.lower().endswith(".png")]
        for idx, fname in enumerate(image_files):
            img_path = os.path.join(graph_dir, fname)
            title = os.path.splitext(fname)[0]
            pdf.add_page()
            pdf.set_font("DejaVu", size=11)
            pdf.cell(0, 10, txt=f"График {idx + 1}: {title}", ln=True)
            try:
                pdf.image(img_path, x=10, y=25, w=190)
            except:
                pdf.cell(0, 10, txt=f"[Ошибка загрузки {fname}]", ln=True)

    pdf.output(output_path)

        
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualization/plots/models", exist_ok=True)
    app = QApplication(sys.argv)

    # Загружаем стиль
    with open("gui/style.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())

    app.setWindowIcon(QIcon("gui/logo.png"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
