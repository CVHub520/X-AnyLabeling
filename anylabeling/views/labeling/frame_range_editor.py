"""Frame Range Editor for X-AnyLabeling"""

import os
import json
import shutil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QFormLayout,
    QCheckBox,
)


class FrameRangeEditor(QDialog):
    """视频帧范围编辑器"""

    def __init__(self, parent):
        """初始化帧范围编辑器"""
        super().__init__(parent)
        self.parent = parent

    def get_selected_shapes(self):
        """获取当前选中的形状"""
        return self.parent.canvas.selected_shapes

    def shapes_are_identical(self, shape1, shape2, tolerance=1.0):
        """
        判断两个形状是否完全一致

        Args:
            shape1, shape2: 要比较的两个形状对象
            tolerance: 坐标点比较的容差值

        Returns:
            bool: 如果两个形状完全一致返回True，否则返回False
        """
        # 比较标签
        if shape1.label != shape2.label:
            return False

        # 比较形状类型
        if shape1.shape_type != shape2.shape_type:
            return False

        # 比较点坐标（考虑容差）
        if len(shape1.points) != len(shape2.points):
            return False

        for p1, p2 in zip(shape1.points, shape2.points):
            if abs(p1.x() - p2.x()) > tolerance or abs(p1.y() - p2.y()) > tolerance:
                return False

        return True

    def show_frame_range_dialog(self):
        """显示帧范围输入对话框"""
        dialog = FrameRangeDialog(self.parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            start_frame = dialog.start_frame
            end_frame = dialog.end_frame
            operation_type = dialog.operation_type
            backup_enabled = dialog.backup_checkbox.isChecked()
            self.process_frame_range(start_frame, end_frame, operation_type, backup_enabled)

    def update_file_list_checkbox(self, frame_num, checked=False):
        """更新文件列表中指定帧的复选框状态"""
        # 获取当前目录
        current_dir = os.path.dirname(self.parent.filename)
        
        # 构造完整路径的帧文件名（只使用jpg格式）
        frame_name = f"frame_{frame_num:05d}.jpg"
        frame_path = os.path.join(current_dir, frame_name)
        
        # 查找文件列表中的项
        items = self.parent.file_list_widget.findItems(
            frame_path, Qt.MatchExactly
        )
        
        if items:
            item = items[0]
            state = Qt.Checked if checked else Qt.Unchecked
            item.setCheckState(state)
            # 强制更新UI
            self.parent.file_list_widget.repaint()
            # 处理所有待处理的事件以确保UI更新
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        else:
            # 如果对应的jpg文件不存在，显示警告
            QMessageBox.warning(
                self.parent,
                self.tr("File Not Found"),
                self.tr("Image file {frame_name} not found in the file list.").format(frame_name=frame_path)
            )

    def delete_frames_in_range(self, start_frame, end_frame, backup_enabled=True):
        """删除指定帧范围内的所有标注数据（直接删除JSON文件）"""
        if not self.parent.filename:
            return

        # 获取当前目录
        current_dir = os.path.dirname(self.parent.filename)

        # 创建备份目录（如果需要）
        backup_dir = None
        if backup_enabled:
            backup_dir = os.path.join(current_dir, "_backup")
            os.makedirs(backup_dir, exist_ok=True)

        deleted_count = 0
        error_count = 0

        # 遍历指定范围内的帧
        for frame_num in range(start_frame, end_frame + 1):
            # 构造帧文件名
            frame_name = f"frame_{frame_num:05d}"
            json_file = os.path.join(current_dir, f"{frame_name}.json")

            # 检查文件是否存在
            if os.path.exists(json_file):
                try:
                    # 如果需要备份，则先复制到备份目录
                    if backup_enabled:
                        backup_file = os.path.join(backup_dir, f"{frame_name}.json")
                        shutil.copy2(json_file, backup_file)

                    # 删除文件
                    os.remove(json_file)
                    deleted_count += 1
                    
                    # 更新UI中该帧的复选框状态为未选中
                    self.update_file_list_checkbox(frame_num, checked=False)
                except Exception as e:
                    print(f"Error deleting file {json_file}: {e}")
                    error_count += 1

        # 显示结果
        message = self.tr("Successfully deleted {deleted_count} annotation files").format(deleted_count=deleted_count)
        if error_count > 0:
            message += "\n" + self.tr("Failed to delete {error_count} files").format(error_count=error_count)

        QMessageBox.information(self.parent, self.tr("Operation completed"), message)

    def remove_selected_shapes_in_range(self, start_frame, end_frame, backup_enabled=True):
        """在指定帧范围内移除与选中对象完全一致的标注数据"""
        selected_shapes = self.get_selected_shapes()
        if not selected_shapes:
            QMessageBox.warning(self.parent, self.tr("Warning"), self.tr("Please select the object to remove first"))
            return

        if not self.parent.filename:
            return

        # 获取当前目录
        current_dir = os.path.dirname(self.parent.filename)

        # 创建备份目录（如果需要）
        backup_dir = None
        if backup_enabled:
            backup_dir = os.path.join(current_dir, "_backup")
            os.makedirs(backup_dir, exist_ok=True)

        removed_count = 0
        deleted_files_count = 0
        error_count = 0
        processed_files = 0

        # 遍历指定范围内的帧
        for frame_num in range(start_frame, end_frame + 1):
            # 构造帧文件名
            frame_name = f"frame_{frame_num:05d}"
            json_file = os.path.join(current_dir, f"{frame_name}.json")

            # 检查文件是否存在
            if os.path.exists(json_file):
                try:
                    # 如果需要备份，则先复制到备份目录
                    if backup_enabled:
                        backup_file = os.path.join(backup_dir, f"{frame_name}.json")
                        shutil.copy2(json_file, backup_file)

                    # 读取JSON文件
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 获取所有形状
                    all_shapes = data.get('shapes', [])
                    
                    # 如果没有形状，直接删除文件
                    if not all_shapes:
                        os.remove(json_file)
                        # 更新UI中该帧的复选框状态为未选中
                        self.update_file_list_checkbox(frame_num, checked=False)
                        deleted_files_count += 1
                        processed_files += 1
                        continue

                    # 筛选出将要被删除的形状
                    shapes_to_remove = []
                    for shape_data in all_shapes:
                        # 将字典转换为Shape对象进行比较
                        from ..labeling.shape import Shape
                        shape = Shape().load_from_dict(shape_data, close=False)

                        # 检查是否与任何选中形状匹配
                        for selected_shape in selected_shapes:
                            if self.shapes_are_identical(shape, selected_shape):
                                shapes_to_remove.append(shape_data)
                                break

                    # 如果所有形状都将被删除，则直接删除文件
                    if len(shapes_to_remove) == len(all_shapes):
                        os.remove(json_file)
                        # 更新UI中该帧的复选框状态为未选中
                        self.update_file_list_checkbox(frame_num, checked=False)
                        deleted_files_count += 1
                    else:
                        # 过滤掉与选中形状完全一致的形状
                        filtered_shapes = [shape for shape in all_shapes if shape not in shapes_to_remove]
                        
                        # 更新移除的形状数量
                        removed_count += len(shapes_to_remove)

                        # 保存修改后的文件（仅在还有形状时保存）
                        data['shapes'] = filtered_shapes
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        # 更新UI中该帧的复选框状态为选中
                        self.update_file_list_checkbox(frame_num, checked=True)
                        
                    processed_files += 1

                except Exception as e:
                    print(f"Error processing file {json_file}: {e}")
                    error_count += 1

        # 显示结果
        message = self.tr("Successfully processed {processed_files} files").format(processed_files=processed_files)
        if removed_count > 0:
            message += "\n" + self.tr("Removed {removed_count} objects").format(removed_count=removed_count)
        if deleted_files_count > 0:
            message += "\n" + self.tr("Deleted {deleted_files_count} annotation files").format(deleted_files_count=deleted_files_count)
        if error_count > 0:
            message += "\n" + self.tr("Failed to process {error_count} files").format(error_count=error_count)

        QMessageBox.information(self.parent, self.tr("Operation completed"), message)

    def add_selected_shapes_to_range(self, start_frame, end_frame, backup_enabled=True):
        """在指定帧范围内添加与选中对象完全一致的标注数据"""
        selected_shapes = self.get_selected_shapes()
        if not selected_shapes:
            QMessageBox.warning(self.parent, self.tr("Warning"), self.tr("Please select the object to add first"))
            return

        if not self.parent.filename:
            return

        # 获取当前目录
        current_dir = os.path.dirname(self.parent.filename)

        # 创建备份目录（如果需要）
        backup_dir = None
        if backup_enabled:
            backup_dir = os.path.join(current_dir, "_backup")
            os.makedirs(backup_dir, exist_ok=True)

        added_count = 0
        created_files_count = 0
        error_count = 0
        processed_files = 0

        # 遍历指定范围内的帧
        for frame_num in range(start_frame, end_frame + 1):
            # 构造帧文件名
            frame_name = f"frame_{frame_num:05d}"
            json_file = os.path.join(current_dir, f"{frame_name}.json")
            image_file = os.path.join(current_dir, f"{frame_name}.jpg")

            # 检查文件是否存在
            image_file_exists = os.path.exists(image_file)
            if not image_file_exists:
                continue
            json_file_exists = os.path.exists(json_file)
            if not json_file_exists:
                # 如果文件不存在，创建一个带有基本结构的新文件
                # 尝试获取图像尺寸
                image_height = 0
                image_width = 0
                if os.path.exists(image_file):
                    try:
                        from PyQt5.QtGui import QImage
                        image = QImage(image_file)
                        if not image.isNull():
                            image_height = image.height()
                            image_width = image.width()
                    except Exception as e:
                        print(f"Error reading image dimensions: {e}")
                
                data = {
                    "version": "5.5.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": f"{frame_name}.jpg",
                    "imageData": None,
                    "imageHeight": image_height,
                    "imageWidth": image_width
                }
                created_files_count += 1
            else:
                try:
                    # 如果需要备份，则先复制到备份目录
                    if backup_enabled:
                        backup_file = os.path.join(backup_dir, f"{frame_name}.json")
                        shutil.copy2(json_file, backup_file)

                    # 读取JSON文件
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading file {json_file}: {e}")
                    error_count += 1
                    continue

            try:
                # 确保shapes字段存在
                if 'shapes' not in data:
                    data['shapes'] = []

                # 记录原始形状数量
                original_shape_count = len(data['shapes'])

                # 为每个选中的形状检查是否已存在，如果不存在则添加
                for selected_shape in selected_shapes:
                    # 检查是否已存在完全相同的形状
                    already_exists = False
                    for shape_data in data['shapes']:
                        # 将字典转换为Shape对象进行比较
                        from ..labeling.shape import Shape
                        shape = Shape().load_from_dict(shape_data, close=False)

                        if self.shapes_are_identical(shape, selected_shape):
                            already_exists = True
                            break

                    # 如果不存在，则添加
                    if not already_exists:
                        data['shapes'].append(selected_shape.to_dict())
                        added_count += 1

                # 保存修改后的文件
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # 更新UI中该帧的复选框状态为选中
                self.update_file_list_checkbox(frame_num, checked=True)

                processed_files += 1

            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                error_count += 1

        # 显示结果
        message = self.tr("Successfully processed {processed_files} files").format(processed_files=processed_files)
        if added_count > 0:
            message += "\n" + self.tr("Added {added_count} objects").format(added_count=added_count)
        if created_files_count > 0:
            message += "\n" + self.tr("Created {created_files_count} annotation files").format(created_files_count=created_files_count)
        if error_count > 0:
            message += "\n" + self.tr("Failed to process {error_count} files").format(error_count=error_count)

        QMessageBox.information(self.parent, self.tr("Operation completed"), message)

    def process_frame_range(self, start_frame, end_frame, operation_type, backup_enabled=True):
        """处理帧范围操作"""
        if operation_type == "delete_all":
            self.delete_frames_in_range(start_frame, end_frame, backup_enabled)
        elif operation_type == "remove_selected":
            self.remove_selected_shapes_in_range(start_frame, end_frame, backup_enabled)
        elif operation_type == "add_selected":
            self.add_selected_shapes_to_range(start_frame, end_frame, backup_enabled)


class FrameRangeDialog(QDialog):
    """帧范围输入对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("Frame Range Editor"))
        self.setModal(True)
        self.resize(300, 200)

        # 初始化变量
        self.start_frame = 0
        self.end_frame = 0
        self.operation_type = "delete_all"

        # 创建UI
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()

        # 表单布局
        form_layout = QFormLayout()

        # 起始帧输入
        self.start_frame_input = QLineEdit()
        self.start_frame_input.setText("0")
        form_layout.addRow(QLabel(self.tr("Start Frame:")), self.start_frame_input)

        # 结束帧输入
        self.end_frame_input = QLineEdit()
        self.end_frame_input.setText("10")
        form_layout.addRow(QLabel(self.tr("End Frame:")), self.end_frame_input)

        # 操作类型选择
        self.operation_combo = QComboBox()
        self.operation_combo.addItem(self.tr("Delete All Annotations"), "delete_all")
        self.operation_combo.addItem(self.tr("Remove Selected Objects"), "remove_selected")
        self.operation_combo.addItem(self.tr("Add Selected Objects"), "add_selected")
        form_layout.addRow(QLabel(self.tr("Operation Type:")), self.operation_combo)

        # 备份选项
        self.backup_checkbox = QCheckBox(self.tr("Create Backup"))
        self.backup_checkbox.setChecked(True)
        form_layout.addRow(QLabel(""), self.backup_checkbox)

        layout.addLayout(form_layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton(self.tr("OK"))
        self.cancel_button = QPushButton(self.tr("Cancel"))
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接信号
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)

        # 根据选中对象状态更新UI
        self.update_ui_state()

    def update_ui_state(self):
        """根据选中对象状态更新UI"""
        try:
            # 检查是否有选中的形状
            selected_shapes = self.parent.canvas.selected_shapes
            has_selection = len(selected_shapes) > 0

            # 如果没有选中对象，禁用移除和添加操作
            if not has_selection:
                # 禁用移除和添加选项
                for i in range(self.operation_combo.count()):
                    if self.operation_combo.itemData(i) in ["remove_selected", "add_selected"]:
                        self.operation_combo.model().item(i).setEnabled(False)
                # 默认选择删除所有
                self.operation_combo.setCurrentIndex(0)
            else:
                # 启用所有选项
                for i in range(self.operation_combo.count()):
                    self.operation_combo.model().item(i).setEnabled(True)
        except Exception as e:
            print(f"Error updating UI state: {e}")

    def on_operation_changed(self, index):
        """操作类型改变时的处理"""
        self.operation_type = self.operation_combo.itemData(index)

    def accept(self):
        """确认操作"""
        try:
            # 获取输入值
            self.start_frame = int(self.start_frame_input.text())
            self.end_frame = int(self.end_frame_input.text())
            self.operation_type = self.operation_combo.currentData()

            # 验证输入
            if self.start_frame < 0 or self.end_frame < 0:
                QMessageBox.warning(self, self.tr("Input Error"), self.tr("Frame number cannot be negative"))
                return

            if self.start_frame > self.end_frame:
                QMessageBox.warning(self, self.tr("Input Error"), self.tr("Start frame cannot be greater than end frame"))
                return

        except ValueError:
            QMessageBox.warning(self, self.tr("Input Error"), self.tr("Please enter a valid number"))
            return

        super().accept()