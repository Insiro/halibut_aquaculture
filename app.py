import sys
from os import path
import cv2
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import numpy as np
from service import get_total_number, detection, TARGET_CLASSES


class MyApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.fig = None
        self.list_view = None
        self.image_label = None
        self.init_ui()

    def right_side(self):
        vbox = QtWidgets.QVBoxLayout()
        openfile = QtWidgets.QAction("Open", self)
        openfile.setShortcut("crtl+o")
        openfile.setStatusTip("select directory")
        input_btn = QtWidgets.QPushButton("select directory for datas")
        input_btn.clicked.connect(self.showDialog)
        openfile.triggered.connect(self.showDialog)
        vbox.addWidget(input_btn)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setText("Image Area")
        vbox.addWidget(self.image_label)
        return vbox

    def left_side(self):
        vbox = QtWidgets.QVBoxLayout()
        self.list_view = QtWidgets.QListWidget()
        self.list_view.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        self.list_view.currentItemChanged.connect(self.itemClicked)
        self.mount_view = QtWidgets.QListWidget()
        self.mount_view.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.fig = plt.figure(figsize=(3, 4))
        self.fig.tight_layout()
        self.fig.set_dpi(100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setVisible(True)
        vbox.addWidget(self.list_view)
        vbox.addWidget(self.mount_view)
        vbox.addWidget(self.canvas)
        return vbox

    def init_ui(self):
        h_box = QtWidgets.QHBoxLayout()
        self.setLayout(h_box)
        h_box.addLayout(self.left_side())
        h_box.addLayout(self.right_side())

        self.setWindowTitle("analysis")
        self.move(300, 300)
        self.show()

    def itemClicked(self, item):
        base_dir = self.save_info["save_dir"]

        img = cv2.imread(path.join(base_dir, item.text()))
        self.update_img(img)

    def showDialog(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Directory", "./dataset/validation-data/images/val"
        )
        ret, self.save_info = detection(source=dir_name, nosave=False)
        _, class_amounts = np.array(get_total_number(ret, 100))

        self.list_view.clear()
        for fname in self.save_info["files"]:
            self.list_view.addItem(fname)
        self.list_view.update()

        self.mount_view.clear()
        for idx in range(len(TARGET_CLASSES)):
            self.mount_view.addItem(f"{TARGET_CLASSES[idx]}\t{class_amounts[idx]}")
        self.mount_view.update()
        # display original image defore labeling
        # plotting
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.pie(class_amounts, labels=TARGET_CLASSES, labeldistance=None)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.7))
        self.canvas.draw()

        base_dir = self.save_info["save_dir"]
        img = cv2.imread(path.join(base_dir, fname))
        self.update_img(img)

    def update_img(self, np_hsv):
        img = cv2.cvtColor(np_hsv, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pxmap = QtGui.QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pxmap)
        self.image_label.update()
        pass


def start_gui(argv):
    app = QtWidgets.QApplication(argv)
    ex = MyApp()
    sys.exit(app.exec_())


