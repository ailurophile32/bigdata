import sys
import traceback

try:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    # 导入主程序
    from spectra_viewer import SpectralViewer
    
    # 创建并显示主窗口
    window = SpectralViewer()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())
except Exception as e:
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {str(e)}\n\n")
        f.write(traceback.format_exc())
    print(f"Error occurred: {str(e)}")
    print("Full traceback written to error_log.txt")
