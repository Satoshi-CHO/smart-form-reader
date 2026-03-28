import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import cv2

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("900x750")
        
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.settings_path = os.path.join(self.base_dir, "settings.json")
        
        self.lang = "English"
        self.app_config = {}
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    self.app_config = json.load(f)
                    self.lang = self.app_config.get("language", "English")
            except:
                pass
                
        self.strings = {
            "English": {
                "title": "OCR Application",
                "lang_label": "Language: ",
                "settings_frame": "Settings / Paths",
                "input_dir": "Input Dir:",
                "output_dir": "Output Dir:",
                "template": "Template Image:",
                "use_gpu": "Use GPU Acceleration (if available)",
                "text_engine": "Text OCR Engine:",
                "masks_dir": "Masks Directory:",
                "browse": "Browse",
                "mask_frame": "Mask Settings",
                "scan_masks": "Scan Masks Directory",
                "delete_selected": "Delete Selected",
                "edit_frame": "Edit Selected Mask",
                "type": "Type:",
                "fixed_check": "Fixed Length Check?",
                "length": "Length:",
                "update_row": "Update Row",
                "edit_region": "Edit Region",
                "log_frame": "Execution Log",
                "start": "Start Processing",
                "error_paths": "ERROR: paths must be specified.",
                "no_images": "No images found in the input directory.",
                "scanned": "Scanned directory. Added {0} new masks."
            },
            "日本語": {
                "title": "OCR アプリケーション",
                "lang_label": "UI言語:",
                "settings_frame": "設定 / パス",
                "input_dir": "入力フォルダ:",
                "output_dir": "出力フォルダ:",
                "template": "テンプレート画像:",
                "use_gpu": "GPUアクセラレーションを使用",
                "text_engine": "テキスト認識エンジン:",
                "masks_dir": "マスクフォルダ:",
                "browse": "参照",
                "mask_frame": "領域（マスク）設定",
                "scan_masks": "マスク画像を自動スキャン",
                "delete_selected": "選択項目を削除",
                "edit_frame": "選択項目のプロパティ編集",
                "type": "文字種:",
                "fixed_check": "桁数一致チェックを有効化",
                "length": "必要桁数:",
                "update_row": "設定を反映",
                "edit_region": "領域を指定",
                "log_frame": "実行ログ",
                "start": "処理開始",
                "error_paths": "エラー：入力・出力・テンプレートのパスはすべて必須です。",
                "no_images": "エラー：対象の画像ファイルが見つかりません。",
                "scanned": "スキャン完了: {0} 件の座標を追加しました。"
            }
        }
        
        self.root.title(self.t("title"))
        self.update_widgets = []
        
        self.input_dir = tk.StringVar(value=os.path.join(self.base_dir, "input"))
        self.output_dir = tk.StringVar(value=os.path.join(self.base_dir, "output"))
        self.template_path = tk.StringVar(value=os.path.join(self.base_dir, "template", "template.png"))
        self.masks_dir = tk.StringVar(value=os.path.join(self.base_dir, "masks"))

        self.use_gpu = tk.BooleanVar(value=self.app_config.get("use_gpu", True))
        self.text_engine_var = tk.StringVar(value=self.app_config.get("text_engine", "MangaOCR"))
        
        self.create_widgets()

    def save_settings(self):
        self.app_config["language"] = self.lang
        self.app_config["use_gpu"] = self.use_gpu.get()
        self.app_config["text_engine"] = self.text_engine_var.get()
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(self.app_config, f, indent=4)

    def t(self, key):
        return self.strings[self.lang].get(key, key)
        
    def track(self, widget, key, attr='text'):
        self.update_widgets.append((widget, key, attr))
        
    def switch_language(self, event=None):
        self.lang = self.lang_combo.get()
        self.save_settings()
        self.root.title(self.t("title"))
        for widget, key, attr in self.update_widgets:
            if attr == 'text':
                widget.config(text=self.t(key))

    def browse_folder(self, var):
        folder_path = filedialog.askdirectory()
        if folder_path:
            var.set(folder_path)
            self.save_settings()

    def browse_file(self, var):
        file_path = filedialog.askopenfilename()
        if file_path:
            var.set(file_path)
            self.save_settings()

    def create_widgets(self):
        # Top Frame for Language
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl = ttk.Label(top_frame, text=self.t("lang_label"))
        lbl.pack(side=tk.LEFT)
        self.track(lbl, "lang_label")
        
        self.lang_combo = ttk.Combobox(top_frame, values=("English", "日本語"), state="readonly", width=15)
        self.lang_combo.set(self.lang)
        self.lang_combo.pack(side=tk.LEFT)
        self.lang_combo.bind("<<ComboboxSelected>>", self.switch_language)
        
        # 1. Path selection frame
        self.path_frame = ttk.LabelFrame(self.root, text=self.t("settings_frame"))
        self.path_frame.pack(fill=tk.X, padx=10, pady=5)
        self.track(self.path_frame, "settings_frame")
        
        # Input Dir
        row_input = ttk.Frame(self.path_frame)
        row_input.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(row_input, text=self.t("input_dir"))
        lbl.pack(side=tk.LEFT, padx=5)
        self.track(lbl, "input_dir")
        ttk.Entry(row_input, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn = ttk.Button(row_input, text=self.t("browse"), command=lambda: self.browse_folder(self.input_dir))
        btn.pack(side=tk.LEFT)
        self.track(btn, "browse")
        
        # Output Dir
        row_output = ttk.Frame(self.path_frame)
        row_output.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(row_output, text=self.t("output_dir"))
        lbl.pack(side=tk.LEFT, padx=5)
        self.track(lbl, "output_dir")
        ttk.Entry(row_output, textvariable=self.output_dir, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn = ttk.Button(row_output, text=self.t("browse"), command=lambda: self.browse_folder(self.output_dir))
        btn.pack(side=tk.LEFT)
        self.track(btn, "browse")
        
        # Template Path
        row_template = ttk.Frame(self.path_frame)
        row_template.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(row_template, text=self.t("template"))
        lbl.pack(side=tk.LEFT, padx=5)
        self.track(lbl, "template")
        ttk.Entry(row_template, textvariable=self.template_path, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn = ttk.Button(row_template, text=self.t("browse"), command=lambda: self.browse_file(self.template_path))
        btn.pack(side=tk.LEFT)
        self.track(btn, "browse")
        
        # Masks Dir
        row_masks = ttk.Frame(self.path_frame)
        row_masks.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(row_masks, text=self.t("masks_dir"))
        lbl.pack(side=tk.LEFT, padx=5)
        self.track(lbl, "masks_dir")
        ttk.Entry(row_masks, textvariable=self.masks_dir, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn = ttk.Button(row_masks, text=self.t("browse"), command=lambda: self.browse_folder(self.masks_dir))
        btn.pack(side=tk.LEFT)
        self.track(btn, "browse")

        # GPU Checkbox
        row_gpu = ttk.Frame(self.path_frame)
        row_gpu.pack(fill=tk.X, pady=2)
        chk_gpu = ttk.Checkbutton(row_gpu, text=self.t("use_gpu"), variable=self.use_gpu, command=self.save_settings)
        chk_gpu.pack(side=tk.LEFT, padx=5)
        self.track(chk_gpu, "use_gpu")

        # Text Engine Dropdown
        row_engine = ttk.Frame(self.path_frame)
        row_engine.pack(fill=tk.X, pady=2)
        lbl_engine = ttk.Label(row_engine, text=self.t("text_engine"))
        lbl_engine.pack(side=tk.LEFT, padx=5)
        self.track(lbl_engine, "text_engine")
        engine_cb = ttk.Combobox(row_engine, textvariable=self.text_engine_var, values=["MangaOCR", "EasyOCR"], state="readonly", width=15)
        engine_cb.pack(side=tk.LEFT, padx=5)
        engine_cb.bind("<<ComboboxSelected>>", lambda e: self.save_settings())
        
        # 2. Mask settings table
        self.mask_frame = ttk.LabelFrame(self.root, text=self.t("mask_frame"))
        self.mask_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.track(self.mask_frame, "mask_frame")
        
        btn_frame = ttk.Frame(self.mask_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        btn = ttk.Button(btn_frame, text=self.t("scan_masks"), command=self.scan_masks)
        btn.pack(side=tk.LEFT, padx=5)
        self.track(btn, "scan_masks")
        btn = ttk.Button(btn_frame, text=self.t("delete_selected"), command=self.delete_mask)
        btn.pack(side=tk.LEFT, padx=5)
        self.track(btn, "delete_selected")
        
        btn = ttk.Button(btn_frame, text=self.t("edit_region"), command=self.edit_region_window)
        btn.pack(side=tk.LEFT, padx=5)
        self.track(btn, "edit_region")
        
        columns = ("ID", "Type", "X", "Y", "W", "H", "Fixed", "Length")
        self.tree = ttk.Treeview(self.mask_frame, columns=columns, show="headings", height=6)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, padx=5, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        # Edit Area
        self.edit_frame = ttk.LabelFrame(self.mask_frame, text=self.t("edit_frame"))
        self.edit_frame.pack(fill=tk.X, padx=5, pady=5)
        self.track(self.edit_frame, "edit_frame")
        
        self.edit_id = tk.StringVar()
        self.edit_type = tk.StringVar()
        self.edit_fixed = tk.BooleanVar()
        self.edit_length = tk.IntVar()
        
        ttk.Label(self.edit_frame, text="ID:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(self.edit_frame, textvariable=self.edit_id, width=15).grid(row=0, column=1, padx=5)
        
        lbl = ttk.Label(self.edit_frame, text=self.t("type"))
        lbl.grid(row=0, column=2, padx=5, pady=2)
        self.track(lbl, "type")
        
        type_cb = ttk.Combobox(self.edit_frame, textvariable=self.edit_type, values=("text", "digits"), width=10, state="readonly")
        type_cb.grid(row=0, column=3, padx=5)
        
        chk = ttk.Checkbutton(self.edit_frame, text=self.t("fixed_check"), variable=self.edit_fixed)
        chk.grid(row=0, column=4, padx=5)
        self.track(chk, "fixed_check")
        
        lbl = ttk.Label(self.edit_frame, text=self.t("length"))
        lbl.grid(row=0, column=5, padx=5, pady=2)
        self.track(lbl, "length")
        
        ttk.Entry(self.edit_frame, textvariable=self.edit_length, width=5).grid(row=0, column=6, padx=5)
        btn = ttk.Button(self.edit_frame, text=self.t("update_row"), command=self.update_row)
        btn.grid(row=0, column=7, padx=10)
        self.track(btn, "update_row")
        
        self.load_config()
        
        # 3. Execution Log
        self.log_frame = ttk.LabelFrame(self.root, text=self.t("log_frame"))
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.track(self.log_frame, "log_frame")
        
        scrollbar = ttk.Scrollbar(self.log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(self.log_frame, height=8, state=tk.DISABLED, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.log_text.yview)
        
        # 4. Actions
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        self.start_btn = ttk.Button(action_frame, text=self.t("start"), command=self.start_processing)
        self.start_btn.pack(side=tk.RIGHT)
        self.track(self.start_btn, "start")

    def load_config(self):
        config_path = os.path.join(self.masks_dir.get(), "config.json")
        saved_dict = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for m in saved:
                    saved_dict[m.get("id")] = m
            except Exception as e:
                print("Error loading config:", e)

        for child in self.tree.get_children():
            self.tree.delete(child)

        mdir = self.masks_dir.get()
        if not os.path.isdir(mdir):
            return

        for f in os.listdir(mdir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                fid = os.path.splitext(f)[0]
                if fid in saved_dict:
                    m = saved_dict[fid]
                    fixed = "True" if m.get("expected_length", 0) > 0 else "False"
                    self.tree.insert("", "end", values=(
                        m.get("id"), m.get("type", "text"), m.get("x", 0), m.get("y", 0),
                        m.get("w", 0), m.get("h", 0), fixed, m.get("expected_length", 0)
                    ))

    def save_masks_config(self):
        masks = []
        for child in self.tree.get_children():
            v = self.tree.item(child, 'values')
            exp_len = int(v[7]) if v[6] == "True" else 0
            masks.append({
                "id": v[0], "type": v[1], 
                "x": int(v[2]), "y": int(v[3]), 
                "w": int(v[4]), "h": int(v[5]), 
                "expected_length": exp_len
            })
            
        masks_dir = self.masks_dir.get()
        if not masks_dir or not os.path.isdir(masks_dir): return
        masks_path = os.path.join(masks_dir, "config.json")
        try:
            with open(masks_path, "w", encoding='utf-8') as f:
                json.dump(masks, f, indent=4)
        except Exception as e:
            print("Error saving config:", e)

    def scan_masks(self):
        mdir = self.masks_dir.get()
        if not os.path.isdir(mdir):
            return
            
        found = 0
        for f in os.listdir(mdir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(mdir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours or cv2.boundingRect(max(contours, key=cv2.contourArea))[2] < 5:
                    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    
                    if w > 5 and h > 5:
                        fid = os.path.splitext(f)[0]
                        exists = False
                        for child in self.tree.get_children():
                            if self.tree.item(child, 'values')[0] == fid:
                                exists = True
                                break
                        if not exists:
                            self.tree.insert("", "end", values=(fid, "text", x, y, w, h, "False", 0))
                            found += 1
        if found > 0:
            self.save_masks_config()
        self.log(self.t("scanned").format(found))

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            v = self.tree.item(selected[0], 'values')
            self.edit_id.set(v[0])
            self.edit_type.set(v[1])
            self.edit_fixed.set(v[6] == "True")
            try:
                self.edit_length.set(int(v[7]))
            except ValueError:
                self.edit_length.set(0)

    def update_row(self):
        selected = self.tree.selection()
        if not selected: return
        
        idx = selected[0]
        old_v = self.tree.item(idx, 'values')
        
        fixed_str = "True" if self.edit_fixed.get() else "False"
        try:
            length_val = self.edit_length.get()
        except tk.TclError:
            length_val = 0
            
        self.tree.item(idx, values=(
            self.edit_id.get(), self.edit_type.get(),
            old_v[2], old_v[3], old_v[4], old_v[5],
            fixed_str, length_val
        ))
        self.save_masks_config()

    def delete_mask(self):
        selected = self.tree.selection()
        if not selected: return
        for item in selected:
            self.tree.delete(item)
        self.save_masks_config()

    def edit_region_window(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row first.")
            return
            
        tpl_path = self.template_path.get()
        if not os.path.exists(tpl_path):
            messagebox.showerror("Error", "Template image not found.")
            return

        idx = selected[0]
        v = self.tree.item(idx, 'values')
        fid = v[0]
        curr_x, curr_y, curr_w, curr_h = int(v[2]), int(v[3]), int(v[4]), int(v[5])

        top = tk.Toplevel(self.root)
        top.title(f"{self.t('edit_region')} - {fid}")
        top.geometry("800x600")
        
        import cv2
        import numpy as np
        import base64
        
        img_gray = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        H, W = img_gray.shape
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        screen_w, screen_h = self.root.winfo_screenwidth() - 100, self.root.winfo_screenheight() - 150
        scale = min(1.0, screen_w / W, screen_h / H)
        disp_w, disp_h = int(W * scale), int(H * scale)
        
        if scale < 1.0:
            img_disp = cv2.resize(img_rgb, (disp_w, disp_h))
        else:
            img_disp = img_rgb

        success, buffer = cv2.imencode('.png', cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR))
        if success:
            b64_data = base64.b64encode(buffer).decode('utf-8')
            photo = tk.PhotoImage(data=b64_data)
        else:
            return

        canvas_frame = ttk.Frame(top)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, width=disp_w, height=disp_h, cursor="cross")
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.image = photo
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        sx, sy = int(curr_x * scale), int(curr_y * scale)
        sw, sh = int(curr_w * scale), int(curr_h * scale)
        rect_id = canvas.create_rectangle(sx, sy, sx+sw, sy+sh, outline="red", width=2)

        state = {"start_x": 0, "start_y": 0, "end_x": 0, "end_y": 0, "drawing": False}

        def on_press(event):
            state["start_x"] = event.x
            state["start_y"] = event.y
            state["drawing"] = True
            canvas.coords(rect_id, event.x, event.y, event.x, event.y)

        def on_drag(event):
            if state["drawing"]:
                canvas.coords(rect_id, state["start_x"], state["start_y"], event.x, event.y)

        def on_release(event):
            state["drawing"] = False

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def save():
            coords = canvas.coords(rect_id)
            if coords:
                x1, y1, x2, y2 = coords
                nx1, ny1, nx2, ny2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                orig_x, orig_y = int(nx1 / scale), int(ny1 / scale)
                orig_w, orig_h = int((nx2 - nx1) / scale), int((ny2 - ny1) / scale)

                self.tree.item(idx, values=(
                    v[0], v[1], orig_x, orig_y, orig_w, orig_h, v[6], v[7]
                ))
                self.save_masks_config()
                
                mdir = self.masks_dir.get()
                if os.path.exists(mdir):
                    mask_img = np.zeros((H, W), dtype=np.uint8)
                    cv2.rectangle(mask_img, (orig_x, orig_y), (orig_x+orig_w, orig_y+orig_h), 255, -1)
                    cv2.imwrite(os.path.join(mdir, f"{fid}.png"), mask_img)

            top.destroy()

        btn = ttk.Button(top, text="Save Region", command=save)
        btn.pack(pady=10)

    def log(self, message):
        self.root.after(0, self._log_ui, message)
        
    def _log_ui(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_processing(self):
        self.log("Starting OCR processing pipeline...")
        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_pipeline_thread, daemon=True).start()

    def run_pipeline_thread(self):
        try:
            self._run_pipeline_logic()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.log(f"ERROR: {e}\n{err}")
        finally:
            self.log("Pipeline execution finished.")
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def _run_pipeline_logic(self):
        in_dir = self.input_dir.get()
        out_dir = self.output_dir.get()
        template_path = self.template_path.get()
        masks_dir = self.masks_dir.get()
        
        if not in_dir or not out_dir or not template_path:
            self.log(self.t("error_paths"))
            return

        os.makedirs(out_dir, exist_ok=True)
        
        masks = []
        for child in self.tree.get_children():
            v = self.tree.item(child, 'values')
            exp_len = int(v[7]) if v[6] == "True" else 0
            masks.append({
                "id": v[0], "type": v[1], 
                "x": int(v[2]), "y": int(v[3]), 
                "w": int(v[4]), "h": int(v[5]), 
                "expected_length": exp_len
            })
            
        os.makedirs(masks_dir, exist_ok=True)
        self.save_masks_config()
        masks_path = os.path.join(masks_dir, "config.json")
        self.log(f"Saved mask configuration to {masks_path}")
        
        try:
            import sys
            p = os.path.dirname(os.path.abspath(__file__))
            if p not in sys.path: sys.path.append(p)
            from src.image_processing import ImageProcessor
            from src.ocr_engine import OCREngine
        except ImportError as e:
            self.log(f"ERROR loading modules: {e}")
            return
            
        use_gpu = self.use_gpu.get()
        text_engine = self.text_engine_var.get()
        
        self.log("Initializing ImageProcessor and OCREngine...")
        image_processor = ImageProcessor(template_path, masks_path)
        ocr_engine = OCREngine(use_gpu=use_gpu, text_engine=text_engine)
        
        self.log("Warming up OCR engines...")
        ocr_engine.warmup()
        
        image_files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            self.log(self.t("no_images"))
            return
            
        self.log(f"Found {len(image_files)} images to process.")
        final_results, crop_report, ocr_report = [], [], []
        
        for idx, file_name in enumerate(image_files):
            self.log(f"[{idx+1}/{len(image_files)}] Processing {file_name}...")
            file_path = os.path.join(in_dir, file_name)
            
            aligned_img, status = image_processor.align_and_correct_rotation(file_path)
            if status != "SUCCESS" or aligned_img is None:
                self.log(f"  -> ERROR: Alignment failed for {file_name} ({status})")
                crop_report.append({"file_name": file_name, "error": status})
                continue
                
            intermediate_dir = os.path.join(out_dir, "intermediate")
            roi_infos = image_processor.extract_rois(aligned_img, intermediate_dir, file_name)
            self.log(f"  -> Extracted {len(roi_infos)} functional ROIs.")
            
            file_data = {"file_name": file_name, "results": []}
            for roi in roi_infos:
                ocr_res = ocr_engine.process_roi(roi)
                if ocr_res["error_flags"]:
                    self.log(f"  -> ROI Validation Warning on {roi['roi_id']}: {ocr_res['error_flags']}")
                    ocr_report.append({
                        "file_name": file_name, "roi_id": roi['roi_id'],
                        "raw_text": ocr_res["raw_text"], "error_flags": ocr_res["error_flags"]
                    })
                file_data["results"].append({
                    "roi_id": roi['roi_id'], "raw_text": ocr_res["raw_text"]
                })
            
            final_results.append(file_data)
            
        intermediate_dir = os.path.join(out_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        with open(os.path.join(intermediate_dir, "crop_report.json"), "w", encoding='utf-8') as f:
            json.dump(crop_report, f, ensure_ascii=False, indent=4)
        with open(os.path.join(out_dir, "ocr_report.json"), "w", encoding='utf-8') as f:
            json.dump(ocr_report, f, ensure_ascii=False, indent=4)
        final_path = os.path.join(out_dir, "results.json")
        with open(final_path, "w", encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
            
        self.log("Processing completed successfully. See raw output at output/ directory.")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
