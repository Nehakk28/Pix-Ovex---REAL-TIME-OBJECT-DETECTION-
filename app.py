import os
import sys
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta

import customtkinter as ctk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# PATHS / TENSORFLOW SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(BASE_DIR, "models", "research")
SLIM_DIR = os.path.join(RESEARCH_DIR, "slim")

sys.path.insert(0, RESEARCH_DIR)
sys.path.insert(0, SLIM_DIR)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

DB_NAME = os.path.join(BASE_DIR, "users.db")
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_model")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.pbtxt")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PixOvexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pix ovex")
        self.geometry("920x760")
        self.minsize(920, 760)
        self.configure(fg_color="#0a0f1f")

        self.current_user = None

        self.login_username_entry = None
        self.login_password_entry = None
        self.signup_username_entry = None
        self.signup_password_entry = None

        self.cap = None
        self.camera_running = False
        self.after_id = None
        self.detect_fn = None
        self.category_index = None
        self.last_saved_time = {}

        self.logo_img_small = None
        self.logo_img_big = None

        self.content = ctk.CTkFrame(self, fg_color="#0a0f1f", corner_radius=0)
        self.content.pack(fill="both", expand=True)

        self.init_db()
        self.load_logo_assets()
        self.show_home()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -----------------------------
    # DATABASE
    # -----------------------------
    def init_db(self):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                object_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TEXT
            )
        """)

        cur.execute("PRAGMA table_info(history)")
        columns = [row[1] for row in cur.fetchall()]
        if "detected_at" not in columns:
            cur.execute("ALTER TABLE history ADD COLUMN detected_at TEXT")

        conn.commit()
        conn.close()

    def save_history(self, username, object_name, confidence):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (username, object_name, confidence, detected_at)
            VALUES (?, ?, ?, ?)
        """, (
            username,
            object_name,
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

    def get_total_detections(self, username):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM history WHERE username=?", (username,))
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_objects_found(self, username):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT object_name) FROM history WHERE username=?", (username,))
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_sessions_today(self, username):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT date(detected_at))
            FROM history
            WHERE username=? AND date(detected_at)=date('now','localtime')
        """, (username,))
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_recent_history(self, username, limit=3):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            SELECT object_name, confidence, COALESCE(detected_at, '')
            FROM history
            WHERE username=?
            ORDER BY id DESC
            LIMIT ?
        """, (username, limit))
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_detection_trend(self, username, days=7):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        cur.execute("""
            SELECT date(detected_at) AS day, COUNT(*)
            FROM history
            WHERE username=? AND detected_at IS NOT NULL
            GROUP BY day
            ORDER BY day
        """, (username,))

        rows = cur.fetchall()
        conn.close()

        today = datetime.now().date()
        date_list = [(today - timedelta(days=i)) for i in reversed(range(days))]
        count_map = {d: 0 for d in date_list}

        for day_str, count in rows:
            try:
                day_obj = datetime.strptime(day_str, "%Y-%m-%d").date()
                if day_obj in count_map:
                    count_map[day_obj] = count
            except Exception:
                pass

        labels = [d.strftime("%d %b") for d in date_list]
        values = [count_map[d] for d in date_list]
        return labels, values

    # -----------------------------
    # ASSETS
    # -----------------------------
    def load_logo_assets(self):
        self.logo_img_small = None
        self.logo_img_big = None

        if os.path.exists(LOGO_PATH):
            try:
                img = Image.open(LOGO_PATH).convert("RGBA")
                self.logo_img_small = ImageTk.PhotoImage(img.resize((38, 38)))
                self.logo_img_big = ImageTk.PhotoImage(img.resize((200, 200)))
            except Exception:
                self.logo_img_small = None
                self.logo_img_big = None

    # -----------------------------
    # HELPERS
    # -----------------------------
    def clear_content(self):
        self.stop_camera()
        for widget in self.content.winfo_children():
            widget.destroy()

    def load_detector(self):
        if self.detect_fn is not None and self.category_index is not None:
            return

        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"Model folder not found: {MODEL_DIR}\n"
                f"Put your TensorFlow SavedModel inside model/saved_model"
            )

        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(
                f"Label map not found: {LABEL_MAP_PATH}\n"
                f"Put your label map file inside model/label_map.pbtxt"
            )

        self.category_index = label_map_util.create_category_index_from_labelmap(
            LABEL_MAP_PATH,
            use_display_name=True
        )
        self.detect_fn = tf.saved_model.load(MODEL_DIR)

    def scalar(self, value, cast=float):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            raise ValueError("Empty detection value")
        return cast(arr[0])

    def add_navbar(self, parent, show_login_btn=True, back_home=False):
        nav = ctk.CTkFrame(parent, fg_color="#0b1324", corner_radius=0, height=70)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        left = ctk.CTkFrame(nav, fg_color="transparent")
        left.pack(side="left", padx=24, pady=12)

        if self.logo_img_small:
            logo_lbl = tk.Label(left, image=self.logo_img_small, bg="#0b1324", bd=0)
            logo_lbl.pack(side="left", padx=(0, 10))
        else:
            ctk.CTkLabel(
                left,
                text="◉",
                text_color="#2bb4ff",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(side="left", padx=(0, 10))

        brand = ctk.CTkFrame(left, fg_color="transparent")
        brand.pack(side="left")

        ctk.CTkLabel(
            brand,
            text="Pix",
            text_color="#2bb4ff",
            font=ctk.CTkFont(size=60, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            brand,
            text="ovex",
            text_color="#a78bfa",
            font=ctk.CTkFont(size=60, weight="bold")
        ).pack(side="left", padx=(6, 0))

        right = ctk.CTkFrame(nav, fg_color="transparent")
        right.pack(side="right", padx=22, pady=16)

        if back_home:
            ctk.CTkButton(
                right,
                text="Back to Home",
                width=120,
                height=38,
                corner_radius=12,
                fg_color="transparent",
                border_width=1,
                border_color="#374151",
                hover_color="#111827",
                text_color="white",
                command=self.show_home
            ).pack(side="right", padx=6)

        if show_login_btn:
            ctk.CTkButton(
                right,
                text="Login",
                width=90,
                height=38,
                corner_radius=12,
                fg_color="transparent",
                border_width=1,
                border_color="#374151",
                hover_color="#111827",
                text_color="white",
                command=self.show_login
            ).pack(side="right", padx=6)

            ctk.CTkButton(
                right,
                text="Sign Up",
                width=90,
                height=38,
                corner_radius=12,
                fg_color="transparent",
                border_width=1,
                border_color="#374151",
                hover_color="#111827",
                text_color="white",
                command=self.show_signup
            ).pack(side="right", padx=6)

    def make_stat_card(self, parent, title, value, value_color="#38bdf8"):
        card = ctk.CTkFrame(parent, fg_color="#20293b", corner_radius=16)
        card.pack(side="left", expand=True, fill="both", padx=10, pady=10)

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14),
            text_color="#94a3b8"
        ).pack(anchor="w", padx=18, pady=(16, 4))

        ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=value_color
        ).pack(anchor="w", padx=18, pady=(0, 16))

    def make_history_row(self, parent, label, conf, time_text):
        row = ctk.CTkFrame(parent, fg_color="#1b2435", corner_radius=12)
        row.pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(
            row,
            text=label,
            font=ctk.CTkFont(size=15),
            text_color="#d1d5db"
        ).pack(side="left", padx=18, pady=14)

        badge = ctk.CTkFrame(row, fg_color="#22354d", corner_radius=999, width=60, height=28)
        badge.pack(side="left", padx=12)
        badge.pack_propagate(False)

        ctk.CTkLabel(
            badge,
            text=conf,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#22d3ee"
        ).pack(expand=True)

        ctk.CTkLabel(
            row,
            text=time_text,
            font=ctk.CTkFont(size=12),
            text_color="#6b7280"
        ).pack(side="right", padx=18, pady=14)

    def draw_history_graph(self, parent, username):
        for widget in parent.winfo_children():
            widget.destroy()

        labels, values = self.get_detection_trend(username, days=7)

        ctk.CTkLabel(
            parent,
            text="Detection History Graph",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=18, pady=(14, 6))

        ctk.CTkLabel(
            parent,
            text="Detections per day for the last 7 days",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        ).pack(anchor="w", padx=18, pady=(0, 8))

        fig = Figure(figsize=(8.4, 3.2), dpi=100)
        ax = fig.add_subplot(111)

        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        x = list(range(len(labels)))
        ax.plot(x, values, marker="o", linewidth=2.5, color="#22d3ee")
        ax.fill_between(x, values, color="#22d3ee", alpha=0.14)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="#cbd5e1", fontsize=9)
        ax.set_ylabel("Detections", color="#cbd5e1")
        ax.set_xlabel("Date", color="#cbd5e1")
        ax.tick_params(axis="y", colors="#cbd5e1")
        ax.grid(True, alpha=0.18)

        for spine in ax.spines.values():
            spine.set_color("#334155")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 12))

    # -----------------------------
    # HOME PAGE
    # -----------------------------
    def show_home(self):
        self.clear_content()
        self.geometry("920x760")
        self.minsize(920, 760)
        self.configure(fg_color="#0a0f1f")

        page = ctk.CTkScrollableFrame(
            self.content,
            fg_color="#0a0f1f",
            scrollbar_button_color="#1f2a3d",
            scrollbar_button_hover_color="#2b3a52"
        )
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=1)

        navbar = ctk.CTkFrame(page, fg_color="#0b1324", corner_radius=18, height=72)
        navbar.pack(fill="x", padx=10, pady=(10, 16))
        navbar.pack_propagate(False)

        left_nav = ctk.CTkFrame(navbar, fg_color="transparent")
        left_nav.pack(side="left", padx=22, pady=18)

        if self.logo_img_small:
            logo_lbl = tk.Label(left_nav, image=self.logo_img_small, bg="#0b1324", bd=0)
            logo_lbl.pack(side="left", padx=(0, 10))
        else:
            ctk.CTkLabel(
                left_nav,
                text="◉",
                text_color="#2bb4ff",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(side="left", padx=(0, 8))

        brand = ctk.CTkFrame(left_nav, fg_color="transparent")
        brand.pack(side="left")

        ctk.CTkLabel(
            brand,
            text="Pix",
            text_color="#2bb4ff",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            brand,
            text="ovex",
            text_color="#a78bfa",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left", padx=(6, 0))

        right_nav = ctk.CTkFrame(navbar, fg_color="transparent")
        right_nav.pack(side="right", padx=22, pady=16)

        ctk.CTkButton(
            right_nav,
            text="Login",
            width=90,
            height=38,
            corner_radius=12,
            fg_color="transparent",
            border_width=1,
            border_color="#374151",
            hover_color="#111827",
            text_color="white",
            command=self.show_login
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            right_nav,
            text="Sign Up",
            width=90,
            height=38,
            corner_radius=12,
            fg_color="transparent",
            border_width=1,
            border_color="#374151",
            hover_color="#111827",
            text_color="white",
            command=self.show_signup
        ).pack(side="left", padx=6)

        hero = ctk.CTkFrame(page, fg_color="transparent")
        hero.pack(fill="x", padx=38, pady=(20, 22))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(hero, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 20), pady=20)

        badge = ctk.CTkFrame(left, fg_color="#0f2b46", corner_radius=999, height=34)
        badge.pack(anchor="w", pady=(0, 22))
        badge.pack_propagate(False)

        ctk.CTkLabel(
            badge,
            text="- AI-powered detection",
            text_color="#2bb4ff",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", padx=(0, 14), pady=6)

        ctk.CTkLabel(
            left,
            text="See every pixel",
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color="white",
            justify="left"
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text="Detect everything.",
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color="#a78bfa",
            justify="left"
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text="Turning cameras into smart AI eyes \nthat understand the world instantly.",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white",
            justify="left"
        ).pack(anchor="w")


        ctk.CTkLabel(
            left,
            text="— Real-time object detection.",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2bb4ff",
            justify="left"
        ).pack(anchor="w", pady=(6, 14))

        ctk.CTkLabel(
            left,
            text=(
                "Detect objects from your laptop camera in real\n"
                "time. Login to start detecting and view your full\n"
                "session history."
            ),
            font=ctk.CTkFont(size=18),
            text_color="#6f8097",
            justify="left"
        ).pack(anchor="w", pady=(10, 0))

        right = ctk.CTkFrame(
            hero,
            fg_color="#111827",
            corner_radius=20,
            border_width=1,
            border_color="#1f2a3d"
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(20, 0), pady=20)

        head_row = ctk.CTkFrame(right, fg_color="transparent")
        head_row.pack(fill="x", padx=18, pady=(14, 10))

        ctk.CTkLabel(
            head_row,
            text="●  Live Camera — detecting various objects",
            font=ctk.CTkFont(size=15),
            text_color="#74839b"
        ).pack(side="left")

        ctk.CTkLabel(
            head_row,
            text="PIXOVEX",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#2bb4ff"
        ).pack(side="right")

        preview = ctk.CTkFrame(right, fg_color="#0c1324", corner_radius=16)
        preview.pack(fill="both", expand=True, padx=16, pady=(0, 0))

        canvas = tk.Canvas(preview, bg="#0c1324", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=8, pady=8)

        for x in range(0, 420, 38):
            canvas.create_line(x, 0, x, 280, fill="#0f2238")
        for y in range(0, 280, 38):
            canvas.create_line(0, y, 420, y, fill="#0f2238")

        canvas.create_text(95, 22, text="Person 96%", fill="#2bb4ff", font=("Arial", 9, "bold"))
        canvas.create_rectangle(60, 32, 164, 176, outline="#2bb4ff", width=2)

        canvas.create_text(285, 42, text="Chair 89%", fill="#a78bfa", font=("Arial", 9, "bold"))
        canvas.create_rectangle(252, 52, 354, 178, outline="#a78bfa", width=2)

        canvas.create_line(12, 88, 398, 88, fill="#174a73")

        bottom = ctk.CTkFrame(right, fg_color="#111827", corner_radius=0)
        bottom.pack(fill="x")

        bottom_inner = ctk.CTkFrame(bottom, fg_color="#111827", corner_radius=0)
        bottom_inner.pack(fill="x", padx=18, pady=12)

        ctk.CTkLabel(
            bottom_inner,
            text="● Detection active",
            font=ctk.CTkFont(size=14),
            text_color="#4ade80"
        ).pack(side="left")

        ctk.CTkLabel(
            bottom_inner,
            text="30 FPS",
            font=ctk.CTkFont(size=14),
            text_color="#42506a"
        ).pack(side="right")

        ctk.CTkFrame(page, fg_color="#152033", height=1, corner_radius=0).pack(fill="x", padx=10, pady=(10, 28))

        ctk.CTkLabel(
            page,
            text="HOW IT WORKS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#2bb4ff"
        ).pack(anchor="w", padx=38, pady=(0, 8))

        ctk.CTkLabel(
            page,
            text="Three steps to start detecting",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="white"
        ).pack(anchor="w", padx=38, pady=(0, 24))

        steps = ctk.CTkFrame(page, fg_color="transparent")
        steps.pack(fill="x", padx=32, pady=(0, 30))

        def step_card(parent, num, title, desc, icon):
            card = ctk.CTkFrame(
                parent,
                fg_color="#111827",
                corner_radius=18,
                border_width=1,
                border_color="#1f2a3d",
                height=210
            )
            card.pack(side="left", expand=True, fill="both", padx=10)
            card.pack_propagate(False)

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=18, pady=(18, 8))

            icon_box = ctk.CTkFrame(top, fg_color="#10263d", corner_radius=14, width=46, height=46)
            icon_box.pack(side="left")
            icon_box.pack_propagate(False)

            ctk.CTkLabel(
                icon_box,
                text=icon,
                text_color="#2bb4ff",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(expand=True)

            ctk.CTkLabel(
                top,
                text=num,
                text_color="#173553",
                font=ctk.CTkFont(size=40, weight="bold")
            ).pack(side="right")

            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color="white"
            ).pack(anchor="w", padx=18, pady=(8, 4))

            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=13),
                text_color="#6f8097",
                justify="left",
                wraplength=230
            ).pack(anchor="w", padx=18, pady=(0, 18))

        step_card(
            steps,
            "01",
            "Create an account",
            "Sign up in seconds and get instant access to all detection features.",
            "👤"
        )
        step_card(
            steps,
            "02",
            "Start your camera",
            "Allow camera access and the AI begins detecting objects in real time instantly.",
            "📷"
        )
        step_card(
            steps,
            "03",
            "Review your history",
            "All detections are saved automatically. Browse your full session history anytime.",
            "⟳"
        )

        footer = ctk.CTkFrame(page, fg_color="transparent")
        footer.pack(fill="x", padx=38, pady=(12, 18))

        ctk.CTkLabel(
            footer,
            text="Powered by computer vision",
            font=ctk.CTkFont(size=12),
            text_color="#55647c"
        ).pack(side="left")

        ctk.CTkLabel(
            footer,
            text="● All systems live",
            font=ctk.CTkFont(size=12),
            text_color="#4ade80"
        ).pack(side="right")

    # -----------------------------
    # LOGIN / SIGNUP
    # -----------------------------
    def show_login(self):
        self.clear_content()
        self.geometry("900x620")

        base = ctk.CTkFrame(self.content, fg_color="#0b1324", corner_radius=0)
        base.pack(fill="both", expand=True)

        self.add_navbar(base, show_login_btn=False, back_home=True)

        card = ctk.CTkFrame(
            base,
            fg_color="#20293b",
            corner_radius=22,
            border_width=1,
            border_color="#2b3648"
        )
        card.pack(pady=70, padx=250, fill="both", expand=False)

        ctk.CTkLabel(
            card,
            text="Welcome back",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w", padx=34, pady=(34, 4))

        ctk.CTkLabel(
            card,
            text="Login to start detecting objects",
            font=ctk.CTkFont(size=14),
            text_color="#7c8aa0"
        ).pack(anchor="w", padx=34, pady=(0, 20))

        ctk.CTkLabel(card, text="Username", text_color="#b7c0cf").pack(anchor="w", padx=34)
        self.login_username_entry = ctk.CTkEntry(card, width=340, height=42, corner_radius=12, placeholder_text="you@example.com")
        self.login_username_entry.pack(padx=34, pady=(4, 16))

        ctk.CTkLabel(card, text="Password", text_color="#b7c0cf").pack(anchor="w", padx=34)
        self.login_password_entry = ctk.CTkEntry(card, width=340, height=42, corner_radius=12, show="*", placeholder_text="••••••••")
        self.login_password_entry.pack(padx=34, pady=(4, 20))

        ctk.CTkButton(
            card,
            text="Login",
            width=340,
            height=42,
            corner_radius=12,
            command=self.login_user
        ).pack(padx=34, pady=(0, 12))

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.pack(fill="x", padx=34, pady=(0, 28))

        ctk.CTkLabel(footer, text="Don't have an account?", text_color="#7c8aa0").pack(side="left")
        ctk.CTkButton(
            footer,
            text="Sign Up",
            width=70,
            height=28,
            fg_color="transparent",
            text_color="#22d3ee",
            hover_color="#263247",
            command=self.show_signup
        ).pack(side="left", padx=(6, 0))

    def show_signup(self):
        self.clear_content()
        self.geometry("900x700")

        base = ctk.CTkFrame(self.content, fg_color="#0b1324", corner_radius=0)
        base.pack(fill="both", expand=True)

        self.add_navbar(base, show_login_btn=False, back_home=True)

        card = ctk.CTkFrame(
            base,
            fg_color="#20293b",
            corner_radius=22,
            border_width=1,
            border_color="#2b3648"
        )
        card.pack(pady=50, padx=250, fill="both", expand=False)

        ctk.CTkLabel(
            card,
            text="Create account",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w", padx=34, pady=(34, 4))

        ctk.CTkLabel(
            card,
            text="Sign up to start detecting objects",
            font=ctk.CTkFont(size=14),
            text_color="#7c8aa0"
        ).pack(anchor="w", padx=34, pady=(0, 20))

        ctk.CTkLabel(card, text="Username", text_color="#b7c0cf").pack(anchor="w", padx=34)
        self.signup_username_entry = ctk.CTkEntry(card, width=340, height=42, corner_radius=12, placeholder_text="John Doe")
        self.signup_username_entry.pack(padx=34, pady=(4, 16))

        ctk.CTkLabel(card, text="Password", text_color="#b7c0cf").pack(anchor="w", padx=34)
        self.signup_password_entry = ctk.CTkEntry(card, width=340, height=42, corner_radius=12, show="*", placeholder_text="••••••••")
        self.signup_password_entry.pack(padx=34, pady=(4, 20))

        ctk.CTkButton(
            card,
            text="Sign Up",
            width=340,
            height=42,
            corner_radius=12,
            command=self.register_user
        ).pack(padx=34, pady=(0, 12))

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.pack(fill="x", padx=34, pady=(0, 28))

        ctk.CTkLabel(footer, text="Already have an account?", text_color="#7c8aa0").pack(side="left")
        ctk.CTkButton(
            footer,
            text="Login",
            width=70,
            height=28,
            fg_color="transparent",
            text_color="#22d3ee",
            hover_color="#263247",
            command=self.show_login
        ).pack(side="left", padx=(6, 0))

    def register_user(self):
        username = self.signup_username_entry.get().strip()
        password = self.signup_password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Please fill all fields")
            return

        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            conn.close()

            messagebox.showinfo("Success", "Account created successfully")
            self.show_login()

        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists")

    def login_user(self):
        username = self.login_username_entry.get().strip()
        password = self.login_password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Please fill all fields")
            return

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cur.fetchone()
        conn.close()

        if user:
            self.current_user = username
            self.show_dashboard(username)
        else:
            messagebox.showerror("Error", "Invalid username or password")

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    def show_dashboard(self, username):
        self.clear_content()
        self.geometry("1200x760")

        base = ctk.CTkFrame(self.content, fg_color="#0b1324", corner_radius=0)
        base.pack(fill="both", expand=True)

        sidebar = ctk.CTkFrame(base, fg_color="#0e1628", corner_radius=0, width=220)
        sidebar.pack(side="left", fill="y")

        if self.logo_img_small:
            tk.Label(sidebar, image=self.logo_img_small, bg="#0e1628", bd=0).pack(anchor="w", padx=18, pady=(22, 6))

        ctk.CTkLabel(
            sidebar,
            text="Pix ovex",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#ffffff"
        ).pack(anchor="w", padx=20, pady=(0, 6))

        ctk.CTkLabel(
            sidebar,
            text=f"Logged in as\n{username}",
            font=ctk.CTkFont(size=13),
            text_color="#8ca0b3",
            justify="left"
        ).pack(anchor="w", padx=20, pady=(0, 20))

        ctk.CTkButton(
            sidebar,
            text="Dashboard",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#1b3b5a",
            hover_color="#263247",
            anchor="w",
            command=lambda: self.show_dashboard(username)
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Detection",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            anchor="w",
            command=self.show_detection
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="History",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            anchor="w",
            command=self.show_history
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Logout",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            command=self.logout
        ).pack(padx=20, pady=(20, 8))

        main = ctk.CTkFrame(base, fg_color="#0b1324", corner_radius=0)
        main.pack(side="left", fill="both", expand=True)

        total_detections = self.get_total_detections(username)
        sessions_today = self.get_sessions_today(username)
        objects_found = self.get_objects_found(username)
        recent = self.get_recent_history(username)

        ctk.CTkLabel(
            main,
            text="Overview",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(anchor="w", padx=22, pady=(20, 8))

        stats = ctk.CTkFrame(main, fg_color="transparent")
        stats.pack(fill="x", padx=12)

        self.make_stat_card(stats, "Total Detections", str(total_detections))
        self.make_stat_card(stats, "Sessions Today", str(sessions_today))
        self.make_stat_card(stats, "Objects Found", str(objects_found))

        camera_card = ctk.CTkFrame(
            main,
            fg_color="#20293b",
            corner_radius=18,
            border_width=1,
            border_color="#2b3648"
        )
        camera_card.pack(fill="both", expand=False, padx=20, pady=16)

        topbar = ctk.CTkFrame(camera_card, fg_color="transparent")
        topbar.pack(fill="x", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            topbar,
            text="Detection Graph",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#d1d5db"
        ).pack(side="left")

        ctk.CTkButton(
            topbar,
            text="Start Detection",
            width=170,
            height=38,
            corner_radius=14,
            command=self.show_detection
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            topbar,
            text="Refresh",
            width=100,
            height=38,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self.show_dashboard(username)
        ).pack(side="right")

        graph_box = ctk.CTkFrame(camera_card, fg_color="#0f172a", corner_radius=16)
        graph_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.draw_history_graph(graph_box, username)

        history_card = ctk.CTkFrame(
            main,
            fg_color="#20293b",
            corner_radius=18,
            border_width=1,
            border_color="#2b3648"
        )
        history_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            history_card,
            text="Recent History",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=18, pady=(14, 8))

        if recent:
            for obj_name, conf, detected_at in recent:
                self.make_history_row(
                    history_card,
                    f"{obj_name} detected",
                    f"{float(conf):.0f}%",
                    detected_at if detected_at else "just now"
                )
        else:
            ctk.CTkLabel(
                history_card,
                text="No detection history yet.",
                text_color="#94a3b8"
            ).pack(anchor="w", padx=18, pady=12)

    # -----------------------------
    # HISTORY PAGE
    # -----------------------------
    def show_history(self):
        self.clear_content()
        self.geometry("1100x720")

        base = ctk.CTkFrame(self.content, fg_color="#0b1324", corner_radius=0)
        base.pack(fill="both", expand=True)

        sidebar = ctk.CTkFrame(base, fg_color="#0e1628", corner_radius=0, width=220)
        sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(
            sidebar,
            text="Pix ovex",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#ffffff"
        ).pack(anchor="w", padx=20, pady=(22, 6))

        ctk.CTkLabel(
            sidebar,
            text=f"Logged in as\n{self.current_user}",
            font=ctk.CTkFont(size=13),
            text_color="#8ca0b3",
            justify="left"
        ).pack(anchor="w", padx=20, pady=(0, 20))

        ctk.CTkButton(
            sidebar,
            text="Dashboard",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            command=lambda: self.show_dashboard(self.current_user)
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Detection",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            command=self.show_detection
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="History",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#1b3b5a",
            hover_color="#263247",
            command=self.show_history
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Logout",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            command=self.logout
        ).pack(padx=20, pady=(20, 8))

        main = ctk.CTkFrame(base, fg_color="#0b1324", corner_radius=0)
        main.pack(side="left", fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            main,
            text=f"Detection History - {self.current_user}",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(10, 6))

        ctk.CTkLabel(
            main,
            text=f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            font=ctk.CTkFont(size=13),
            text_color="#8ca0b3"
        ).pack(pady=(0, 10))

        table_frame = ctk.CTkFrame(main, fg_color="#20293b", corner_radius=18)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("username", "object_name", "confidence", "detected_at")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)

        tree.heading("username", text="Username")
        tree.heading("object_name", text="Object Name")
        tree.heading("confidence", text="Confidence (%)")
        tree.heading("detected_at", text="Detected At")

        tree.column("username", width=140, anchor="center")
        tree.column("object_name", width=220, anchor="center")
        tree.column("confidence", width=150, anchor="center")
        tree.column("detected_at", width=250, anchor="center")

        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass

        style.configure(
            "Treeview",
            background="#111827",
            foreground="white",
            fieldbackground="#111827",
            rowheight=28,
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#1f2937",
            foreground="white",
            relief="flat"
        )
        style.map("Treeview.Heading", background=[("active", "#334155")])

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10, padx=(0, 10))

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            SELECT username, object_name, confidence, COALESCE(detected_at, '')
            FROM history
            WHERE username=?
            ORDER BY id DESC
        """, (self.current_user,))
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            username, object_name, confidence, detected_at = row
            tree.insert("", "end", values=(
                username,
                object_name,
                f"{float(confidence):.2f}",
                detected_at
            ))

        button_frame = ctk.CTkFrame(main, fg_color="transparent")
        button_frame.pack(pady=10)

        ctk.CTkButton(
            button_frame,
            text="Refresh",
            width=140,
            height=40,
            corner_radius=14,
            command=self.show_history
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame,
            text="Back to Dashboard",
            width=160,
            height=40,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self.show_dashboard(self.current_user)
        ).pack(side="left", padx=10)

    # -----------------------------
    # DETECTION PAGE
    # -----------------------------
    def show_detection(self):
        self.clear_content()
        self.geometry("1200x760")

        base = ctk.CTkFrame(self.content, fg_color="#0b1324", corner_radius=0)
        base.pack(fill="both", expand=True)

        sidebar = ctk.CTkFrame(base, fg_color="#0e1628", corner_radius=0, width=220)
        sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(
            sidebar,
            text="Pix ovex",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#ffffff"
        ).pack(anchor="w", padx=20, pady=(22, 6))

        ctk.CTkLabel(
            sidebar,
            text=f"Logged in as\n{self.current_user}",
            font=ctk.CTkFont(size=13),
            text_color="#8ca0b3",
            justify="left"
        ).pack(anchor="w", padx=20, pady=(0, 20))

        ctk.CTkButton(
            sidebar,
            text="Dashboard",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            command=lambda: self.show_dashboard(self.current_user)
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Detection",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#1b3b5a",
            hover_color="#263247",
            command=self.show_detection
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="History",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="transparent",
            hover_color="#162238",
            command=self.show_history
        ).pack(padx=20, pady=8)

        ctk.CTkButton(
            sidebar,
            text="Logout",
            width=180,
            height=42,
            corner_radius=14,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            command=self.logout
        ).pack(padx=20, pady=(20, 8))

        main = ctk.CTkFrame(base, fg_color="#0b1324", corner_radius=0)
        main.pack(side="left", fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            main,
            text="Live Detection",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w", pady=(10, 10))

        topbar = ctk.CTkFrame(main, fg_color="#20293b", corner_radius=18)
        topbar.pack(fill="x", padx=10, pady=(0, 12))

        left_top = ctk.CTkFrame(topbar, fg_color="transparent")
        left_top.pack(side="left", padx=18, pady=12)

        ctk.CTkLabel(
            left_top,
            text="Connect your laptop camera and detect objects inside this same window.",
            font=ctk.CTkFont(size=14),
            text_color="#cbd5e1"
        ).pack(anchor="w")

        right_top = ctk.CTkFrame(topbar, fg_color="transparent")
        right_top.pack(side="right", padx=18, pady=12)

        ctk.CTkButton(
            right_top,
            text="Start Camera",
            width=140,
            height=38,
            corner_radius=14,
            command=self.start_camera
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            right_top,
            text="Stop Camera",
            width=140,
            height=38,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=self.stop_camera
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            right_top,
            text="Back to Dashboard",
            width=160,
            height=38,
            corner_radius=14,
            fg_color="#334155",
            hover_color="#475569",
            command=lambda: self.show_dashboard(self.current_user)
        ).pack(side="left", padx=6)

        camera_frame = ctk.CTkFrame(main, fg_color="#111827", corner_radius=18, border_width=1, border_color="#1f2a3d")
        camera_frame.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        self.camera_label = tk.Label(camera_frame, bg="#111827")
        self.camera_label.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(
            main,
            text="Tip: Stop Camera before switching pages.",
            font=ctk.CTkFont(size=12),
            text_color="#8ca0b3"
        ).pack(anchor="w", padx=12, pady=(0, 6))

    def start_camera(self):
        if self.camera_running:
            return

        try:
            self.load_detector()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open laptop camera.")
            return

        self.camera_running = True
        self.last_saved_time = {}
        self.update_camera()

    def stop_camera(self):
        self.camera_running = False

        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def update_camera(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after_id = self.after(30, self.update_camera)
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]

        detections = self.detect_fn(input_tensor)
        num_detections = int(self.scalar(detections.pop("num_detections"), float))

        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections.items()
        }

        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        boxes = np.asarray(detections.get("detection_boxes", []))
        classes = np.asarray(detections.get("detection_classes", []))
        scores = np.asarray(detections.get("detection_scores", []))

        if len(boxes) > 0:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_rgb,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.5
            )

        if len(scores) > 0:
            for cls_id, score in zip(classes, scores):
                try:
                    cls_id = self.scalar(cls_id, int)
                    score = self.scalar(score, float)
                except Exception:
                    continue

                if score < 0.50:
                    continue

                label = self.category_index.get(cls_id, {}).get("name", f"class_{cls_id}")
                now = datetime.now().timestamp()

                if (
                    label not in self.last_saved_time
                    or (now - self.last_saved_time[label]) >= 2.0
                ):
                    self.save_history(self.current_user, label, score * 100.0)
                    self.last_saved_time[label] = now

        output_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(output_frame)
        img = img.resize((860, 520))
        photo = ImageTk.PhotoImage(img)

        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

        self.after_id = self.after(30, self.update_camera)

    # -----------------------------
    # LOGOUT / CLOSE
    # -----------------------------
    def logout(self):
        self.current_user = None
        self.stop_camera()
        self.show_home()

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = PixOvexApp()
    app.mainloop()