"""
edu_pipeline GUI 런처
더블클릭으로 실행 → 영상 선택 → 옵션 설정 → 파이프라인 자동 실행
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "input"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

LANGUAGES = {
    "자동 감지": None,
    "한국어": "ko",
    "영어": "en",
    "베트남어": "vi",
    "라오스어": "lo",
}

TARGET_LANGUAGES = {
    "한국어": "ko",
    "영어": "en",
    "베트남어": "vi",
    "라오스어": "lo",
}

STEPS = {
    "처음부터": 1,
    "2. STT부터": 2,
    "3. LLM부터": 3,
    "4. TTS부터": 4,
    "5. 임베딩부터": 5,
}


# ── 스타일 상수 ─────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
BG_INPUT = "#313244"
FG       = "#cdd6f4"
FG_DIM   = "#a6adc8"
ACCENT   = "#89b4fa"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
ORANGE   = "#fab387"
DARK     = "#11111b"
FONT     = "맑은 고딕"


class PipelineLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("edu_pipeline")
        self.geometry("750x650")
        self.resizable(True, True)
        self.configure(bg=BG)
        self._build_ui()
        self._process = None

    # ── UI 구성 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # 제목
        top = tk.Frame(self, bg=BG, pady=10, padx=16)
        top.pack(fill=tk.X)
        tk.Label(top, text="교육 영상 파이프라인", font=(FONT, 16, "bold"),
                 bg=BG, fg=FG).pack(anchor=tk.W)
        tk.Label(top, text="영상을 선택하고 옵션을 설정한 뒤 실행하세요",
                 font=(FONT, 10), bg=BG, fg=FG_DIM).pack(anchor=tk.W, pady=(2, 0))

        # ── 파일 선택 ────────────────────────────────────────────────────────
        file_frame = tk.Frame(self, bg=BG, padx=16, pady=(6, 2))
        file_frame.pack(fill=tk.X)
        tk.Label(file_frame, text="영상 파일", font=(FONT, 9, "bold"),
                 bg=BG, fg=FG_DIM).pack(anchor=tk.W)

        file_row = tk.Frame(file_frame, bg=BG)
        file_row.pack(fill=tk.X, pady=(2, 0))
        self.file_var = tk.StringVar()
        tk.Entry(file_row, textvariable=self.file_var, font=("Consolas", 11),
                 bg=BG_INPUT, fg=FG, insertbackground=FG, relief=tk.FLAT, bd=6
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(file_row, text="찾아보기", font=(FONT, 10, "bold"),
                  bg=ACCENT, fg=BG, relief=tk.FLAT, padx=14, pady=5,
                  cursor="hand2", command=self._browse_file
                  ).pack(side=tk.LEFT, padx=(8, 0))

        # ── 옵션 패널 ────────────────────────────────────────────────────────
        opt_frame = tk.Frame(self, bg=BG, padx=16, pady=(8, 2))
        opt_frame.pack(fill=tk.X)
        tk.Label(opt_frame, text="옵션", font=(FONT, 9, "bold"),
                 bg=BG, fg=FG_DIM).pack(anchor=tk.W, pady=(0, 4))

        opts = tk.Frame(opt_frame, bg=BG)
        opts.pack(fill=tk.X)
        # 4열 그리드
        for i in range(4):
            opts.columnconfigure(i, weight=1 if i % 2 else 0)

        row = 0
        # 원본 언어
        tk.Label(opts, text="원본 언어", font=(FONT, 9), bg=BG, fg=FG
                 ).grid(row=row, column=0, sticky=tk.W, padx=(0, 6))
        self.src_lang_var = tk.StringVar(value="자동 감지")
        src_cb = ttk.Combobox(opts, textvariable=self.src_lang_var,
                              values=list(LANGUAGES.keys()), state="readonly", width=12)
        src_cb.grid(row=row, column=1, sticky=tk.W, padx=(0, 16))

        # 번역 언어
        tk.Label(opts, text="번역 언어", font=(FONT, 9), bg=BG, fg=FG
                 ).grid(row=row, column=2, sticky=tk.W, padx=(0, 6))
        self.tgt_lang_var = tk.StringVar(value="한국어")
        tgt_cb = ttk.Combobox(opts, textvariable=self.tgt_lang_var,
                              values=list(TARGET_LANGUAGES.keys()), state="readonly", width=12)
        tgt_cb.grid(row=row, column=3, sticky=tk.W)

        row = 1
        # 시작 단계
        tk.Label(opts, text="시작 단계", font=(FONT, 9), bg=BG, fg=FG
                 ).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        self.step_var = tk.StringVar(value="처음부터")
        step_cb = ttk.Combobox(opts, textvariable=self.step_var,
                               values=list(STEPS.keys()), state="readonly", width=12)
        step_cb.grid(row=row, column=1, sticky=tk.W, padx=(0, 16), pady=(6, 0))

        # ── 체크박스 ─────────────────────────────────────────────────────────
        chk_frame = tk.Frame(self, bg=BG, padx=16, pady=(8, 2))
        chk_frame.pack(fill=tk.X)

        self.translate_var = tk.BooleanVar(value=False)
        self.skip_tts_var = tk.BooleanVar(value=False)
        self.chat_var = tk.BooleanVar(value=False)

        tk.Checkbutton(chk_frame, text="번역 강제 실행 (원본=목표 언어여도 번역)",
                       variable=self.translate_var, font=(FONT, 9),
                       bg=BG, fg=FG, selectcolor=BG_INPUT, activebackground=BG,
                       activeforeground=FG).pack(anchor=tk.W)
        tk.Checkbutton(chk_frame, text="TTS 더빙 건너뛰기",
                       variable=self.skip_tts_var, font=(FONT, 9),
                       bg=BG, fg=FG, selectcolor=BG_INPUT, activebackground=BG,
                       activeforeground=FG).pack(anchor=tk.W)
        tk.Checkbutton(chk_frame, text="완료 후 RAG 챗봇 실행",
                       variable=self.chat_var, font=(FONT, 9),
                       bg=BG, fg=FG, selectcolor=BG_INPUT, activebackground=BG,
                       activeforeground=FG).pack(anchor=tk.W)

        # ── 실행 / 중지 버튼 ─────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG, pady=8, padx=16)
        btn_frame.pack(fill=tk.X)

        self.run_btn = tk.Button(btn_frame, text="▶  실행", font=(FONT, 12, "bold"),
                                 bg=GREEN, fg=BG, relief=tk.FLAT,
                                 padx=24, pady=8, cursor="hand2",
                                 command=self._run_pipeline)
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(btn_frame, text="■  중지", font=(FONT, 12, "bold"),
                                  bg=RED, fg=BG, relief=tk.FLAT,
                                  padx=24, pady=8, cursor="hand2",
                                  state=tk.DISABLED, command=self._stop_pipeline)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.status_label = tk.Label(btn_frame, text="대기 중", font=(FONT, 10),
                                     bg=BG, fg=FG_DIM)
        self.status_label.pack(side=tk.RIGHT)

        # ── 로그 출력 ────────────────────────────────────────────────────────
        log_frame = tk.Frame(self, bg=BG, padx=16, pady=(2, 12))
        log_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(log_frame, text="실행 로그", font=(FONT, 9, "bold"),
                 bg=BG, fg=FG_DIM).pack(anchor=tk.W, pady=(0, 2))

        self.log = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9),
                                             bg=DARK, fg=FG,
                                             insertbackground=FG,
                                             relief=tk.FLAT, bd=8, wrap=tk.WORD,
                                             state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    # ── 파일 탐색기 ──────────────────────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="교육 영상 파일 선택",
            initialdir=str(INPUT_DIR) if INPUT_DIR.exists() else str(BASE_DIR),
            filetypes=[
                ("영상 파일", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
                ("모든 파일", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)

    # ── 명령어 조립 ──────────────────────────────────────────────────────────
    def _build_cmd(self, filepath):
        cmd = [str(VENV_PYTHON), "main.py", "-i", filepath]

        # 원본 언어
        src = LANGUAGES[self.src_lang_var.get()]
        if src:
            cmd += ["--lang", src]

        # 번역 언어
        tgt = TARGET_LANGUAGES[self.tgt_lang_var.get()]
        cmd += ["--target-lang", tgt]

        # 시작 단계
        step = STEPS[self.step_var.get()]
        if step > 1:
            cmd += ["--start-from", str(step)]

        # 체크박스
        if self.translate_var.get():
            cmd.append("--force-translate")
        if self.skip_tts_var.get():
            cmd.append("--skip-tts")
        if self.chat_var.get():
            cmd.append("--chat")

        return cmd

    # ── 로그 ─────────────────────────────────────────────────────────────────
    def _log_append(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_running(self, running: bool):
        if running:
            self.run_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.status_label.configure(text="실행 중...", fg=GREEN)
        else:
            self.run_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    # ── 실행 ─────────────────────────────────────────────────────────────────
    def _run_pipeline(self):
        filepath = self.file_var.get().strip()
        if not filepath:
            self._log_append("[오류] 파일을 먼저 선택하세요.\n")
            return
        if not Path(filepath).is_file():
            self._log_append(f"[오류] 파일을 찾을 수 없습니다: {filepath}\n")
            return

        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

        cmd = self._build_cmd(filepath)
        self._log_append(f"[실행] {' '.join(cmd)}\n\n")
        self._set_running(True)
        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in self._process.stdout:
                self.after(0, self._log_append, line)
            self._process.wait()

            code = self._process.returncode
            if code == 0:
                self.after(0, self._on_done, True)
            else:
                self.after(0, self._log_append, f"\n[오류] 프로세스 종료 코드: {code}\n")
                self.after(0, self._on_done, False)
        except Exception as e:
            self.after(0, self._log_append, f"\n[오류] {e}\n")
            self.after(0, self._on_done, False)
        finally:
            self._process = None

    def _on_done(self, success):
        self._set_running(False)
        if success:
            self.status_label.configure(text="완료!", fg=GREEN)
        else:
            self.status_label.configure(text="오류 발생", fg=RED)

    def _stop_pipeline(self):
        if self._process:
            self._process.terminate()
            self._log_append("\n[중지됨] 사용자에 의해 중지되었습니다.\n")
            self.status_label.configure(text="중지됨", fg=ORANGE)
            self._set_running(False)


if __name__ == "__main__":
    app = PipelineLauncher()
    app.mainloop()
