"""
Vietnamese Keyword Spotting (KWS) — Interactive Live-Microphone Demo Application.
Bundle Version: Phase 13.4 Final Frozen Candidate v11.

Academic Presentation Prototype.

Mandatory Disclaimers:
- EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
- NOT_VALIDATED_ON_NATURAL_SPEECH
- NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
- NOT_VALIDATED_PENDING_DATA
"""

import os
import sys
import io
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project and demo paths in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEMO_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v13_4" / "bundle"

for p in [str(DEMO_DIR), str(BUNDLE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from live_mic_engine import (
    LiveKWSPipeline, AudioStreamManager, DetectionEvent,
    CLASSES, TARGET_CLASSES, DEFAULT_THRESHOLD
)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

DISCLAIMER_TEXT = (
    "EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED | "
    "NOT_VALIDATED_ON_NATURAL_SPEECH | "
    "NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION"
)

EXPECTED_V11_CKPT_HASH = "3c4369d2f95059b973ba38ab5f2fd5aa611c7ef3322156b95be21278d32a60a2"
EXPECTED_V11_ONNX_HASH = "54298c503f51c8299d86f6b4346d7cae4e1b7e045deb884a8650764fbe4bcc99"
EXPECTED_E1_ONNX_HASH = "20ecc2964d88d5a9d1d15b1c665e9697a32d702b1e3dd12ab1e5f829aff016b8"
EXPECTED_E1_CKPT_HASH = "f3c0fe8f7ccebb8fe354cb50acb247e0674232044706db0187feb92b1706b878"
EXPECTED_VAD_CONFIG_HASH = "af5c3891fe73f831bb2972f2ce4722b3cbb6b669ad306bd408b8e941ee7de9b3"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def log_event_to_file(event: DetectionEvent, log_path: Path):
    """Appends structured metadata of detection event without raw audio."""
    log_entry = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(event.timestamp)),
        "time_str": event.time_str,
        "command": event.command,
        "confidence": event.confidence,
        "latency_ms": event.latency_ms,
        "vad_active": event.vad_active,
        "db_level": event.db_level,
        "segment_duration_ms": event.segment_duration_ms,
        "probabilities": event.probabilities,
        "profile_name": getattr(event, "profile_name", "legacy_v11"),
        "model_artifact_hash": getattr(event, "model_artifact_hash", ""),
        "vad_config_hash": getattr(event, "vad_config_hash", ""),
        "vad_algorithm": getattr(event, "vad_algorithm", ""),
        "segment_start_sec": getattr(event, "segment_start_sec", 0.0),
        "segment_end_sec": getattr(event, "segment_end_sec", 0.0)
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write event log: {e}", file=sys.stderr)


def render_rich_dashboard(pipeline: LiveKWSPipeline) -> Panel:
    """Renders real-time telemetry into a Rich Panel."""
    st = pipeline.get_status_dict()
    pred = st["last_prediction"]
    probs = pred["probabilities"]

    # Header / Meta
    header_text = Text()
    if st.get("profile") == "experimental_e1_speechband":
        header_text.append("[ HỒ SƠ B THỬ NGHIỆM: E1 ONNX + SPEECHBAND VAD (PHASE 18) ]\n", style="bold yellow")
        header_text.append(f"Mã quản trị: {DISCLAIMER_TEXT}\n", style="dim italic red")
        header_text.append(
            f"Mô hình: E1 ONNX ({st.get('model_hash', '')}) | VAD: {st.get('vad_algorithm', '')} ({st.get('vad_config_hash', '')})\n",
            style="bold yellow"
        )
    else:
        header_text.append("[ DEMO TRỰC TIẾP MICRO LAPTOP — NHẬN DIỆN KHẨU LỆNH TIẾNG VIỆT ]\n", style="bold cyan")
        header_text.append(f"Mã quản trị: {DISCLAIMER_TEXT}\n", style="dim italic red")

    mode_badge = (
        "[bold green]CHẾ ĐỘ PHÂN ĐOẠN KHẨU LỆNH (SEGMENTED - KHUYÊN DÙNG)[/]"
        if st["mode"] == "segmented"
        else "[bold yellow]CHẾ ĐỘ STREAMING TRƯỢT 160ms (SLIDING WINDOW)[/]"
    )
    header_text.append(
        f"Thiết bị: [{st['device_id']}] {st['device_name']} ({st['device_api']}) | "
        f"Backend: {st['backend'].upper()} | Ngưỡng theta: {st['theta_cmd']:.2f} | Chế độ: {mode_badge}\n",
        style="dim white"
    )

    # Audio Level VU Meter
    db = st["current_rms_db"]
    norm_db = max(0.0, min(1.0, (db + 60.0) / 60.0))
    bar_width = 30
    filled = int(norm_db * bar_width)
    empty = bar_width - filled

    if norm_db > 0.8:
        vu_color = "red"
    elif norm_db > 0.4:
        vu_color = "yellow"
    else:
        vu_color = "green"

    vu_bar = f"[{vu_color}]{'█' * filled}[/][dim white]{'░' * empty}[/]"

    if st["mode"] == "segmented":
        state_badge = (
            "[bold yellow][ ĐANG THU LỆNH / ACCUMULATING ][/]"
            if st["segmenter_state"] == "ACCUMULATING"
            else "[dim green][ CHỜ TIẾNG NÓI / WAITING ][/]"
        )
    else:
        state_badge = "[bold yellow][ ACTIVE SPEECH ][/]" if st["vad_active"] else "[dim][ SILENCE ][/]"

    vu_text = Text.from_markup(f"Cường độ âm thanh: {vu_bar} {db:5.1f} dB  {state_badge}")

    # Probabilities Table
    table = Table(title="Phân Phối Xác Suất Lệnh Vừa Xử Lý (1.0s Window)", expand=True, box=None)
    table.add_column("Lớp (Class)", justify="left", style="bold")
    table.add_column("Thanh Xác Suất", justify="left")
    table.add_column("Xác Suất", justify="right")
    table.add_column("Trạng Thái", justify="center")

    bar_len = 25
    for c in CLASSES:
        p = probs.get(c, 0.0)
        p_filled = int(p * bar_len)
        p_empty = bar_len - p_filled

        if c in TARGET_CLASSES:
            if p >= st["theta_cmd"]:
                bar_markup = f"[bold green]{'█' * p_filled}[/][dim]{'░' * p_empty}[/]"
                status_markup = "[bold green]KÍCH HOẠT[/]"
                c_style = "bold green"
            else:
                bar_markup = f"[cyan]{'█' * p_filled}[/][dim]{'░' * p_empty}[/]"
                status_markup = "[dim]DƯỚI NGƯỠNG[/]"
                c_style = "white"
        else:  # REJECT
            bar_markup = f"[dim magenta]{'█' * p_filled}[/][dim]{'░' * p_empty}[/]"
            status_markup = "[magenta]BỎ QUA[/]" if p >= 0.5 else "[dim]BỎ QUA[/]"
            c_style = "dim magenta"

        table.add_row(
            Text(c, style=c_style),
            Text.from_markup(bar_markup),
            f"{p * 100:5.1f}%",
            Text.from_markup(status_markup)
        )

    # Latest Event Alert Panel
    if st["recent_events"]:
        last_ev: DetectionEvent = st["recent_events"][-1]
        time_elapsed = time.time() - last_ev.timestamp
        dur_info = f" | Độ dài: {last_ev.segment_duration_ms:.0f} ms" if last_ev.segment_duration_ms > 0 else ""
        if time_elapsed < 3.0:
            ev_panel = Panel(
                f"[bold white on green]  >>> PHÁT HIỆN LỆNH: [ {last_ev.command} ]  <<<  [/]\n"
                f"Độ tin cậy: [bold]{last_ev.confidence * 100:.1f}%[/] | Thời điểm: {last_ev.time_str} | "
                f"Độ trễ suy luận: {last_ev.latency_ms:.1f} ms{dur_info}",
                style="bold green",
                title="SỰ KIỆN KÍCH HOẠT MỚI NHẤT"
            )
        else:
            ev_panel = Panel(
                f"Lệnh vừa qua: [bold cyan]{last_ev.command}[/] ({last_ev.confidence * 100:.1f}%) lúc {last_ev.time_str}{dur_info}",
                style="dim",
                title="LỊCH SỬ KÍCH HOẠT"
            )
    else:
        ev_panel = Panel(
            "[dim italic]Chưa có khẩu lệnh nào được kích hoạt. Hãy hướng về micro và nói rõ ràng từng từ: TRÁI, PHẢI, LÊN, XUỐNG (nghỉ 300ms giữa các từ).[/]",
            style="dim",
            title="TRẠNG THÁI CHỜ LỆNH"
        )

    # Recent Events List
    events_summary = Text()
    if st["recent_events"]:
        events_summary.append("Lịch sử các sự kiện gần nhất (Thứ tự kích hoạt thực tế):\n", style="bold underline")
        for ev in st["recent_events"][-5:]:
            dur_txt = f"[{ev.segment_duration_ms:.0f}ms]" if ev.segment_duration_ms > 0 else ""
            events_summary.append(
                f"  • [{ev.time_str}] Lệnh: {ev.command:<6} (conf: {ev.confidence:.3f}, trễ: {ev.latency_ms:.1f}ms {dur_txt})\n",
                style="cyan"
            )
    else:
        events_summary.append("(Lịch sử sự kiện trống)", style="dim")

    footer_text = Text.from_markup(
        f"[dim]Chế độ: [bold]{st['mode'].upper()}[/] | Độ trễ suy luận: {pred['latency_ms']:.1f} ms | "
        f"Khoảng lặng ngắt lệnh: {st['silence_interval_ms']} ms | "
        f"Nhấn [bold red]Ctrl+C[/] để dừng an toàn.[/]"
    )

    content = Layout()
    content.split_column(
        Layout(header_text, size=4),
        Layout(vu_text, size=2),
        Layout(table, size=9),
        Layout(ev_panel, size=4),
        Layout(events_summary, size=6),
        Layout(footer_text, size=2)
    )

    return Panel(content, border_style="cyan", title="VIETNAMESE KWS CNN — LIVE DEMO")


def run_terminal_dashboard(pipeline: LiveKWSPipeline, max_duration_sec: Optional[float] = None):
    """Runs interactive live terminal dashboard."""
    console = Console()
    pipeline.start()

    start_time = time.time()
    try:
        if HAS_RICH:
            with Live(render_rich_dashboard(pipeline), refresh_per_second=8, console=console) as live:
                while pipeline.is_running:
                    live.update(render_rich_dashboard(pipeline))
                    time.sleep(0.12)
                    if max_duration_sec and (time.time() - start_time) >= max_duration_sec:
                        break
        else:
            print("\n" + "=" * 70)
            print("  VIETNAMESE KWS LIVE DEMO (Simple CLI Mode)")
            print("=" * 70)
            while pipeline.is_running:
                st = pipeline.get_status_dict()
                pred = st["last_prediction"]
                print(
                    f"\rLevel: {st['current_rms_db']:5.1f} dB | Mode: {st['mode']} | State: {st['segmenter_state']} | "
                    f"Pred: {pred['predicted_class']:6s} ({pred['confidence']:.2f}) | Events: {len(st['recent_events'])}",
                    end="", flush=True
                )
                time.sleep(0.15)
                if max_duration_sec and (time.time() - start_time) >= max_duration_sec:
                    break
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Đã nhận tín hiệu dừng (Ctrl+C). Đang giải phóng thiết bị audio...[/]")
    finally:
        pipeline.stop()
        console.print("[bold green]Microphone đã được đóng an toàn. Phiên demo kết thúc thành công.[/]")


def run_tkinter_gui(pipeline: LiveKWSPipeline):
    """Runs a graphical desktop window using built-in Tkinter."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    st_init = pipeline.get_status_dict()
    is_exp = (st_init.get("profile") == "experimental_e1_speechband")
    win_title = "Vietnamese KWS CNN — Live Microphone Demo (Profile B Experimental)" if is_exp else "Vietnamese KWS CNN — Live Microphone Demo"
    root.title(win_title)
    root.geometry("640x640")
    root.resizable(False, False)
    root.configure(bg="#0f172a")

    title_text = "VIETNAMESE KWS CNN — PROFILE B (EXPERIMENTAL)" if is_exp else "VIETNAMESE KEYWORD SPOTTING CNN"
    title_fg = "#f59e0b" if is_exp else "#38bdf8"
    title_lbl = tk.Label(
        root, text=title_text,
        font=("Segoe UI", 15, "bold"), fg=title_fg, bg="#0f172a"
    )
    title_lbl.pack(pady=(14, 2))

    banner_lbl = tk.Label(
        root, text=DISCLAIMER_TEXT,
        font=("Segoe UI", 8, "italic"), fg="#f87171", bg="#0f172a"
    )
    banner_lbl.pack(pady=(0, 8))

    info_frame = tk.Frame(root, bg="#1e293b", padx=12, pady=8)
    info_frame.pack(fill="x", padx=20, pady=4)

    prof_str = f"Profile: {st_init.get('profile', 'legacy_v11')} | Model: {st_init.get('model_hash', '')} | VAD: {st_init.get('vad_algorithm', '')} ({st_init.get('vad_config_hash', '')})"
    dev_str = f"Device: [{st_init['device_id']}] {st_init['device_name']} | Mode: {st_init['mode'].upper()} | Backend: {st_init['backend'].upper()}\n{prof_str}"
    dev_lbl = tk.Label(info_frame, text=dev_str, font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b", justify="left")
    dev_lbl.pack(anchor="w")

    vu_frame = tk.Frame(root, bg="#1e293b", padx=12, pady=8)
    vu_frame.pack(fill="x", padx=20, pady=6)

    vu_lbl = tk.Label(vu_frame, text="Audio Level: -60.0 dB", font=("Segoe UI", 10, "bold"), fg="#f8fafc", bg="#1e293b")
    vu_lbl.pack(anchor="w")

    vu_canvas = tk.Canvas(vu_frame, width=580, height=16, bg="#334155", highlightthickness=0)
    vu_canvas.pack(pady=4)

    state_lbl = tk.Label(vu_frame, text="State: WAITING FOR SPEECH", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
    state_lbl.pack(anchor="w")

    event_box = tk.Frame(root, bg="#1e293b", relief="groove", bd=2, padx=12, pady=12)
    event_box.pack(fill="x", padx=20, pady=8)

    event_title = tk.Label(event_box, text="LỆNH ĐƯỢC KÍCH HOẠT MỚI NHẤT", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#1e293b")
    event_title.pack()

    event_display = tk.Label(event_box, text="[ CHỜ LỆNH ]", font=("Segoe UI", 26, "bold"), fg="#94a3b8", bg="#1e293b")
    event_display.pack(pady=4)

    event_sub = tk.Label(event_box, text="Hãy nói từng lệnh: TRÁI, PHẢI, LÊN, XUỐNG (nghỉ 300ms)", font=("Segoe UI", 9), fg="#64748b", bg="#1e293b")
    event_sub.pack()

    prob_frame = tk.Frame(root, bg="#1e293b", padx=12, pady=8)
    prob_frame.pack(fill="x", padx=20, pady=6)

    prob_bars = {}
    prob_labels = {}
    for c in CLASSES:
        row = tk.Frame(prob_frame, bg="#1e293b")
        row.pack(fill="x", pady=2)
        c_lbl = tk.Label(row, text=f"{c:<8}", font=("Consolas", 10, "bold"), fg="#f8fafc", bg="#1e293b", width=8, anchor="w")
        c_lbl.pack(side="left")

        bar = ttk.Progressbar(row, orient="horizontal", length=380, mode="determinate")
        bar.pack(side="left", padx=8)
        prob_bars[c] = bar

        val_lbl = tk.Label(row, text="0.0%", font=("Consolas", 10), fg="#94a3b8", bg="#1e293b", width=6, anchor="e")
        val_lbl.pack(side="left")
        prob_labels[c] = val_lbl

    btn_frame = tk.Frame(root, bg="#0f172a")
    btn_frame.pack(fill="x", padx=20, pady=12)

    is_running_var = tk.BooleanVar(value=False)

    def toggle_stream():
        if is_running_var.get():
            pipeline.stop()
            is_running_var.set(False)
            btn_toggle.config(text="BẮT ĐẦU MICRO (START)", bg="#22c55e")
            event_sub.config(text="Microphone đã dừng.")
        else:
            pipeline.start()
            is_running_var.set(True)
            btn_toggle.config(text="DỪNG MICRO (STOP)", bg="#ef4444")
            event_sub.config(text="Hãy nói từng lệnh: TRÁI, PHẢI, LÊN, XUỐNG (nghỉ 300ms)")

    btn_toggle = tk.Button(
        btn_frame, text="BẮT ĐẦU MICRO (START)", font=("Segoe UI", 11, "bold"),
        fg="#ffffff", bg="#22c55e", activebackground="#16a34a", padx=16, pady=8,
        command=toggle_stream
    )
    btn_toggle.pack(side="left", expand=True, fill="x", padx=4)

    def on_closing():
        if is_running_var.get():
            pipeline.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    def update_gui():
        if is_running_var.get():
            st = pipeline.get_status_dict()
            pred = st["last_prediction"]
            probs = pred["probabilities"]

            db = st["current_rms_db"]
            vu_lbl.config(text=f"Audio Level: {db:5.1f} dB (Peak: {st['peak_amp']:.3f})")
            norm = max(0.0, min(1.0, (db + 60.0) / 60.0))
            vu_canvas.delete("all")
            fill_w = int(norm * 580)
            fill_color = "#ef4444" if norm > 0.8 else ("#f59e0b" if norm > 0.4 else "#22c55e")
            vu_canvas.create_rectangle(0, 0, fill_w, 16, fill=fill_color, width=0)

            if st["mode"] == "segmented":
                txt = "State: ACCUMULATING SPEECH" if st["segmenter_state"] == "ACCUMULATING" else "State: WAITING FOR SPEECH"
                fg = "#f59e0b" if st["segmenter_state"] == "ACCUMULATING" else "#22c55e"
            else:
                txt = "VAD Status: ACTIVE SPEECH" if st["vad_active"] else "VAD Status: SILENCE"
                fg = "#f59e0b" if st["vad_active"] else "#64748b"
            state_lbl.config(text=txt, fg=fg)

            for c in CLASSES:
                p = probs.get(c, 0.0)
                prob_bars[c]["value"] = p * 100
                prob_labels[c].config(text=f"{p * 100:5.1f}%")

            if st["recent_events"]:
                last_ev = st["recent_events"][-1]
                time_diff = time.time() - last_ev.timestamp
                dur_str = f" | {last_ev.segment_duration_ms:.0f}ms" if last_ev.segment_duration_ms > 0 else ""
                if time_diff < 3.0:
                    event_display.config(text=f"LỆNH: {last_ev.command}", fg="#22c55e")
                    event_sub.config(
                        text=f"Độ tin cậy: {last_ev.confidence * 100:.1f}% | Lúc {last_ev.time_str} | Trễ: {last_ev.latency_ms:.1f}ms{dur_str}",
                        fg="#22c55e"
                    )
                else:
                    event_display.config(text=f"[ {last_ev.command} ]", fg="#38bdf8")
                    event_sub.config(text=f"Lệnh vừa qua lúc {last_ev.time_str}{dur_str}", fg="#94a3b8")

        root.after(100, update_gui)

    root.after(100, update_gui)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Vietnamese KWS CNN Live Microphone Demo (Academic Review)")
    parser.add_argument("--profile", choices=["legacy_v11", "experimental_e1_speechband"], default="legacy_v11",
                        help="Execution profile: 'legacy_v11' (default frozen candidate) or 'experimental_e1_speechband' (Phase 18 opt-in)")
    parser.add_argument("--mode", choices=["segmented", "streaming"], default="segmented",
                        help="Operating mode: 'segmented' (default, highly recommended for multi-command sequences) or 'streaming' (legacy sliding-window)")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device ID")
    parser.add_argument("--list-devices", action="store_true", help="List all available audio input devices and exit")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx", help="Inference backend (default: onnx)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Command threshold (default: 0.20)")
    parser.add_argument("--silence-ms", type=int, default=300, help="Trailing silence interval in ms to finalize segment (default: 300, test range 250-350)")
    parser.add_argument("--preroll-ms", type=int, default=150, help="Pre-roll audio buffer in ms before speech onset (default: 150)")
    parser.add_argument("--energy-db", type=float, default=-48.0, help="Speech onset energy threshold in dB (default: -48.0)")
    parser.add_argument("--vad", choices=["formant", "energy", "none"], default="formant", help="VAD gating algorithm")
    parser.add_argument("--gui", action="store_true", help="Launch Tkinter desktop GUI")
    parser.add_argument("--log", type=Path, default=DEMO_DIR / "live_demo_events.log", help="Path to event log file")
    parser.add_argument("--duration", type=float, default=None, help="Optional maximum run time in seconds")

    args = parser.parse_args()

    if args.list_devices:
        print("\n=== CÁC THIẾT BỊ MICROPHONE KHẢ DỤNG TRÊN HỆ THỐNG ===")
        devs = AudioStreamManager.list_input_devices()
        for d in devs:
            print(f"ID [{d['id']:2d}]: {d['name']} (API: {d['hostapi']}, Kênh: {d['channels']}, Tần số gốc: {d['default_sr']} Hz)")
        print("\nSử dụng tham số `--device <ID>` để chỉ định thiết bị mong muốn.")
        return

    # Pre-capture integrity and artifact validation
    v11_ckpt_path = BUNDLE_DIR / "checkpoint" / "variant_v11_hard_negative_weighted_seed42.pt"
    v11_onnx_path = BUNDLE_DIR / "onnx" / "vietnamese_kws_v11.onnx"
    e1_onnx_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "onnx" / "vietnamese_kws_v12_e1.onnx"
    e1_ckpt_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "checkpoints" / "exp_v12_e1_baseline_scale_seed42.pt"
    vad_config_path = PROJECT_ROOT / "experiments" / "vieneu_kws_research_v12" / "configs" / "selected_phase15_vad_config.json"

    if args.profile == "legacy_v11":
        if args.backend == "onnx":
            if not v11_onnx_path.exists():
                print(f"[ERROR] Legacy ONNX model missing: {v11_onnx_path}", file=sys.stderr)
                sys.exit(1)
            h = compute_file_sha256(v11_onnx_path)
            if h != EXPECTED_V11_ONNX_HASH:
                print(f"[ERROR] Legacy ONNX hash mismatch!\nExpected: {EXPECTED_V11_ONNX_HASH}\nActual:   {h}", file=sys.stderr)
                sys.exit(1)
            model_target_path = v11_onnx_path
            model_hash_val = h
        else:
            if not v11_ckpt_path.exists():
                print(f"[ERROR] Legacy Checkpoint missing: {v11_ckpt_path}", file=sys.stderr)
                sys.exit(1)
            h = compute_file_sha256(v11_ckpt_path)
            if h != EXPECTED_V11_CKPT_HASH:
                print(f"[ERROR] Legacy Checkpoint hash mismatch!\nExpected: {EXPECTED_V11_CKPT_HASH}\nActual:   {h}", file=sys.stderr)
                sys.exit(1)
            model_target_path = v11_ckpt_path
            model_hash_val = h
        vad_cfg_target = None
        vad_hash_val = ""
    else:  # experimental_e1_speechband
        if args.backend == "onnx":
            if not e1_onnx_path.exists():
                print(f"[ERROR] Experimental Profile B ONNX model missing: {e1_onnx_path}", file=sys.stderr)
                print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
                sys.exit(1)
            h = compute_file_sha256(e1_onnx_path)
            if h != EXPECTED_E1_ONNX_HASH:
                print(f"[ERROR] Experimental ONNX hash mismatch!\nExpected: {EXPECTED_E1_ONNX_HASH}\nActual:   {h}", file=sys.stderr)
                print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
                sys.exit(1)
            model_target_path = e1_onnx_path
            model_hash_val = h
        else:
            if not e1_ckpt_path.exists():
                print(f"[ERROR] Experimental Profile B PyTorch checkpoint missing: {e1_ckpt_path}", file=sys.stderr)
                print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
                sys.exit(1)
            h = compute_file_sha256(e1_ckpt_path)
            if h != EXPECTED_E1_CKPT_HASH:
                print(f"[ERROR] Experimental PyTorch hash mismatch!\nExpected: {EXPECTED_E1_CKPT_HASH}\nActual:   {h}", file=sys.stderr)
                print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
                sys.exit(1)
            model_target_path = e1_ckpt_path
            model_hash_val = h

        if not vad_config_path.exists():
            print(f"[ERROR] Selected Phase 15 VAD config missing: {vad_config_path}", file=sys.stderr)
            print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
            sys.exit(1)
        vad_h = compute_file_sha256(vad_config_path)
        if vad_h != EXPECTED_VAD_CONFIG_HASH:
            print(f"[ERROR] Selected Phase 15 VAD config hash mismatch!\nExpected: {EXPECTED_VAD_CONFIG_HASH}\nActual:   {vad_h}", file=sys.stderr)
            print("[ERROR] Terminating immediately. Fallback to v11 is strictly prohibited in experimental mode.", file=sys.stderr)
            sys.exit(1)
        vad_cfg_target = vad_config_path
        vad_hash_val = vad_h

    # Initialize Pipeline
    def on_detection_event(event: DetectionEvent):
        log_event_to_file(event, args.log)

    pipeline = LiveKWSPipeline(
        mode=args.mode,
        profile=args.profile,
        backend=args.backend,
        model_path=model_target_path,
        vad_config_path=vad_cfg_target,
        model_hash=model_hash_val,
        vad_config_hash=vad_hash_val,
        theta_cmd=args.threshold,
        device_id=args.device,
        vad_mode=args.vad,
        silence_interval_ms=args.silence_ms,
        preroll_ms=args.preroll_ms,
        energy_threshold_db=args.energy_db,
        event_callback=on_detection_event
    )

    if args.gui:
        run_tkinter_gui(pipeline)
    else:
        run_terminal_dashboard(pipeline, max_duration_sec=args.duration)


if __name__ == "__main__":
    main()
