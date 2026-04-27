from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.boss_capture import (
    build_capture_key,
    CAPTURE_COLUMNS,
    delete_captured_job,
    DEFAULT_CAPTURE_FILE,
    apply_jd_structuring_to_captured_row,
    extract_current_job_detail,
    format_tab_label,
    get_tab_html,
    import_captured_jobs,
    list_debug_tabs,
    load_captured_jobs,
    save_captured_job_with_status,
)
from modules.llm_settings import get_task_llm_config, render_page_task_llm_settings


CHROME_START_COMMAND = (
    '"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
    "--remote-debugging-port=9222 "
    "--remote-allow-origins=* "
    '--user-data-dir="C:\\selenium\\manual_chrome"'
)


def _read_import_file(uploaded_file) -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, dtype=str, encoding="utf-8-sig").fillna("")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, dtype=str, encoding="gb18030").fillna("")

    return pd.read_excel(uploaded_file, dtype=str).fillna("")


def _get_debug_tabs(debug_port: int) -> tuple[list[dict], str]:
    try:
        return list_debug_tabs(int(debug_port)), ""
    except Exception as exc:
        return [], str(exc)


def _render_setup_panel(debug_port: int, tab_count: int, connection_error: str):
    st.subheader("采集状态")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("调试端口", debug_port)
    c2.metric("可读标签页", tab_count)
    if connection_error:
        c3.warning(f"未连接到 Chrome：{connection_error}")
    else:
        c3.success("Chrome 调试端口已连接")

    with st.expander("浏览器启动与格式说明", expanded=False):
        st.markdown("先用下面的命令启动 Chrome，再在该浏览器里登录 BOSS 直聘并打开搜索页或岗位详情页。")
        st.code(CHROME_START_COMMAND, language="powershell")
        st.markdown("采集会读取当前选中的岗位详情，保存前会清洗详情文本、提取标签，并自动去重。")


def _render_hotkey_section(debug_port: int, selected_tab: dict | None = None):
    st.subheader("快捷键监听")
    st.caption("适合连续浏览岗位：监听启动后按 Ctrl+Shift+Z 抓取并保存；按 Ctrl+Shift+X 停止监听。每次按键都会重新读取 Chrome 当前可用的 BOSS 页面，不再固定绑定上方选中的标签。")

    project_root = Path(__file__).resolve().parents[1]
    log_path = project_root / "data" / "boss_jobs" / "hotkey_listener.log"
    command = [
        sys.executable,
        "-m",
        "modules.boss_capture",
        "--hotkeys",
        "--debug-port",
        str(int(debug_port)),
    ]
    auto_llm_enabled = bool(st.session_state.get("boss_capture_auto_llm_recognition", False))
    if auto_llm_enabled:
        command.append("--auto-llm-struct")
    cmd_text = " ".join(f'"{part}"' if " " in part else part for part in command)
    st.code(cmd_text, language="powershell")

    proc = st.session_state.get("boss_hotkey_listener_proc")
    is_running = proc is not None and proc.poll() is None
    left, right = st.columns([1, 2])
    with left:
        if st.button("启动快捷键监听", disabled=is_running, use_container_width=True):
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(log_path, "a", encoding="utf-8")
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                proc = subprocess.Popen(
                    command,
                    cwd=project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                st.session_state["boss_hotkey_listener_proc"] = proc
                st.success("快捷键监听已启动。")
            except Exception as exc:
                st.error(f"启动失败：{exc}")
    with right:
        if is_running:
            st.success("监听中：Ctrl+Shift+Z 抓取并保存，Ctrl+Shift+X 退出。")
            st.caption("每次抓取都会重新读取 Chrome 当前可用的 BOSS 页面内容。")
            if auto_llm_enabled:
                st.caption("自动识别已随本次监听启动：Ctrl+Shift+Z 抓取后会直接执行 JD 结构化识别并保存。")
        else:
            st.info(f"未启动。日志位置：{log_path}")
            st.caption("启动后不会固定读取上方选中的标签。切换 BOSS 页面或岗位后，直接按 Ctrl+Shift+Z 即可抓取当时页面内容。")
            if auto_llm_enabled:
                st.caption("自动识别开关已开启。启动监听后，快捷键抓取也会自动执行 JD 结构化识别。")

    captured_count = len(load_captured_jobs())
    st.caption(f"当前采集库：{captured_count} 条。快捷键保存后如页面未自动变化，请刷新采集库或重新加载页面。")

    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-12:])
        except Exception as exc:
            tail = f"读取日志失败：{exc}"
        st.text_area("快捷键监听日志", value=tail, height=180, disabled=True)


def _render_browser_tab_selector(tabs: list[dict], connection_error: str) -> dict | None:
    st.markdown(
        """
        <div style="
            border: 1px solid rgba(250, 250, 250, 0.18);
            border-radius: 12px;
            padding: 18px 20px 16px;
            margin: 4px 0 18px;
            background: rgba(60, 80, 110, 0.18);
        ">
            <div style="font-size: 13px; color: rgba(250,250,250,0.62); margin-bottom: 4px;">当前读取对象</div>
            <div style="font-size: 28px; font-weight: 800; line-height: 1.25;">读取的浏览器标签</div>
            <div style="font-size: 15px; color: rgba(250,250,250,0.68); margin-top: 8px;">
                先确认要读取哪个 Chrome 标签页。下面的手动抓取和快捷键监听都围绕这个标签来源工作。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_col, refresh_col = st.columns([3, 1])
    with refresh_col:
        if st.button("刷新标签列表", use_container_width=True):
            st.rerun()

    if connection_error:
        st.warning("请先确认 Chrome 已用调试端口启动。")
        return None
    if not tabs:
        st.info("当前没有可读取的 Chrome 页面标签。")
        return None

    boss_tabs = [tab for tab in tabs if "zhipin.com" in str(tab.get("url", ""))]
    display_tabs = boss_tabs or tabs
    with status_col:
        if boss_tabs:
            st.success(f"已找到 {len(boss_tabs)} 个 BOSS 直聘标签，可读标签共 {len(tabs)} 个。")
        else:
            st.info(f"未识别到 BOSS 直聘标签，当前展示全部 {len(tabs)} 个可读标签。")

    labels = [format_tab_label(tab) for tab in display_tabs]
    selected_label = st.selectbox("读取的浏览器标签", labels, index=0)
    selected_tab = display_tabs[labels.index(selected_label)]
    st.markdown(f"**当前标签 URL：** {selected_tab.get('url', '')}")
    return selected_tab


def _render_manual_capture_section(selected_tab: dict | None):
    st.subheader("手动抓取")
    st.caption("适合单次确认：点击后读取上方选中的浏览器标签，并自动写入采集库。")

    if selected_tab is None:
        st.button("抓取当前岗位并自动入库", type="primary", use_container_width=True, disabled=True)
        st.info("先在上方确认可读取的浏览器标签。")
        return

    if st.button("抓取当前岗位并自动入库", type="primary", use_container_width=True):
        try:
            title, url, html = get_tab_html(selected_tab)
            row = extract_current_job_detail(html, url)
            auto_llm_enabled = bool(st.session_state.get("boss_capture_auto_llm_recognition", False))
            if auto_llm_enabled:
                with st.spinner("正在执行 JD 结构化 LLM 识别..."):
                    row = apply_jd_structuring_to_captured_row(row, get_task_llm_config("jd_struct"))
            saved, save_status = save_captured_job_with_status(row)
            st.session_state["boss_capture_last_row"] = row
            st.session_state["boss_capture_last_save_status"] = save_status
            action_text = "已更新重复岗位" if save_status == "updated" else "已新增岗位"
            llm_text = "，已自动完成 LLM 识别" if auto_llm_enabled else ""
            st.success(f"{action_text}{llm_text}，当前采集库共 {len(saved)} 条。页面：{title or url}")
        except Exception as exc:
            st.error(f"抓取失败：{exc}")


def _build_llm_preview_rows(row: dict) -> pd.DataFrame:
    preview_cols = [
        "职位名称",
        "企业名称",
        "所在地区",
        "薪资解析",
        "LLM核心目标",
        "LLM所属行业",
        "LLM岗位类型",
        "LLM岗位工作内容",
        "LLM岗位要求",
        "LLM加分项",
        "LLM必须技能",
        "LLM加分技能",
        "LLM工具栈",
        "LLM学历要求",
        "LLM经验要求",
        "LLM应届友好度",
        "LLM资历级别",
        "LLM结构化提取错误",
    ]
    rows = []
    for col in preview_cols:
        if col not in row:
            continue
        value = row.get(col)
        if isinstance(value, list):
            value = "，".join(str(item) for item in value if str(item).strip())
        if str(value or "").strip():
            rows.append({"字段": col, "识别结果": value})
    return pd.DataFrame(rows)


def _render_llm_recognition_section(selected_tab: dict | None):
    st.subheader("当前岗位 LLM 识别")
    st.caption("读取上方选中的浏览器标签，对当前岗位做 JD 结构化识别；也可以开启抓取后自动识别，直接把识别结果写入采集库。")

    render_page_task_llm_settings(["jd_struct"], title="当前岗位 LLM 设置", include_switches=False)
    llm_config = get_task_llm_config("jd_struct")
    provider = str(llm_config.get("provider") or "local").lower()
    model_label = llm_config.get("remote_model") if provider == "remote" else llm_config.get("local_model")
    st.caption(f"当前识别模型：{provider} / {model_label}")
    st.toggle(
        "抓取后自动执行 LLM 识别并保存",
        value=bool(st.session_state.get("boss_capture_auto_llm_recognition", False)),
        key="boss_capture_auto_llm_recognition",
        help="开启后，点击“抓取当前岗位并自动入库”会先抓取、再调用当前 JD 结构化模型识别，最后保存识别后的结果。快捷键监听需重新启动后才会带上这个开关。",
    )

    if selected_tab is None:
        st.button("识别当前岗位", type="primary", use_container_width=True, disabled=True)
        st.info("先在上方确认可读取的浏览器标签。")
    elif st.button("识别当前岗位", type="primary", use_container_width=True, key="llm_recognize_current_job"):
        try:
            _, url, html = get_tab_html(selected_tab)
            row = extract_current_job_detail(html, url)
            recognized_row = apply_jd_structuring_to_captured_row(row, llm_config)
            st.session_state["boss_capture_llm_row"] = recognized_row
            st.success("当前岗位 LLM 识别完成。")
        except Exception as exc:
            st.error(f"LLM 识别失败：{exc}")

    llm_row = st.session_state.get("boss_capture_llm_row")
    if isinstance(llm_row, dict) and llm_row:
        st.markdown("##### 识别结果预览")
        preview_df = _build_llm_preview_rows(llm_row)
        if preview_df.empty:
            st.info("当前没有可展示的 LLM 识别字段。")
        else:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

        save_col, clear_col = st.columns([1, 1])
        with save_col:
            if st.button("保存识别结果到采集库", use_container_width=True, key="save_llm_recognized_job"):
                saved, save_status = save_captured_job_with_status(llm_row)
                st.session_state["boss_capture_last_row"] = llm_row
                st.session_state["boss_capture_last_save_status"] = save_status
                action_text = "已更新重复岗位" if save_status == "updated" else "已新增岗位"
                st.success(f"{action_text}，当前采集库共 {len(saved)} 条。")
        with clear_col:
            if st.button("清空当前识别结果", use_container_width=True, key="clear_llm_recognized_job"):
                st.session_state.pop("boss_capture_llm_row", None)
                st.rerun()


def _render_last_result():
    row = st.session_state.get("boss_capture_last_row")
    if not isinstance(row, dict) or not row:
        return

    st.subheader("最近抓取结果")
    save_status = st.session_state.get("boss_capture_last_save_status")
    if save_status == "updated":
        st.info("该岗位与采集库已有记录重合，已自动更新原记录。")
    elif save_status == "created":
        st.info("该岗位已自动新增到采集库。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("职位", row.get("职位名称") or "-")
    c2.metric("公司", row.get("企业名称") or "-")
    c3.metric("城市", row.get("所在地区") or "-")
    c4.metric("薪资", row.get("薪资解析") or row.get("薪资原始") or "-")

    show_cols = ["职位名称", "企业名称", "所在地区", "薪资解析", "经验要求", "学历要求", "详情链接"]
    preview_df = pd.DataFrame([row])
    st.dataframe(preview_df[[col for col in show_cols if col in preview_df.columns]], use_container_width=True, hide_index=True)

    tag_cols = [col for col in ["硬技能标签", "软素质标签", "业务职责标签", "行业场景标签", "全部标签"] if col in row]
    if tag_cols:
        tag_rows = []
        for col in tag_cols:
            value = row.get(col)
            if isinstance(value, list):
                value = "，".join(str(item) for item in value if str(item).strip())
            if str(value or "").strip():
                tag_rows.append({"标签类型": col, "标签": value})
        if tag_rows:
            st.dataframe(pd.DataFrame(tag_rows), use_container_width=True, hide_index=True)

    with st.expander("查看岗位详情文本", expanded=False):
        st.text_area("岗位详情", value=row.get("岗位详情", ""), height=240, disabled=True)


def _render_capture_tab(debug_port: int, tabs: list[dict], connection_error: str):
    selected_tab = _render_browser_tab_selector(tabs, connection_error)
    st.markdown("#### 选择抓取方式")
    manual_col, hotkey_col = st.columns([1, 1])
    with manual_col:
        _render_manual_capture_section(selected_tab)
    with hotkey_col:
        _render_hotkey_section(debug_port, selected_tab)
    st.markdown("---")
    _render_llm_recognition_section(selected_tab)
    st.markdown("---")
    _render_last_result()


def _render_library_tab():
    st.subheader("本地岗位采集库")
    captured_df = load_captured_jobs()
    if captured_df.empty:
        st.info("还没有保存的岗位。")
        return

    t1, t2, t3 = st.columns([1, 1, 2])
    t1.metric("采集岗位数", len(captured_df))
    enabled = bool(st.session_state.get("boss_captured_jobs_enabled", False))
    with t2:
        st.session_state["boss_captured_jobs_enabled"] = st.toggle("加入当前会话分析", value=enabled)
    t3.caption(f"保存位置：{DEFAULT_CAPTURE_FILE}")

    show_cols = ["抓取时间", "职位名称", "企业名称", "所在地区", "薪资解析", "经验要求", "学历要求", "详情链接"]
    st.dataframe(captured_df[[col for col in show_cols if col in captured_df.columns]], use_container_width=True, hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        csv_bytes = captured_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载采集库 CSV",
            data=csv_bytes,
            file_name="boss_captured_jobs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        with st.expander("删除岗位", expanded=False):
            delete_options = []
            for _, item in captured_df.iterrows():
                label = " | ".join(
                    [
                        str(item.get("职位名称", "") or "-"),
                        str(item.get("企业名称", "") or "-"),
                        str(item.get("所在地区", "") or "-"),
                        str(item.get("详情链接", "") or item.get("页面URL", "") or "-")[:80],
                    ]
                )
                delete_options.append((label, build_capture_key(item)))

            selected_delete = st.selectbox(
                "选择要删除的岗位",
                delete_options,
                format_func=lambda option: option[0],
                index=0,
            )
            if st.button("删除选中岗位", type="secondary", use_container_width=True):
                updated = delete_captured_job(selected_delete[1])
                st.session_state.pop("boss_capture_last_row", None)
                st.session_state.pop("boss_capture_last_save_status", None)
                st.success(f"已删除。当前采集库共 {len(updated)} 条岗位。")
                st.rerun()


def _render_import_tab():
    st.subheader("Excel 导入采集库")
    st.caption("用于把历史岗位表、手工整理表或导出的采集库批量导入本地岗位采集库。导入时会清洗详情、提取标签，并自动去重。")

    with st.expander("上传格式要求", expanded=True):
        st.markdown(
            "支持 `.xlsx` / `.xls`，也兼容 `.csv`。首行必须是字段名。\n\n"
            "推荐字段：`抓取时间`、`页面URL`、`职位名称`、`企业名称`、`所在地区`、`薪资原始`、`薪资解析`、"
            "`经验要求`、`学历要求`、`工作地址`、`岗位详情`、`详情链接`、`数据来源`。\n\n"
            "至少应提供 `详情链接` 或 `页面URL`。如果没有链接，请提供 `职位名称`、`企业名称`、`所在地区`、`岗位详情`，否则去重质量会下降。"
        )
        st.dataframe(pd.DataFrame(columns=CAPTURE_COLUMNS), use_container_width=True, hide_index=True)

    uploaded_file = st.file_uploader(
        "上传岗位 Excel / CSV",
        type=["xlsx", "xls", "csv"],
        key="boss_capture_import_file",
    )

    mode_label = st.radio(
        "导入模式",
        ["合并模式（重复则更新）", "仅新增模式（重复则跳过）", "覆盖模式（替换整个采集库）"],
        horizontal=True,
    )
    mode_map = {
        "合并模式（重复则更新）": "merge_update",
        "仅新增模式（重复则跳过）": "append_new",
        "覆盖模式（替换整个采集库）": "overwrite",
    }
    enable_after_import = st.checkbox("导入后加入当前会话分析", value=True)

    if uploaded_file is None:
        return

    try:
        import_df = _read_import_file(uploaded_file)
    except Exception as exc:
        st.error(f"读取上传文件失败：{exc}")
        return

    if import_df.empty:
        st.warning("上传文件没有可导入的数据行。")
        return

    missing_recommended = [col for col in CAPTURE_COLUMNS if col not in import_df.columns]
    if missing_recommended:
        st.warning(f"缺少推荐字段：{missing_recommended}。缺失字段会自动补为空，但可能影响去重质量。")

    identity_cols = {"详情链接", "页面URL", "职位名称", "企业名称", "所在地区", "岗位详情"}
    if not identity_cols.intersection(set(import_df.columns)):
        st.error("上传文件缺少可用于识别岗位的字段，请至少提供详情链接、页面URL 或职位/公司/详情相关字段。")
        return

    st.write(f"待导入行数：{len(import_df)}")
    preview_cols = [col for col in CAPTURE_COLUMNS if col in import_df.columns]
    st.dataframe(import_df[preview_cols].head(20) if preview_cols else import_df.head(20), use_container_width=True, hide_index=True)

    if st.button("执行导入", type="primary", use_container_width=True):
        try:
            _, summary = import_captured_jobs(import_df, mode=mode_map[mode_label])
            st.session_state["boss_captured_jobs_enabled"] = bool(enable_after_import)
            st.success(
                "导入完成："
                f"原有 {summary['existing_count']} 条，上传 {summary['import_count']} 条，"
                f"当前采集库 {summary['final_count']} 条。"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"导入失败：{exc}")


def render_capture_page(df_raw: pd.DataFrame | None = None, processing_options: dict | None = None):
    st.header("岗位采集")
    st.caption("抓取、管理和导入岗位数据。采集页只负责数据入库，不会触发全量分析。")

    setup_left, setup_right = st.columns([1, 3])
    with setup_left:
        debug_port = st.number_input("Chrome 调试端口", min_value=1, max_value=65535, value=9222, step=1)

    tabs, connection_error = _get_debug_tabs(int(debug_port))
    with setup_right:
        _render_setup_panel(int(debug_port), len(tabs), connection_error)

    capture_tab, library_tab, import_tab = st.tabs(["抓取岗位", "采集库", "Excel 导入"])
    with capture_tab:
        _render_capture_tab(int(debug_port), tabs, connection_error)
    with library_tab:
        _render_library_tab()
    with import_tab:
        _render_import_tab()
