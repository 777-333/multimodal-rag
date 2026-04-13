import base64
import os
import tempfile
import zipfile
from datetime import date
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, rag, auth

st.set_page_config(page_title="Multimodal RAG", page_icon="🔍", layout="wide")

# ── Login Gate ────────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.title("Multimodal RAG — Login")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Anmelden")
            username = st.text_input("Benutzername")
            password = st.text_input("Passwort", type="password")
            submitted = st.form_submit_button("Login", type="primary", width="stretch")
        if submitted:
            user = auth.login(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Ungültige Anmeldedaten oder Konto deaktiviert.")
    st.stop()

# ── Logged-in helpers ─────────────────────────────────────────────────────────
current_user = st.session_state["user"]
role = current_user["role"]  # admin | user | viewer
is_admin = role == "admin"
is_viewer = role == "viewer"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**{current_user['username']}** `{role}`")
    if st.button("Logout"):
        del st.session_state["user"]
        st.rerun()

    st.divider()
    st.header("Settings")
    top_k = st.slider("Top K results", 1, 50, 10)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05)
    filter_type = st.selectbox(
        "Content type filter",
        ["all", "text", "image", "pdf", "audio", "video"],
    )
    use_codex = st.checkbox("Use Codex reasoning", value=True)

    st.divider()
    st.header("Database Stats")
    if st.button("Refresh stats"):
        st.cache_data.clear()

    @st.cache_data(ttl=30)
    def _stats():
        return db.get_stats()

    try:
        stats = _stats()
        st.metric("Total documents", stats["total"])
        for ctype, count in stats["by_type"].items():
            st.metric(ctype.capitalize(), count)
    except Exception as e:
        st.error(f"Could not load stats: {e}")

# ── Tabs (role-based) ─────────────────────────────────────────────────────────
st.title("Multimodal RAG with Gemini Embedding")

if is_admin:
    tab_upload, tab_search, tab_browse, tab_admin = st.tabs(
        ["Upload & Embed", "Search", "Browse", "Admin"]
    )
elif is_viewer:
    (tab_search,) = st.tabs(["Search"])
    tab_upload = tab_browse = tab_admin = None
else:  # user
    tab_upload, tab_search, tab_browse = st.tabs(["Upload & Embed", "Search", "Browse"])
    tab_admin = None

# ── Tab: Upload & Embed ───────────────────────────────────────────────────────
if tab_upload:
    with tab_upload:
        st.subheader("Upload files to embed")
        uploaded_files = st.file_uploader(
            "Choose one or more files",
            type=["txt", "png", "jpg", "jpeg", "webp", "gif", "pdf", "mp3", "wav", "mp4", "mov", "avi"],
            accept_multiple_files=True,
        )
        title = st.text_input("Document title (applied to all files)", placeholder="My document")

        if uploaded_files and title:
            if st.button("Embed & Store", type="primary"):
                total_stored = 0
                for file_idx, uploaded in enumerate(uploaded_files):
                    st.write(f"**Processing {file_idx+1}/{len(uploaded_files)}: {uploaded.name}**")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chunks_done = [0]

                    def on_progress(msg: str, _bar=progress_bar, _st=status_text, _c=chunks_done):
                        _c[0] += 1
                        _st.text(msg)
                        _bar.progress(min(_c[0] * 10, 100))

                    try:
                        file_bytes = uploaded.read()
                        mime = uploaded.type or "application/octet-stream"
                        status_text.text("Processing...")
                        results = rag.ingest(
                            file_bytes=file_bytes,
                            filename=uploaded.name,
                            title=title,
                            mime_type=mime,
                            on_progress=on_progress,
                        )
                        progress_bar.progress(100)
                        status_text.text("Done!")
                        total_stored += len(results)
                    except Exception as e:
                        st.error(f"Error processing {uploaded.name}: {e}")
                        raise
                st.success(f"Stored {total_stored} chunk(s) across {len(uploaded_files)} file(s)")
                st.cache_data.clear()
        elif uploaded_files and not title:
            st.warning("Please enter a document title.")

        # ── Server-Pfad Import ────────────────────────────────────────────────
        st.divider()
        st.subheader("Server-Pfad Import")
        st.caption("Dateien per SFTP auf den Server kopieren → Pfad angeben → einbetten (kein Upload-Limit).")

        server_title = st.text_input(
            "Dokumententitel",
            value=date.today().strftime("%Y-%m-%d"),
            key="server_title",
        )
        server_path = st.text_input(
            "Ordnerpfad auf dem Server",
            placeholder="/imports/mein-ordner",
            key="server_path",
        )

        if server_path and server_title:
            if st.button("Server-Pfad einbetten", type="primary", key="server_embed"):
                if not os.path.isdir(server_path):
                    st.error(f"Pfad nicht gefunden: `{server_path}`")
                else:
                    SUPPORTED_EXT_SRV = {
                        ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif",
                        ".mp3", ".wav", ".mp4", ".mov", ".avi",
                        ".docx", ".xlsx", ".xls", ".csv",
                    }
                    all_files_srv = []
                    for root, dirs, files in os.walk(server_path):
                        depth = root.replace(server_path, "").count(os.sep)
                        if depth > 4:
                            dirs.clear()
                            continue
                        for fname in sorted(files):
                            ext = os.path.splitext(fname)[1].lower()
                            if ext in SUPPORTED_EXT_SRV:
                                all_files_srv.append(os.path.join(root, fname))

                    if not all_files_srv:
                        st.warning("Keine unterstützten Dateien im Pfad gefunden.")
                    else:
                        st.info(f"{len(all_files_srv)} Dateien gefunden — starte Embedding...")
                        overall_srv = st.progress(0)
                        log_srv = st.empty()
                        total_stored_srv = 0
                        errors_srv = []

                        for idx, filepath in enumerate(all_files_srv):
                            fname = os.path.basename(filepath)
                            rel_path = os.path.relpath(filepath, server_path)
                            log_srv.text(f"[{idx+1}/{len(all_files_srv)}] {rel_path}")
                            ext = os.path.splitext(fname)[1].lower()
                            mime = MIME_BY_EXT.get(ext, "application/octet-stream")
                            try:
                                with open(filepath, "rb") as f:
                                    file_bytes = f.read()
                                results = rag.ingest(
                                    file_bytes=file_bytes,
                                    filename=fname,
                                    title=server_title,
                                    mime_type=mime,
                                )
                                total_stored_srv += len(results)
                            except Exception as e:
                                errors_srv.append(f"{rel_path}: {e}")
                            overall_srv.progress((idx + 1) / len(all_files_srv))

                        log_srv.text("Fertig!")
                        st.success(f"{total_stored_srv} Chunks aus {len(all_files_srv)} Dateien gespeichert.")
                        if errors_srv:
                            with st.expander(f"{len(errors_srv)} Fehler"):
                                for err in errors_srv:
                                    st.error(err)
                        st.cache_data.clear()

        # ── Ordner-Import via ZIP ─────────────────────────────────────────────
        st.divider()
        st.subheader("Ordner-Import (ZIP)")
        st.caption("Ordner als ZIP hochladen — alle Unterordner werden rekursiv verarbeitet.")

        SUPPORTED_EXT = {
            ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif",
            ".mp3", ".wav", ".mp4", ".mov", ".avi",
            ".docx", ".xlsx", ".xls", ".csv",
        }
        MIME_BY_EXT = {
            ".txt": "text/plain", ".pdf": "application/pdf",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif",
            ".mp3": "audio/mpeg", ".wav": "audio/wav",
            ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel", ".csv": "text/csv",
        }

        zip_title = st.text_input(
            "Dokumententitel (für alle Dateien)",
            value=date.today().strftime("%Y-%m-%d"),
            key="zip_title",
        )
        zip_file = st.file_uploader("ZIP-Datei hochladen", type=["zip"], key="zip_upload")

        if zip_file and zip_title:
            if st.button("Ordner einbetten", type="primary", key="zip_embed"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "upload.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.read())
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmpdir)

                    # Alle unterstützten Dateien rekursiv sammeln (max. 4 Ebenen)
                    all_files = []
                    for root, dirs, files in os.walk(tmpdir):
                        depth = root.replace(tmpdir, "").count(os.sep)
                        if depth > 4:
                            dirs.clear()
                            continue
                        for fname in sorted(files):
                            ext = os.path.splitext(fname)[1].lower()
                            if ext in SUPPORTED_EXT and fname != "upload.zip":
                                all_files.append(os.path.join(root, fname))

                    if not all_files:
                        st.warning("Keine unterstützten Dateien im ZIP gefunden.")
                    else:
                        st.info(f"{len(all_files)} Dateien gefunden — starte Embedding...")
                        overall = st.progress(0)
                        log = st.empty()
                        total_stored = 0
                        errors = []

                        for idx, filepath in enumerate(all_files):
                            fname = os.path.basename(filepath)
                            rel_path = os.path.relpath(filepath, tmpdir)
                            log.text(f"[{idx+1}/{len(all_files)}] {rel_path}")
                            ext = os.path.splitext(fname)[1].lower()
                            mime = MIME_BY_EXT.get(ext, "application/octet-stream")
                            try:
                                with open(filepath, "rb") as f:
                                    file_bytes = f.read()
                                results = rag.ingest(
                                    file_bytes=file_bytes,
                                    filename=fname,
                                    title=zip_title,
                                    mime_type=mime,
                                )
                                total_stored += len(results)
                            except Exception as e:
                                errors.append(f"{rel_path}: {e}")
                            overall.progress((idx + 1) / len(all_files))

                        log.text("Fertig!")
                        st.success(f"{total_stored} Chunks aus {len(all_files)} Dateien gespeichert.")
                        if errors:
                            with st.expander(f"{len(errors)} Fehler"):
                                for err in errors:
                                    st.error(err)
                        st.cache_data.clear()

# ── Tab: Search ───────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Query your documents")
    query_text = st.text_area("Enter your query", height=100)

    if st.button("Search", type="primary", key="search_btn"):
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                try:
                    result = rag.query(
                        query_text=query_text.strip(),
                        top_k=top_k,
                        threshold=threshold,
                        filter_type=filter_type,
                        use_codex=use_codex,
                    )
                except Exception as e:
                    st.error(f"Search error: {e}")
                    raise

            if result["answer"]:
                st.subheader("Answer")
                st.markdown(result["answer"])

            st.subheader(f"Sources ({len(result['sources'])} matches)")
            if not result["sources"]:
                st.info("No matching documents found. Try lowering the similarity threshold.")

            for src in result["sources"]:
                sim = src.get("similarity", 0)
                with st.expander(
                    f"[{sim:.3f}] {src['title']} — {src['original_filename']} "
                    f"(chunk {src['chunk_index']}/{src['chunk_total']}, {src['content_type']})",
                    expanded=src["content_type"] in ("image", "video"),
                ):
                    if src["content_type"] == "image" and src.get("file_data"):
                        img_bytes = base64.b64decode(src["file_data"])
                        st.image(img_bytes, caption=src["original_filename"], width="stretch")
                    elif src["content_type"] == "video" and src.get("file_data"):
                        vid_bytes = base64.b64decode(src["file_data"])
                        mime = (src.get("metadata") or {}).get("mime_type", "video/mp4")
                        st.video(vid_bytes, format=mime)
                    elif src["content_type"] == "pdf" and src.get("file_data"):
                        st.markdown(
                            f'<iframe src="data:application/pdf;base64,{src["file_data"]}" '
                            f'width="100%" height="600px" style="border:none;border-radius:4px;"></iframe>',
                            unsafe_allow_html=True,
                        )
                        pdf_bytes = base64.b64decode(src["file_data"])
                        st.download_button(
                            label="PDF herunterladen",
                            data=pdf_bytes,
                            file_name=src["original_filename"],
                            mime="application/pdf",
                            key=f"pdf_{src.get('id', src['original_filename'])}_{src['chunk_index']}",
                        )
                        if src.get("text_content"):
                            st.text(src["text_content"][:2000])
                    elif src.get("text_content"):
                        st.text(src["text_content"][:2000])
                    else:
                        st.caption(f"Non-text content ({src['content_type']})")
                    if src.get("metadata"):
                        st.json(src["metadata"])

# ── Tab: Browse ───────────────────────────────────────────────────────────────
if tab_browse:
    with tab_browse:
        st.subheader("All documents")

        try:
            docs = db.get_all_documents()
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            docs = []

        if not docs:
            st.info("No documents yet. Upload something in the first tab.")
        else:
            st.dataframe(
                docs,
                width="stretch",
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Title"),
                    "content_type": st.column_config.TextColumn("Type"),
                    "original_filename": st.column_config.TextColumn("Filename"),
                    "chunk_index": st.column_config.NumberColumn("Chunk"),
                    "chunk_total": st.column_config.NumberColumn("Total"),
                    "created_at": st.column_config.TextColumn("Created"),
                },
            )

            if is_admin:
                st.divider()
                st.subheader("Delete documents")
                delete_id = st.text_input("Document ID to delete")
                if st.button("Delete", type="secondary"):
                    if delete_id.strip():
                        try:
                            db.delete_document(delete_id.strip())
                            st.success(f"Deleted {delete_id}")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete error: {e}")
                    else:
                        st.warning("Enter a document ID.")

# ── Tab: Admin ────────────────────────────────────────────────────────────────
if tab_admin:
    with tab_admin:
        st.subheader("Benutzerverwaltung")

        # ── Neuen User anlegen ────────────────────────────────────────────────
        with st.expander("Neuen Benutzer anlegen", expanded=False):
            with st.form("create_user_form"):
                new_username = st.text_input("Benutzername")
                new_email = st.text_input("E-Mail")
                new_password = st.text_input("Passwort", type="password")
                new_role = st.selectbox("Rolle", ["user", "viewer", "admin"])
                if st.form_submit_button("Benutzer erstellen", type="primary"):
                    if new_username and new_email and new_password:
                        try:
                            auth.create_user(new_username, new_email, new_password, new_role)
                            st.success(f"Benutzer '{new_username}' wurde erstellt.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                    else:
                        st.warning("Alle Felder ausfüllen.")

        st.divider()

        # ── User-Liste ────────────────────────────────────────────────────────
        try:
            users = auth.get_all_users()
        except Exception as e:
            st.error(f"Fehler beim Laden der Benutzer: {e}")
            users = []

        for u in users:
            col_info, col_role, col_active, col_del = st.columns([3, 2, 1, 1])
            with col_info:
                st.markdown(f"**{u['username']}** — {u['email']}")
                st.caption(f"Erstellt: {u['created_at'][:10] if u['created_at'] else '-'}")
            with col_role:
                new_role = st.selectbox(
                    "Rolle",
                    ["user", "viewer", "admin"],
                    index=["user", "viewer", "admin"].index(u["role"]),
                    key=f"role_{u['id']}",
                    label_visibility="collapsed",
                )
                if new_role != u["role"]:
                    if st.button("Speichern", key=f"save_{u['id']}"):
                        auth.update_user(u["id"], {"role": new_role})
                        st.success(f"Rolle für '{u['username']}' gespeichert.")
                        st.rerun()
            with col_active:
                label = "Aktiv" if u["is_active"] else "Inaktiv"
                if st.button(label, key=f"toggle_{u['id']}"):
                    # Admins können sich nicht selbst deaktivieren
                    if u["id"] == current_user["id"]:
                        st.warning("Du kannst dich nicht selbst deaktivieren.")
                    else:
                        auth.update_user(u["id"], {"is_active": not u["is_active"]})
                        st.rerun()
            with col_del:
                if st.button("Löschen", key=f"del_{u['id']}", type="secondary"):
                    if u["id"] == current_user["id"]:
                        st.warning("Du kannst dich nicht selbst löschen.")
                    else:
                        auth.delete_user(u["id"])
                        st.rerun()
            st.divider()
