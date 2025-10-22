import os
# Set environment variable to use polling instead of inotify
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import re
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile

import pandas as pd
import streamlit as st

# Try to import google.generativeai, but handle if it's not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI not available. AI analysis will be disabled.")

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random

try:
    from scrape_youtube_channel import (
        ytdlp_extract_channel_video_ids,
        ytdlp_extract_video_details,
        ytdlp_extract_channel_title,
        safe_filename,
        scrape_channel_to_excel_realtime,
    )
except ImportError as e:
    st.error(f"Missing required dependencies: {e}")
    st.info("Please ensure all requirements are installed by running: pip install -r requirements.txt")
    st.stop()

st.set_page_config(page_title="YouTube → Excel (yt-dlp)", page_icon="📊", layout="centered")
st.title("YouTube Channel → Excel")
st.write("Paste a YouTube channel link or @handle. The app will extract video Title, Views, Date, Link, and Analysis and offer an Excel download.")


def normalize_channel_url(u: str) -> str:
    u = u.strip()
    if u.startswith("@"):
        return f"https://www.youtube.com/{u}/videos"
    if u.startswith("https://www.youtube.com/@") and "/videos" not in u:
        return u.rstrip("/") + "/videos"
    return u


def base_channel_url(u: str) -> str:
    b = u.strip()
    if b.startswith("@"):
        b = f"https://www.youtube.com/{b}"
    return b.rstrip("/")


def list_gemini_models(api_key: str) -> List[str]:
    """Return available Gemini model names supporting text generation for this key.
    Falls back to common options if listing fails."""
    default = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
    ]
    if not api_key or not GEMINI_AVAILABLE:
        return default
    try:
        genai.configure(api_key=api_key)
        names: List[str] = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None)
            if methods and ("generateContent" in methods or "generate_text" in methods):
                names.append(m.name)
        prioritized = [n for n in default if n in names]
        others = [n for n in names if n not in prioritized]
        return prioritized + others
    except Exception:
        return default


c1, c2 = st.columns([3, 2])
with c1:
    url_input = st.text_input("Channel link or @handle", placeholder="https://www.youtube.com/@example or @example")
with c2:
    default_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API Key (optional)", type="password", placeholder="AIza... or from HF secrets", help="Provide your own Google Gemini API key to add title analysis sheet.", value=default_key)

max_videos = st.number_input(
    "Maximum videos to extract (0 for unlimited)", 
    min_value=0, 
    max_value=10000, 
    value=1000,
    help="Limit for large channels to prevent timeouts on free tier. Set to 0 for no limit (not recommended for large channels)."
)

if gemini_key and GEMINI_AVAILABLE:
    available_models = list_gemini_models(gemini_key)
else:
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b"]

model_name = st.selectbox(
    "Gemini model",
    options=available_models,
    index=0 if available_models else None,
    help="Choose the model for title analysis. Flash is fastest; Pro is higher quality but slower.",
    disabled=not GEMINI_AVAILABLE
)

run = st.button("Run and Prepare Excel")

if run:
    if not url_input.strip():
        st.error("Please enter a channel link or @handle")
    else:
        videos_url = normalize_channel_url(url_input)
        base_url = base_channel_url(url_input)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        def update_progress(current, total):
            progress_percent = int((current / total) * 100) if total > 0 else 0
            progress_bar.progress(progress_percent)
            
            if total > 1000:
                elapsed_time = time.time() - start_time
                if current > 0:
                    eta_seconds = (elapsed_time / current) * (total - current)
                    eta_formatted = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    status_text.text(f"Processing video {current} of {total} ({progress_percent}%) - ETA: {eta_formatted}")
                else:
                    status_text.text(f"Processing video {current} of {total} ({progress_percent}%)")
            else:
                status_text.text(f"Processing video {current} of {total} ({progress_percent}%)")

        with st.spinner("Collecting videos (may take a few minutes for large channels)..."):
            try:
                channel_title = ytdlp_extract_channel_title(base_url)
                clean_name = safe_filename(channel_title)
                
                # Create a temporary file for real-time Excel writing
                temp_excel_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                temp_excel_file.close()
                temp_excel_path = temp_excel_file.name

                scrape_channel_to_excel_realtime(
                    videos_url,
                    temp_excel_path,
                    gemini_key=gemini_key if GEMINI_AVAILABLE else "",
                    model_name=model_name if GEMINI_AVAILABLE else "",
                    max_workers=15, # Increased workers for Streamlit
                    progress_callback=update_progress,
                    max_videos=max_videos if max_videos > 0 else None
                )

            except Exception as e:
                st.error(f"Error collecting videos: {str(e)}")
                st.info("Try using the explicit /videos URL, e.g. https://www.youtube.com/@handle/videos")
                try:
                    alt_url = videos_url + "/videos" if "/videos" not in videos_url else videos_url
                    st.info(f"Trying alternative approach with URL: {alt_url}")
                    
                    temp_excel_file_alt = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    temp_excel_file_alt.close()
                    temp_excel_path = temp_excel_file_alt.name

                    scrape_channel_to_excel_realtime(
                        alt_url,
                        temp_excel_path,
                        gemini_key=gemini_key if GEMINI_AVAILABLE else "",
                        model_name=model_name if GEMINI_AVAILABLE else "",
                        max_workers=15,
                        progress_callback=update_progress,
                        max_videos=max_videos if max_videos > 0 else None
                    )

                    channel_title = ytdlp_extract_channel_title(base_url)
                    clean_name = safe_filename(channel_title)
                except Exception as e2:
                    st.error(f"Alternative approach also failed: {str(e2)}")
                    st.stop()
                finally:
                    progress_bar.empty()
                    status_text.empty()
            else:
                progress_bar.empty()
                status_text.empty()

        if not os.path.exists(temp_excel_path) or os.path.getsize(temp_excel_path) == 0:
            st.error("No data extracted. Try the explicit /videos URL, e.g. https://www.youtube.com/@handle/videos")
            st.info("Note: Some channels may have restrictions that prevent data extraction.")
        else:
            # Read the generated Excel file to display data and statistics
            try:
                df_display = pd.read_excel(temp_excel_path)
            except Exception as e:
                st.error(f"Error reading generated Excel file: {e}")
                st.stop()

            st.success(f"Found {len(df_display)} videos for '{channel_title}'.")
            
            st.subheader("Preview of Extracted Data")
            st.dataframe(df_display.head(20), use_container_width=True)
            
            st.subheader("Channel Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Videos", len(df_display))
            with col2:
                if 'Views' in df_display.columns and not df_display['Views'].empty:
                    st.metric("Total Views", f"{df_display['Views'].sum():,}")
            with col3:
                if 'Date' in df_display.columns and not df_display['Date'].isnull().all():
                    st.metric("Date Range", f"{df_display['Date'].min()} to {df_display['Date'].max()}")

            with open(temp_excel_path, "rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f.read(),
                    file_name=f"{clean_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            
            # Clean up the temporary file
            os.remove(temp_excel_path)

st.caption("Powered by yt-dlp + openpyxl + Streamlit")
