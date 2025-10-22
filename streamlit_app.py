import os
# Set environment variable to use polling instead of inotify
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import re
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional

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
        build_dataframe_fast,
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


def analyze_titles_gemini(titles: List[str], api_key: str, model_name: str) -> pd.DataFrame:
    """Return a DataFrame with columns: title, analysis (2-3 sentences)."""
    if not api_key or not GEMINI_AVAILABLE:
        # Create empty DataFrame with proper columns
        df = pd.DataFrame(data=None, columns=["title", "analysis"])
        return df
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        # Create empty DataFrame with proper columns
        df = pd.DataFrame(data=None, columns=["title", "analysis"])
        return df

    rows = []
    for t in titles:
        prompt = (
            "Analyze this YouTube video title in 2-3 sentences. Focus only on: "
            "(1) What primary emotions does this title trigger? "
            "(2) What patterns or hooks are used in the title? "
            "Title: '" + t + "'"
        )
        try:
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()
        except Exception as e:
            text = f"Analysis unavailable: {e}"
        rows.append({"title": t, "analysis": text})
    
    # Create DataFrame with proper columns
    if rows:
        df = pd.DataFrame(rows)
        df = df.reindex(columns=["title", "analysis"])
    else:
        df = pd.DataFrame(data=None, columns=["title", "analysis"])
    return df


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
            # Different SDK versions expose different capabilities fields
            methods = getattr(m, "supported_generation_methods", None)
            if methods and ("generateContent" in methods or "generate_text" in methods):
                names.append(m.name)
        # Keep stable ordering: prefer common models first, then others
        prioritized = [n for n in default if n in names]
        others = [n for n in names if n not in prioritized]
        return prioritized + others
    except Exception:
        return default


c1, c2 = st.columns([3, 2])
with c1:
    url_input = st.text_input("Channel link or @handle", placeholder="https://www.youtube.com/@example or @example")
with c2:
    # Get API key from environment variable or user input
    default_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API Key (optional)", type="password", placeholder="AIza... or from HF secrets", help="Provide your own Google Gemini API key to add title analysis sheet.", value=default_key)

# Model selection for Gemini analysis (populated when key is provided)
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
        
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define progress callback function
        def update_progress(current, total):
            progress_percent = int((current / total) * 100)
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing video {current} of {total}...")

        with st.spinner("Collecting videos (may take a few minutes for large channels)..."):
            try:
                channel_title = ytdlp_extract_channel_title(base_url)
                clean_name = safe_filename(channel_title)
                # Use fast parallel version with progress callback
                df = build_dataframe_fast(videos_url, gemini_key if GEMINI_AVAILABLE else None, model_name if GEMINI_AVAILABLE else None, max_workers=5, progress_callback=update_progress)  # Reduced workers to be less aggressive
            except Exception as e:
                st.error(f"Error collecting videos: {str(e)}")
                st.info("Try using the explicit /videos URL, e.g. https://www.youtube.com/@handle/videos")
                # Try alternative approach
                try:
                    alt_url = videos_url + "/videos" if "/videos" not in videos_url else videos_url
                    st.info(f"Trying alternative approach with URL: {alt_url}")
                    df = build_dataframe_fast(alt_url, gemini_key if GEMINI_AVAILABLE else None, model_name if GEMINI_AVAILABLE else None, max_workers=3, progress_callback=update_progress)  # Even fewer workers
                    channel_title = ytdlp_extract_channel_title(base_url)
                    clean_name = safe_filename(channel_title)
                except Exception as e2:
                    st.error(f"Alternative approach also failed: {str(e2)}")
                    st.stop()
                finally:
                    # Clear progress bar after completion
                    progress_bar.empty()
                    status_text.empty()
            else:
                # Clear progress bar after successful completion
                progress_bar.empty()
                status_text.empty()

        if df.empty:
            st.error("No data extracted. Try the explicit /videos URL, e.g. https://www.youtube.com/@handle/videos")
            st.info("Note: Some channels may have restrictions that prevent data extraction.")
        else:
            st.success(f"Found {len(df)} videos for '{channel_title}'.")
            
            # Show a preview of the data
            st.subheader("Preview of Extracted Data")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Show some statistics
            st.subheader("Channel Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Videos", len(df))
            with col2:
                if not df['views'].empty:
                    st.metric("Total Views", f"{df['views'].sum():,}")
            with col3:
                if not df['date'].isnull().all():
                    st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")

            # Write Excel with single sheet containing analysis column
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Videos")
            output.seek(0)

            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"{clean_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

st.caption("Powered by yt-dlp + pandas + Streamlit")
