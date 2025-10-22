import re
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

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
    )
except ImportError:
    st.error("Missing required dependencies. Please ensure all requirements are installed.")
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


def build_dataframe_fast(channel_videos_url: str, gemini_key: Optional[str] = None, model_name: Optional[str] = None, max_workers: int = 15) -> pd.DataFrame:
    """
    Extract video details from a YouTube channel using parallel processing and optional Gemini analysis.
    
    Args:
        channel_videos_url: URL to the channel's videos page
        gemini_key: Optional Gemini API key for title analysis
        model_name: Optional model name for Gemini analysis
        max_workers: Number of parallel workers for scraping (increased to 15)
    
    Returns:
        DataFrame with video details including optional analysis column
    """
    video_urls = ytdlp_extract_channel_video_ids(channel_videos_url)
    total = len(video_urls)
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed = [0]
    prog = st.progress(0, text=f"Fetching details... 0/{total}")
    
    # Initialize Gemini if key provided and available
    model = None
    if gemini_key and GEMINI_AVAILABLE:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
        except Exception as e:
            st.warning(f"Gemini init failed: {e}")
    
    def fetch_video(url: str) -> Optional[Dict[str, Any]]:
        try:
            details = ytdlp_extract_video_details(url)
            
            # Add analysis inline if model available
            if model and GEMINI_AVAILABLE:
                try:
                    prompt = (
                        "Analyze this YouTube video title in 2-3 sentences. Focus only on: "
                        "(1) What primary emotions does this title trigger? "
                        "(2) What patterns or hooks are used in the title? "
                        f"Title: '{details['title']}'"
                    )
                    resp = model.generate_content(prompt)
                    details["analysis"] = (resp.text or "").strip()
                except Exception:
                    details["analysis"] = "Analysis unavailable"
            else:
                details["analysis"] = ""
            
            return details
        except Exception as e:
            st.warning(f"Failed: {url} ({e})")
            return None
    
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_video, url): url for url in video_urls}
        
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                rows.append(result)
            
            with progress_lock:
                completed[0] += 1
                prog.progress(
                    min(completed[0]/total, 1.0), 
                    text=f"Fetching details... {completed[0]}/{total}"
                )
    
    if not rows:
        # Create empty DataFrame with proper columns
        df = pd.DataFrame(data=None, columns=["title", "views", "date", "link", "analysis"])
        return df

    df = pd.DataFrame(rows)
    cols = ["title", "views", "date", "link"]
    if "analysis" in df.columns:
        cols.append("analysis")
    
    # Reorder columns
    df = df.reindex(columns=cols)

    def date_key(x):
        try:
            return datetime.strptime(x, "%Y-%m-%d") if isinstance(x, str) else datetime.min
        except Exception:
            return datetime.min

    # Sort by date descending where available
    df_sorted = df.copy()
    df_sorted["sort_date"] = df_sorted["date"].apply(date_key)
    df_sorted = df_sorted.sort_values(by="sort_date", ascending=False)
    df_sorted = df_sorted.drop(columns=["sort_date"])
    
    return df_sorted


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
        with st.spinner("Collecting videos (may take a few minutes for large channels)..."):
            channel_title = ytdlp_extract_channel_title(base_url)
            clean_name = safe_filename(channel_title)
            # Use fast parallel version
            df = build_dataframe_fast(videos_url, gemini_key if GEMINI_AVAILABLE else None, model_name if GEMINI_AVAILABLE else None, max_workers=15)

        if df.empty:
            st.error("No data extracted. Try the explicit /videos URL, e.g. https://www.youtube.com/@handle/videos")
        else:
            st.success(f"Found {len(df)} videos for '{channel_title}'.")
            st.dataframe(df.head(20), use_container_width=True)

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
