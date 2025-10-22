import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from yt_dlp import YoutubeDL
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random


def ytdlp_extract_channel_video_ids(channel_url: str) -> List[str]:
    """
    Extract all video URLs from a YouTube channel using yt-dlp without downloading.
    Returns a list of watch URLs.
    """
    # Primary extraction method
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        # Extract channel/videos tab as a playlist of entries
        "extract_flat": "in_playlist",
        # Add caching and retry settings to reduce errors
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        "no_warnings": True,
        # Add headers to avoid being blocked
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        # Add timeout settings
        "socket_timeout": 60,
        "request_timeout": 60,
        # Add delay between requests
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        # Try to get all videos, not just recent ones
        "playlist_end": -1,
    }
    
    video_urls: List[str] = []
    
    # Try primary method first
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            # Some channels resolve to a channel object with multiple tabs; handle playlist-like entries
            entries = info.get("entries", []) if isinstance(info, dict) else []
            for entry in entries:
                # Entries might themselves be a playlist (like "Videos" tab) — dive one level if needed
                if isinstance(entry, dict) and entry.get("entries"):
                    for sub in entry.get("entries", []):
                        vid_url = sub.get("url") or sub.get("webpage_url")
                        if vid_url:
                            # yt-dlp may return bare IDs in flat mode; ensure full watch URL
                            if vid_url.startswith("http"):
                                video_urls.append(vid_url)
                            else:
                                video_urls.append(f"https://www.youtube.com/watch?v={vid_url}")
                else:
                    vid_url = (entry or {}).get("url") or (entry or {}).get("webpage_url")
                    if vid_url:
                        if vid_url.startswith("http"):
                            video_urls.append(vid_url)
                        else:
                            video_urls.append(f"https://www.youtube.com/watch?v={vid_url}")
    except Exception as e:
        print(f"Primary extraction method failed: {e}")
        
    # If primary method failed or returned no results, try fallback methods
    if not video_urls:
        print("Trying fallback extraction methods...")
        
        # Fallback 1: Try with different extract_flat setting
        fallback_opts_1 = ydl_opts.copy()
        fallback_opts_1["extract_flat"] = True
        
        try:
            with YoutubeDL(fallback_opts_1) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                entries = info.get("entries", []) if isinstance(info, dict) else []
                for entry in entries:
                    vid_url = (entry or {}).get("url") or (entry or {}).get("webpage_url")
                    if vid_url:
                        if vid_url.startswith("http"):
                            video_urls.append(vid_url)
                        else:
                            video_urls.append(f"https://www.youtube.com/watch?v={vid_url}")
        except Exception as e:
            print(f"Fallback method 1 failed: {e}")
            
        # Fallback 2: Try with /videos appended to URL if not already there
        if not video_urls and "/videos" not in channel_url:
            try:
                alt_url = channel_url.rstrip("/") + "/videos"
                print(f"Trying alternative URL: {alt_url}")
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(alt_url, download=False)
                    entries = info.get("entries", []) if isinstance(info, dict) else []
                    for entry in entries:
                        vid_url = (entry or {}).get("url") or (entry or {}).get("webpage_url")
                        if vid_url:
                            if vid_url.startswith("http"):
                                video_urls.append(vid_url)
                            else:
                                video_urls.append(f"https://www.youtube.com/watch?v={vid_url}")
            except Exception as e:
                print(f"Alternative URL method failed: {e}")
                
        # Fallback 3: Try with different user agent
        if not video_urls:
            fallback_opts_3 = ydl_opts.copy()
            fallback_opts_3["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            
            try:
                with YoutubeDL(fallback_opts_3) as ydl:
                    info = ydl.extract_info(channel_url, download=False)
                    entries = info.get("entries", []) if isinstance(info, dict) else []
                    for entry in entries:
                        vid_url = (entry or {}).get("url") or (entry or {}).get("webpage_url")
                        if vid_url:
                            if vid_url.startswith("http"):
                                video_urls.append(vid_url)
                            else:
                                video_urls.append(f"https://www.youtube.com/watch?v={vid_url}")
            except Exception as e:
                print(f"Fallback method 3 failed: {e}")

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in video_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def ytdlp_extract_video_details(video_url: str) -> Dict[str, Any]:
    """
    Extract detailed metadata for a given YouTube video without downloading.
    Returns dict with title, views, date, link.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        # Add caching and retry settings to reduce errors
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        # Reduce metadata to only what we need
        "include_ads": False,
        "no_warnings": True,
        # Add headers to avoid being blocked
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        # Add timeout settings
        "socket_timeout": 60,
        "request_timeout": 60,
        # Add delay between requests
        "sleep_interval": 1,
        "max_sleep_interval": 5,
    }
    
    # Implement exponential backoff for retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

            title = info.get("title", "Unknown Title")
            view_count = info.get("view_count", 0)
            upload_date = info.get("upload_date")  # format: YYYYMMDD or None
            webpage_url = info.get("webpage_url") or video_url

            # Convert upload_date to ISO format YYYY-MM-DD for readability
            date_str = None
            if upload_date:
                try:
                    date_str = datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
                except Exception:
                    date_str = upload_date

            return {
                "title": title,
                "views": view_count,
                "date": date_str,
                "link": webpage_url,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed for {video_url}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error extracting video details for {video_url} after {max_retries} attempts: {e}")
                # Try fallback method with different options
                try:
                    print(f"Trying fallback method for {video_url}")
                    fallback_opts = ydl_opts.copy()
                    fallback_opts["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    fallback_opts["socket_timeout"] = 90
                    fallback_opts["request_timeout"] = 90
                    
                    with YoutubeDL(fallback_opts) as ydl:
                        info = ydl.extract_info(video_url, download=False)
                    
                    title = info.get("title", "Unknown Title")
                    view_count = info.get("view_count", 0)
                    upload_date = info.get("upload_date")
                    webpage_url = info.get("webpage_url") or video_url
                    
                    date_str = None
                    if upload_date:
                        try:
                            date_str = datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
                        except Exception:
                            date_str = upload_date
                    
                    return {
                        "title": title,
                        "views": view_count,
                        "date": date_str,
                        "link": webpage_url,
                    }
                except Exception as fallback_e:
                    print(f"Fallback method also failed for {video_url}: {fallback_e}")
                    return {
                        "title": "Error fetching title",
                        "views": 0,
                        "date": None,
                        "link": video_url,
                    }


def ytdlp_extract_channel_title(channel_url: str) -> str:
    """
    Extract the channel's display title using yt-dlp.
    Returns a non-empty string or 'channel' as fallback.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        # Add caching and retry settings to reduce errors
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        "no_warnings": True,
        # Add headers to avoid being blocked
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        # Add timeout settings
        "socket_timeout": 60,
        "request_timeout": 60,
    }
    title = None
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if isinstance(info, dict):
                title = info.get("title") or info.get("channel") or info.get("uploader")
    except Exception as e:
        print(f"Error extracting channel title: {e}")
        pass
    return title or "channel"


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename (Windows-safe)."""
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"[^A-Za-z0-9 _\.-]", "_", name)
    name = name.strip(" .") or "channel"
    return name


def build_dataframe_fast(channel_videos_url: str, gemini_key: str = "", model_name: str = "", max_workers: int = 10, progress_callback=None) -> pd.DataFrame:
    """
    Extract video details from a YouTube channel using parallel processing and optional Gemini analysis.
    
    Args:
        channel_videos_url: URL to the channel's videos page
        gemini_key: Optional Gemini API key for title analysis
        model_name: Optional model name for Gemini analysis
        max_workers: Number of parallel workers for scraping (default 10)
        progress_callback: Optional callback function to report progress (current, total)
    
    Returns:
        DataFrame with video details including optional analysis column
    """
    video_urls = ytdlp_extract_channel_video_ids(channel_videos_url)
    if not video_urls:
        print("No video URLs found. Returning empty DataFrame.")
        # Create empty DataFrame with proper columns
        df = pd.DataFrame(columns=["title", "views", "date", "link", "analysis"])
        # Ensure we have the right columns even if empty
        for col in ["title", "views", "date", "link", "analysis"]:
            if col not in df.columns:
                df[col] = None
        return df
    
    total = len(video_urls)
    print(f"Found {total} videos. Starting extraction...")
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed = [0]
    
    # Initialize Gemini if key provided
    model = None
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
        except Exception as e:
            print(f"Gemini init failed: {e}")
    
    def fetch_video(url: str) -> Optional[Dict[str, Any]]:
        try:
            details = ytdlp_extract_video_details(url)
            
            # Add analysis inline if model available
            if model:
                try:
                    prompt = (
                        "Analyze this YouTube video title in 2-3 sentences. Focus only on: "
                        "(1) What primary emotions does this title trigger? "
                        "(2) What patterns or hooks are used in the title? "
                        f"Title: '{details['title']}'"
                    )
                    resp = model.generate_content(prompt)
                    details["analysis"] = (resp.text or "").strip()
                except Exception as e:
                    details["analysis"] = f"Analysis unavailable: {e}"
            else:
                details["analysis"] = ""
            
            return details
        except Exception as e:
            print(f"Failed to fetch video {url}: {e}")
            return None
    
    rows = []
    # Use a simple progress indicator for CLI version
    print(f"Fetching details for {total} videos...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_video, url): url for url in video_urls}
        
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                rows.append(result)
            
            with progress_lock:
                completed[0] += 1
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed[0], total)
                elif completed[0] % 10 == 0 or completed[0] == total:
                    print(f"Processed {completed[0]}/{total} videos...")
    
    if not rows:
        print("No rows extracted. Returning empty DataFrame.")
        # Create empty DataFrame with proper columns
        df = pd.DataFrame(columns=["title", "views", "date", "link", "analysis"])
        # Ensure we have the right columns even if empty
        for col in ["title", "views", "date", "link", "analysis"]:
            if col not in df.columns:
                df[col] = None
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
    if not df.empty:
        df_sorted = df.copy()
        df_sorted["sort_date"] = df_sorted["date"].apply(lambda x: date_key(x))
        df_sorted = df_sorted.sort_values(by="sort_date", ascending=False)
        df_sorted = df_sorted.drop(columns=["sort_date"])
        return df_sorted
    else:
        return df


def scrape_channel_to_excel(channel_url: str, output_path: str) -> None:
    print("Collecting video list (this may take a minute)...")
    # Use the fast parallel version without AI analysis for CLI
    df = build_dataframe_fast(channel_url, gemini_key="", model_name="", max_workers=10)
    
    if df.empty:
        print("No data extracted.")
        return

    # Write to Excel
    df.to_excel(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube channel videos (title, views, date, link, analysis) to Excel using yt-dlp.")
    parser.add_argument("channel_url", help="YouTube channel URL or @handle or /videos tab URL")
    parser.add_argument("--output", "-o", default=None, help="Output Excel file path. If omitted, will use '<channel>.xlsx'.")
    parser.add_argument("--gemini-key", "-g", default=None, help="Optional Gemini API key for title analysis")
    parser.add_argument("--model", "-m", default="gemini-1.5-flash", help="Gemini model to use for analysis (default: gemini-1.5-flash)")
    args = parser.parse_args()

    # Normalize handle-only input
    original_input = args.channel_url.strip()
    url = original_input
    if url.startswith("@"):  # allow passing just @handle
        url = f"https://www.youtube.com/{url}/videos"
    elif url.endswith("/@"):
        url = url + "videos"
    elif url.endswith("/videos"):
        pass
    else:
        # Prefer the /videos tab to ensure entries are the channel's uploads
        if url.startswith("https://www.youtube.com/@") and "/videos" not in url:
            url = url.rstrip("/") + "/videos"

    # Determine output filename
    output_path = args.output
    if not output_path:
        # Use base (non-/videos) URL to get the channel title if possible
        base_url = original_input
        if base_url.startswith("@"):  # handle-only
            base_url = f"https://www.youtube.com/{base_url}"
        base_url = base_url.rstrip("/")
        channel_title = ytdlp_extract_channel_title(base_url)
        output_path = f"{safe_filename(channel_title)}.xlsx"

    try:
        print("Collecting video list (this may take a minute)...")
        # Use the fast parallel version with optional AI analysis for CLI
        df = build_dataframe_fast(url, gemini_key=args.gemini_key or "", model_name=args.model, max_workers=10)
        
        if df.empty:
            print("No data extracted.")
            return

        # Write to Excel
        df.to_excel(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
