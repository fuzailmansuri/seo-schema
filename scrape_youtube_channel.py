
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import openpyxl
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from yt_dlp import YoutubeDL
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os

# Try to import Google API client, but handle if it's not available
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Google API client not installed. Install with: pip install google-api-python-client")

def initialize_youtube_api():
    """Initialize YouTube Data API client if key is available"""
    if not GOOGLE_API_AVAILABLE:
        return None
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return None
    
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        return youtube
    except Exception as e:
        print(f"Failed to initialize YouTube API client: {e}")
        return None

# Initialize YouTube API client
youtube_api = initialize_youtube_api()

def ytdlp_extract_video_details(video_url: str) -> Dict[str, Any]:
    """
    Extract detailed metadata for a given YouTube video without downloading.
    Returns dict with title, views, date, link.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 15,  # Increased retries
        "fragment_retries": 15,  # Increased retries
        "skip_unavailable_fragments": True,
        "include_ads": False,
        "no_warnings": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        "socket_timeout": 90,  # Increased timeout
        "request_timeout": 90,  # Increased timeout
        "sleep_interval": 2,  # Added sleep interval
        "max_sleep_interval": 10,  # Increased max sleep
    }
    
    max_retries = 7  # Increased retries
    for attempt in range(max_retries):
        try:
            with YoutubeDL(ydl_opts) as ydl:
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
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (0.1 * attempt)  # Add jitter
                print(f"Attempt {attempt + 1} failed for {video_url}: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error extracting video details for {video_url} after {max_retries} attempts: {e}")
                try:
                    print(f"Trying fallback method for {video_url}")
                    fallback_opts = ydl_opts.copy()
                    fallback_opts["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    fallback_opts["socket_timeout"] = 120
                    fallback_opts["request_timeout"] = 120
                    fallback_opts["sleep_interval"] = 5
                    fallback_opts["max_sleep_interval"] = 15
                    
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
    
    return {
        "title": "Error fetching title",
        "views": 0,
        "date": None,
        "link": video_url,
    }


def ytdlp_extract_channel_title(channel_url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 15,  # Increased retries
        "fragment_retries": 15,  # Increased retries
        "skip_unavailable_fragments": True,
        "no_warnings": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        "socket_timeout": 90,  # Increased timeout
        "request_timeout": 90,  # Increased timeout
        "sleep_interval": 2,  # Added sleep interval
        "max_sleep_interval": 10,  # Increased max sleep
    }
    
    # Try multiple approaches for channel title extraction
    title = None
    
    # Approach 1: Standard extraction
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if isinstance(info, dict):
                title = info.get("title") or info.get("channel") or info.get("uploader")
    except Exception as e:
        print(f"Standard channel title extraction failed: {e}")
    
    # Approach 2: Fallback with different options
    if not title:
        try:
            fallback_opts = ydl_opts.copy()
            fallback_opts["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            fallback_opts["socket_timeout"] = 120
            fallback_opts["request_timeout"] = 120
            
            with YoutubeDL(fallback_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if isinstance(info, dict):
                    title = info.get("title") or info.get("channel") or info.get("uploader")
        except Exception as e:
            print(f"Fallback channel title extraction failed: {e}")
    
    # Approach 3: Try with /videos suffix if not already present
    if not title and "/videos" not in channel_url:
        try:
            alt_url = channel_url.rstrip("/") + "/videos"
            print(f"Trying alternative URL for channel title: {alt_url}")
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(alt_url, download=False)
                if isinstance(info, dict):
                    title = info.get("title") or info.get("channel") or info.get("uploader")
        except Exception as e:
            print(f"Alternative URL channel title extraction failed: {e}")
    
    return title or "channel"


def safe_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"[^A-Za-z0-9 _\.-]", "_", name)
    name = name.strip(" ._") or "channel"
    return name


def ytdlp_extract_channel_video_ids(channel_url: str, max_videos=None) -> List[str]:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "cachedir": True,
        "retry_sleep_functions": {"http": lambda n: 2 ** n, "fragment": lambda n: 2 ** n},
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        "no_warnings": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        "socket_timeout": 60,
        "request_timeout": 60,
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        "playlist_end": max_videos if max_videos else -1,
    }
    
    video_urls: List[str] = []
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            entries = info.get("entries", []) if isinstance(info, dict) else []
            for entry in entries:
                if isinstance(entry, dict) and entry.get("entries"):
                    for sub in entry.get("entries", []):
                        vid_url = sub.get("url") or sub.get("webpage_url")
                        if vid_url:
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
        
    if not video_urls:
        print("Trying fallback extraction methods...")
        
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

    seen = set()
    deduped = []
    for u in video_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def scrape_channel_to_excel_realtime(channel_videos_url: str, output_path: str, gemini_key: str = "", model_name: str = "", max_workers: int = 10, progress_callback=None, max_videos=None):
    # Import our robust extraction functions
    try:
        from youtube_api_wrapper import (
            robust_extract_channel_videos,
            robust_extract_video_details
        )
        # Use robust extraction with fallback
        video_urls = robust_extract_channel_videos(channel_videos_url, max_videos)
    except ImportError:
        # Fallback to original yt-dlp only approach
        video_urls = ytdlp_extract_channel_video_ids(channel_videos_url, max_videos)
    
    if not video_urls:
        print("No video URLs found.")
        return

    total = len(video_urls)
    print(f"Found {total} videos. Starting extraction...")

    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is not None:
        ws.title = "Videos"
        header = ["Title", "Views", "Date", "Link", "Analysis"]
        ws.append(header)
    wb.save(output_path)

    progress_lock = threading.Lock()
    completed = [0]

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
            # Use robust extraction with fallback
            try:
                from youtube_api_wrapper import robust_extract_video_details
                details = robust_extract_video_details(url)
            except ImportError:
                # Fallback to original approach
                details = ytdlp_extract_video_details(url)
            
            if model and details:
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
            elif details:
                details["analysis"] = ""
            
            return details
        except Exception as e:
            print(f"Failed to fetch video {url}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_video, url): url for url in video_urls}
        
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                with progress_lock:
                    ws.append([result["title"], result["views"], result["date"], result["link"], result["analysis"]])
                    wb.save(output_path)
            
            with progress_lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], total)
                elif completed[0] % 10 == 0 or completed[0] == total:
                    print(f"Processed {completed[0]}/{total} videos...")

def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube channel videos (title, views, date, link, analysis) to Excel using yt-dlp.")
    parser.add_argument("channel_url", help="YouTube channel URL or @handle or /videos tab URL")
    parser.add_argument("--output", "-o", default=None, help="Output Excel file path. If omitted, will use '<channel>.xlsx'.")
    parser.add_argument("--gemini-key", "-g", default=None, help="Optional Gemini API key for title analysis")
    parser.add_argument("--model", "-m", default="gemini-1.5-flash", help="Gemini model to use for analysis (default: gemini-1.5-flash)")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to scrape")
    args = parser.parse_args()

    original_input = args.channel_url.strip()
    url = original_input
    if url.startswith("@"):
        url = f"https://www.youtube.com/{url}/videos"
    elif url.endswith("/@"):
        url = url + "videos"
    elif not url.endswith("/videos"):
        if url.startswith("https://www.youtube.com/@") and "/videos" not in url:
            url = url.rstrip("/") + "/videos"

    output_path = args.output
    if not output_path:
        base_url = original_input
        if base_url.startswith("@"):
            base_url = f"https://www.youtube.com/{base_url}"
        base_url = base_url.rstrip("/")
        channel_title = ytdlp_extract_channel_title(base_url)
        output_path = f"{safe_filename(channel_title)}.xlsx"

    try:
        scrape_channel_to_excel_realtime(url, output_path, gemini_key=args.gemini_key or "", model_name=args.model, max_workers=10, max_videos=args.max_videos)
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)

if __name__ == "__main__":
    main()


