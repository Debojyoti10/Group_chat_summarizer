import regex as re
import datetime
from dateutil.parser import parse
import google.generativeai as palm
import Constants
from typing import List, Tuple

# ——— PROMPTS & CONFIG ———
SUMMARY_PROMPT = """Analyze this WhatsApp chat and provide a structured summary including:
- Main discussion topics with message counts
- Key shared links/resources
- Active participation patterns
- Important decisions/action items
- Media shared count
- Notable dates/events

Format the summary with clear section headings and bullet points."""
NEWSLETTER_PROMPT = """Create an engaging newsletter introduction paragraph that highlights key topics from these chat insights:"""
MAX_WORD_COUNT = 2000

# ——— CONFIGURE GEMINI ———
try:
    palm.configure(api_key=Constants.GEMINI_API_KEY)
except Exception as e:
    print(f"Gemini API configuration error: {e}")

# ——— PARSING ———
TIMESTAMP_RE = r'(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))\s*-\s*'

def read_file(file_path) -> str:
    try:
        data = file_path.getvalue()
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")

def parse_whatsapp(text: str) -> List[Tuple[datetime.date, str]]:
    parts = re.split(TIMESTAMP_RE, text)[1:]
    parsed = []
    for ts, msg in zip(parts[0::2], parts[1::2]):
        ts_clean = ts.replace('\u202F',' ').strip()
        try:
            dt = parse(ts_clean, dayfirst=True)
        except:
            continue
        content = msg.strip()
        if ': ' in content:
            content = content.split(': ',1)[1]
        parsed.append((dt.date(), content))
    return parsed

# ——— FILTERING & CHUNKING ———
def filter_messages_by_dates(messages, start_day, end_day):
    return [m for m in messages if start_day <= m[0] <= end_day]

def whatsapp_chunk_text(messages):
    chunks, current, wc = [], [], 0
    for _, msg in messages:
        words = msg.split()
        if wc + len(words) > MAX_WORD_COUNT:
            chunks.append(' '.join(current))
            current, wc = [], 0
        current.append(msg)
        wc += len(words)
    if current:
        chunks.append(' '.join(current))
    return chunks

# ——— GEMINI WRAPPER ———
def palm_api(prompt: str, model: str = "models/gemini-1.5-pro-001") -> str:
    """
    Uses your Gemini key + model.
    Default is now "models/gemini-1.5-pro-001". You may swap to any other model
    you saw in palm.list_models(), e.g. "models/gemini-2.5-pro-preview-03-25".
    """
    try:
        gm = palm.GenerativeModel(model)
        resp = gm.generate_content(
            prompt,
            generation_config={"temperature":0.7, "max_output_tokens":2000},
            safety_settings={
                "HARM_CATEGORY_DANGEROUS":"BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT":"BLOCK_NONE"
            }
        )
        return resp.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return (
            "⚠️ Gemini API error (check your key/quota).\n\n"
            "**No summary available**"
        )

def summarize_text(text: str, model: str) -> str:
    return palm_api(f"{SUMMARY_PROMPT}\n\n{text}", model)

def generate_newsletter_intro(text: str, model: str) -> str:
    return palm_api(f"{NEWSLETTER_PROMPT}\n{text}", model)

def summarize_messages(chunks, model):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        try:
            parts.append(summarize_text(chunk, model))
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
            parts.append(f"Summary part {i} unavailable")
    return "\n\n".join(parts)

# ——— MAIN ———
def main(chat_export_file, summary_file, start_day_s, end_day_s, is_newsletter, model):
    try:
        sd = datetime.datetime.strptime(start_day_s, "%m/%d/%Y").date()
        ed = datetime.datetime.strptime(end_day_s, "%m/%d/%Y").date()
    except ValueError:
        print("Invalid date format. Use MM/DD/YYYY")
        return

    text = read_file(chat_export_file)
    msgs = parse_whatsapp(text)
    if not msgs:
        print("No messages parsed. Check chat file format.")
        return

    filtered = filter_messages_by_dates(msgs, sd, ed)
    if not filtered:
        print("No messages in selected date range")
        return

    chunks  = whatsapp_chunk_text(filtered)
    summary = summarize_messages(chunks, model)

    if is_newsletter:
        try:
            intro = generate_newsletter_intro(summary, model)
            summary = f"{intro}\n\n{summary}"
        except Exception as e:
            print(f"Newsletter intro error: {e}")

    print("\n" + "*"*20 + " FINAL SUMMARY " + "*"*20)
    print(summary)

    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {summary_file}")
    except IOError as e:
        print(f"File save error: {e}")

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])