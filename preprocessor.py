import re
import pandas as pd
from dateutil.parser import parse

def preprocess(raw: str) -> pd.DataFrame:
    """
    Turn a WhatsApp export into a DataFrame with:
      ['date','user','message','only_date','year','month_num','month',
       'day','day_name','hour','minute','period']
    """

    # 1) Regex to capture the entire timestamp (mm/dd/yy, hh:mm am/pm) plus the " - "
    TIMESTAMP_RE = (
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s*'      # date
        r'\d{1,2}:\d{2}\s*'                   # time
        r'(?:am|pm|AM|PM))'                   # am/pm
        r'\s*-\s*'                            # separator
    )

    # 2) Split on that regex, keeping the timestamps
    parts = re.split(TIMESTAMP_RE, raw)
    # parts = ["", ts1, msg1, ts2, msg2, ...]
    parts = parts[1:]  # drop any leading empty

    if len(parts) < 2:
        return pd.DataFrame()

    timestamps = parts[0::2]
    messages   = parts[1::2]

    records = []
    for ts, msg in zip(timestamps, messages):
        # normalize WhatsApp’s narrow-NBSP to normal spaces
        ts_clean = ts.replace('\u202F', ' ').strip()
        try:
            dt = parse(ts_clean, dayfirst=True)
        except Exception:
            continue

        # split off "Sender: message"
        entry = re.split(r'^([^:]+):\s*', msg, maxsplit=1)
        if len(entry) == 3:
            user, text = entry[1].strip(), entry[2].strip()
        else:
            user, text = 'group_notification', msg.strip()

        records.append((dt, user, text))

    if not records:
        return pd.DataFrame()

    # 3) Build the DataFrame
    df = pd.DataFrame(records, columns=['date','user','message'])

    # 4) Derive all the usual date/time features
    df['only_date'] = df['date'].dt.date
    df['year']      = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month']     = df['date'].dt.month_name()
    df['day']       = df['date'].dt.day
    df['day_name']  = df['date'].dt.day_name()
    df['hour']      = df['date'].dt.hour
    df['minute']    = df['date'].dt.minute
    df['period']    = df['hour'].apply(lambda h: f"{h}-{(h+1)%24}")

    return df
