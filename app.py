import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import your hybrid summarizer functions
import group_chat_summarizer

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

def main():
    st.title("WhatsApp Chat Analyzer")
    
    uploaded_file = st.file_uploader("Upload WhatsApp Chat File", type="txt")
    if not uploaded_file:
        return  # nothing to do until a file is uploaded

    # 1) PROCESS UPLOAD
    try:
        raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = preprocessor.preprocess(raw)
        if df.empty:
            raise ValueError("No messages could be parsed. Check file format.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return

    # Force message column to string
    df['message'] = df['message'].astype(str)

    # DISPLAY BASIC STATS
    st.header("Basic Statistics")
    num_messages, words, num_media, num_links = helper.fetch_stats("Overall", df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", num_messages)
    c2.metric("Total Words", words)
    c3.metric("Media Shared", num_media)
    c4.metric("Links Shared", num_links)

    # ACTIVITY CHARTS
    st.header("Activity Patterns")
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    busy_day = helper.week_activity_map("Overall", df)
    ax[0].bar(busy_day.index, busy_day.values)
    ax[0].set_title("Most Active Days")
    ax[0].tick_params(axis='x', rotation=45)

    heatmap = helper.activity_heatmap("Overall", df)
    if not heatmap.empty:
        sns.heatmap(heatmap, ax=ax[1], cmap="viridis")
        ax[1].set_title("Hourly Activity")
    else:
        ax[1].text(0.5, 0.5, "No hourly data", ha='center')
    st.pyplot(fig)

    # TEXT ANALYSIS
    st.header("Text Insights")
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.subheader("Word Cloud")
        wc = helper.create_wordcloud("Overall", df)
        if wc:
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc)
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.write("No word cloud available.")

    with wcol2:
        st.subheader("Most Common Words")
        common = helper.most_common_words("Overall", df)
        if not common.empty:
            fig_cw, ax_cw = plt.subplots()
            ax_cw.barh(common['word'], common['count'])
            ax_cw.invert_yaxis()
            st.pyplot(fig_cw)
        else:
            st.write("No common words data.")

    # 2) SUMMARY SECTION
    st.header("📋 Group Chat Summary")

    # Date filters default to the min/max in the data
    start_date = st.date_input("Start date", value=df['only_date'].min())
    end_date   = st.date_input("End date",   value=df['only_date'].max())

    # Optionally let user pick a Gemini (or HF) model; default to your chosen one
    model = st.selectbox(
        "Summarization model",
        options=[
            "models/gemini-1.5-pro-001",
            "models/gemini-2.5-pro-preview-03-25",
            "local"  # will force the local HF fallback
        ],
        index=0
    )

    if st.button("Generate Summary"):
        with st.spinner("Generating summary…"):
            # Assemble the list of (date, message)
            msgs = [(row['only_date'], row['message']) for _, row in df.iterrows()]
            filtered = group_chat_summarizer.filter_messages_by_dates(msgs, start_date, end_date)
            if not filtered:
                st.error("No messages in the selected date range.")
            else:
                chunks = group_chat_summarizer.whatsapp_chunk_text(filtered)
                try:
                    # If user selected "local", pass model=None to force HF
                    summary = group_chat_summarizer.summarize_messages(
                        chunks,
                        None if model=="local" else model
                    )
                    st.text_area("Summary", summary, height=300)
                except Exception:
                    st.error("Summary generation failed. Please try again.")

if __name__ == "__main__":
    main()
