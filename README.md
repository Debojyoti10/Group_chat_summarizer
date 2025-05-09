
# Whatsapp Chat Analyzer and Summarizer

This is a Streamlit-based application for analyzing WhatsApp group chat data. The application allows users to upload their WhatsApp chat data, preprocesses it, and provides various analyses and visualizations to gain insights into the communication patterns within the group.



## Usage

1. Upload WhatsApp chat data: Click on the "Choose a file" button in the sidebar to upload your WhatsApp chat data in text format.

2. Select user and analyze: Choose a user from the dropdown list to view statistics and visualizations specific to that user. Click the "Show Analysis" button to generate and display insights.

3. Date Range and Model Selection: Use the sidebar to select a date range and choose a summarization model from the available options. You can also opt to generate a newsletter intro.

4. Summarize: Click the "Summarize" button to generate a summary based on the selected date range and model. The summary will be displayed below the button.

### Exporting chat from WhatsApp
To export your group chat from WhatsApp, follow these steps:

1. Open WhatsApp and go to the group chat you want to export.
2. Tap on the group name at the top of the screen to open the group info.
3. Scroll down and tap on "Export chat".
4. Choose whether to include media files or not.
5. Select how you want to share the chat export file. You can send it to yourself via email, save it to your device, or use any other method.
6. Save the chat export file as a text file with a .txt extension.

## Features

- **Statistics Area:** Provides overall statistics such as total messages, total words, media shared, and links shared.

- **Timeline Visualizations:** Presents monthly and daily timelines of message activity within the group.

- **Activity Maps:** Displays the busiest day, busiest month, and a weekly activity heatmap.

- **Most Busy Users (Group Level):** Identifies and displays the most active users in the group.

- **WordCloud:** Generates a word cloud based on the selected user's messages.

- **Most Common Words:** Displays a bar chart of the most common words used by the selected user.

- **Emoji Analysis:** Provides a dataframe and a pie chart showing the distribution of emojis used by the selected user.

- **Summarization:** Allows users to summarize the chat data within a specified date range using different summarization models. The summary can be displayed and saved.


