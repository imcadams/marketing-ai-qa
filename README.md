# Marketing Campaign Chatbot
Ask questions about marketing data, get advice, analysis, or generate any form of text based content. Give the Marketing Campaign Chatbot your campaign directory path and it will index the data for you, and analyze it with the help of ChatGPT to provide you with the marketing assistance you need.

# Installation
Marketing Campaign Chatbot needs [Python 3.10](https://www.python.org/downloads/release/python-3100/) to run.

At the root directory of the program, run the `setup.bat` script first. 

Then you must update the `.env` file with the [AzureOpenAI](https://oai.azure.com/) API key and the full directory path you wish to apply cognitive search to.

Once that's complete, you can start the program by executing the `run.bat` script.

# Usage
The Marketing Campaign Chatbot uses the ChatGPT 3.5 model in conjunction with cognitive search to intelligently query the data, in a human like fashion. It uses converstaion memory, so it can reference previously mentioned items for new queries.

Here are some examples of what is can generate:
- Marketing emails
- Social media post copy, including emojis and hashtags
- Blog posts
- Articles
- Action items, next steps
- Image recommendations based on text based image metadata
- General marketing analysis and recommendations

Here are some other things it can do, as a marketing assistant:
- Analyze other text based files such as
    - Task lists: depending on the amount of detail, it can extrapolate information such as due dates, completion of task, next steps, items due today, etc.
    - Contact lists: it can provide contact information based on a quick text based query
    - Word documents
    - PDFs
    - CSVs

That is only what has been tested, the possibilites are endless!