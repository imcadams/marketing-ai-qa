# Marketing AI QA Bot
Marketing AI QA bot, with ability to add a directory to analyze textual data. The goal is to be able to be able to have the AI analyze the data and generate text based marketing materials.

# Installation
Marketing AI QA bot needs [Python 3.10](https://www.python.org/downloads/release/python-3100/) to run.

At the root directory of the program, run the `setup.bat` script first. 

Then you must update the `.env` file with the [AzureOpenAI](https://oai.azure.com/) API key and the full directory path you wish to apply cognitive search to.

Once that's complete, you can start the program by executing the `run.bat` script.

# Usage
The Marketing AI QA bot uses the ChatGPT 3.5 model in conjunction with cognitive search to intelligently query the data, in a human like fashion. It uses converstaion memory, so it can reference previously mentioned items for new queries.

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