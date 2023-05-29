from bs4 import BeautifulSoup
import requests
from textwrap3 import wrap
import tiktoken
import pyperclip3 as pyperclip
import re
from dotenv import load_dotenv
import os
import openai
import time
import numpy as np
from sklearn.cluster import KMeans

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


def summarize_chunk(chunk: str, model="gpt-3.5-turbo"):
    summary = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Please summarize the following text in about 200 words or less: {chunk}"}
        ],
        temperature=0,
    )

    return summary['choices'][0]['message']['content']


def cluster_embedding(embedding, n_clusters: int):
    matrix = np.vstack(embedding.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(matrix)

    return kmeans.labels_


def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']


def remove_extra_spaces(text: str):
    # Replace all whitespace (space, tab, newline etc.) sequences with a single space
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk_string(string: str, max_length: int):
    """
    This function splits a string into chunks with a maximum token length.

    Args:
        string (str): The string to split into chunks.
        max_length (int): The maximum token length of each chunk.

    Returns:
      list: A list of chunks.
    """
    chunks = wrap(string, max_length)

    # result = []

    return chunks


def scrape_url(url: str):
    """
    This function takes in a URL and uses the requests library to retrieve the HTML from the URL.
    Then, it uses the BeautifulSoup library to parse the HTML and store it in a variable called s.
    Finally, it loops through all the children of the body tag and checks if they are tags that we're interested in and returns the text of those tags.
    """

    """ 
    Requests library allows us to send HTTP requests easily. 
    Specifically, we will be using the get() function to retrieve the HTML from the URL we pass in as a parameter.
    """
    r = requests.get(url)

    print(f"Status code: {r.status_code}")

    """
    We'll use the BeautifulSoup library to parse the HTML and store it in a variable called s.
    requests.content returns the data of an HTTP response, and we use the html.parser to parse the data that we got from the request.
    Something to note about the difference between requests.content and requests.text is that requests.text returns the Unicode whereas requests.content returns the bytes.
    BeautifulSoup accepts both Unicode and bytes and will do the necessary conversion(s) internally.
    """
    s = BeautifulSoup(r.content, 'html.parser')

    urlDetails = []

    # Get the body content
    body_content = s.find('body')

    # tags that we're interested in
    tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span', 'a']

    # Loop through all the children of body_content and check if they are tags that we're interested in then append them to urlDetails
    for child in body_content.children:
        # Check if this is a tag that we're interested in
        if child.name in tags:
            # Append the text of this tag to urlDetails
            urlDetails.append(child.text.strip())

    # Combine all the text in urlDetails into one string
    urlDetails = ' '.join(urlDetails)

    # Remove extra spaces
    urlDetails = remove_extra_spaces(urlDetails)

    return urlDetails


def get_num_tokens(text: str, model="gpt-3.5-turbo", encoding_name="cl100k_base"):
    """
    This function takes in a string and returns the number of tokens in that string using tiktoken.
    """
    encoding = tiktoken.get_encoding(encoding_name)

    # Check if the encoding is valid
    assert encoding.decode(encoding.encode(text)) == text

    num_tokens = len(encoding.encode(text))
    return num_tokens


if __name__ == "__main__":
    tic = time.perf_counter()

    bold = "\033[1m"
    green = "\033[32m"
    reset = "\033[0m"

    url = "https://en.wikipedia.org/wiki/%22Hello,_World!%22_program"
    print(f"URL: {url}\n")

    urlDetails = scrape_url(url)  # type: str
    print(f"URL length: {len(urlDetails)}")
    pyperclip.copy(urlDetails)

    print(f"Number of tokens: {get_num_tokens(urlDetails)}")
    # print(urlDetails)

    chunks = chunk_string(urlDetails, 4000)
    print(f"Number of chunks: {len(chunks)}\n")

    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(
            f"{bold}{green}Number of tokens for chunk #{i} : {get_num_tokens(chunk)}{reset}")
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)
        # print(f"Chunk summary:\n{summarize_chunk(chunk)}\n")
        clustered_embedding = cluster_embedding(embedding, 4)
        print(f"Clustered embedding:\n{clustered_embedding}\n")

    print(f"\nNumber of chunk embeddings: {len(chunk_embeddings)}")
    print(f"\nThe number of chunks and chunk embeddings should be the same.")

    toc = time.perf_counter()
    print(f"\nTime taken: {toc - tic:0.4f} seconds")
