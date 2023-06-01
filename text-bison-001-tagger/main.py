"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""

import google.generativeai as palm
import time
import sys
from dotenv import load_dotenv
import os

load_dotenv()

PALM_API_KEY = os.getenv("PALM_API_KEY")

tic = time.perf_counter()

bold = "\033[1m"
green = "\033[32m"
reset = "\033[0m"

url = sys.argv[1]
# url = "https://rogermartin.medium.com/business-model-generation-playing-to-win-d50c33d6dfeb"

palm.configure(api_key=PALM_API_KEY)

defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    'safety_settings': [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1}, {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1}, {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2}, {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2}, {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2}, {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2}],
}
prompt = f"""Please analyze the following url of an article/blog/document and generate two distinct lists of tags that encapsulate its content.

'Themes': Generate up to five tags that represent the overarching themes or major topics conveyed in the text. These tags should illuminate broad concepts and ideas that capture the core of the content.

'Specifics': Create up to five tags pointing out specific entities, details, or subtopics directly referenced within the text. These tags should underscore notable individuals, events, or unique details discussed in the content.

Keep each tag concise to a maximum of two words. The objective here is to meaningfully capture the essence of the content accurately, not to fill each list with five tags for the sake of it. If a smaller number of tags provide a complete and accurate representation of the content's themes and specifics, don't add more to reach five. Prioritize precision and relevance over quantity.

Pay close attention to both explicit and implicit details, ensuring that the tags are truly pertinent and meaningful. Avoid biases and maintain objectivity. The goal is to provide an accurate, succinct, and comprehensive representation of the text's content through these tags.

Also pay close attention to repeats. Ensure that each tag is unique and that no two tags are synonymous. If a tag is too similar to another, consider combining them into one tag or removing one of them entirely. Make sure that each tag is distinct and provides a unique insight into the content and they are not dirivatives of one another.

The text (between "---- START OF TEXT ----" and "---- END OF TEXT ----") for analysis is as follows:

---- START OF TEXT ----
{url}
---- END OF TEXT ----

Remember: Quality over quantity. Each tag should provide valuable insight into the content. Avoid including tags that only marginally relate to the content or dilute the significance of the other tags. The ultimate goal is to create a clear, concise snapshot of the text's content through the most illuminating and relevant tags.

Ensure the returned format is like the example below:
Themes:
1. First tag
2. Second tag

Specifics:
1. First tag
2. Second tag
3. Third tag
"""

response = palm.generate_text(
    **defaults,
    prompt=prompt
)
print(response.result)

toc = time.perf_counter()
print(f"\n{green}{bold}Time taken to tagify {url}: {toc - tic:0.4f} seconds{reset}")
