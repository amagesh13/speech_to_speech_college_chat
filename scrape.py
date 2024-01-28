# Requests is a Python libary for using HTTPS requests.
import requests

# BeautifulSoup is a Python libary that parses structured data from HTML and XML files.
from bs4 import BeautifulSoup

# Gets the website's HTML.
def get_website_content(url):
    page = requests.get(url)
    if page.status_code != 200:
        print("Error, failed to fetch the content from", url)
    else:
        return page.content

# Extracts the text from the HTML.
def extract_text_from_content(html_content):

    # Constructor for BeautifulSoup, pass in the HTML content and specify HTML parser
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extracts the text from the soup of HTML content.  Separates the blocks of text w/ a space.
    text_content = soup.get_text(separator=' ')

    # Removes whitespace from beginning and end before returning the text.
    return text_content.strip()


url_list = ["https://billpay.umd.edu/UndergraduateTuition", "https://billpay.umd.edu/GraduateTuition", "https://billpay.umd.edu/billing", "https://reslife.umd.edu/terp-housing/residence-halls", "https://reslife.umd.edu/terp-housing/rates-fees", "https://reslife.umd.edu/terp-housing/room-layouts-tours", "https://reslife.umd.edu/terp-housing/dates", "https://dining.umd.edu/hours-locations/dining-halls", "https://dining.umd.edu/hours-locations/cafes", "https://dining.umd.edu/student-dining-plans", "https://recwell.umd.edu/facilities/facilities"]
info = ""
for url in url_list:
    html_content = get_website_content(url_list)
    if html_content:
        text = extract_text_from_content(html_content)
        formatted_text = ' '.join(text.split())
        info = info + formatted_text
    else:
        # Error occurred, continue with the next url.
        continue



# Creates a text file called data.txt and writes the scraped data to it.
with open('./data.txt', 'w') as file:
    # Write content to the file
    file.write(info)