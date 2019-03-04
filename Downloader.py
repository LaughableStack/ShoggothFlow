from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pickle
def collect(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if verify(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))
        return None
def verify(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)
def remove_chars(someString,charset):
  return "".join(c for c in someString if c not in charset)
def savefile(path,content):
  o = open(path,"w")
  o.write(content)
  o.close()
def get_bsr(url):
  r = collect(url)
  return BeautifulSoup(r,'html.parser')
appendix = get_bsr('http://www.hplovecraft.com/writings/texts/')
writings = []
for li in appendix.select('li'):
  try:
    writings.append(li.a['href'])
  except:
    pass
passages = []
for w in writings:
  writingsoup = get_bsr(("http://www.hplovecraft.com/writings/texts/"+w))
  divs = []
  for div in (writingsoup.select('div')):
    try:
      div['align'] #align feature is not within inapplicable divs. This returns an error if it isn't present
      divs.append(div)
    except:
      pass
  passages.append(divs[0].text)# the first one is always the actual text
f = open("LovecraftPassages.dat","wb+")
pickle.dump(passages,f)
f.close()