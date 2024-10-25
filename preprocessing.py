import re
import string
import unicodedata
import zipfile
from email import policy
from email.parser import BytesParser
import chardet



def load_data(path):
    labels = []
    content = []
    file_names = []  
    data = zipfile.ZipFile(path, 'r')
    for name in data.namelist():
        if name.endswith('labels'):
            continue
        labels.append(int(name[-1]))
        content.append(data.read(name).decode('utf-8', errors='ignore'))
        file_names.append(name) 
    
    return content, labels, file_names

def normalize_string(s):
    """Normalizes a string by lowercasing, removing diacritics, removing punctuation and removing non-alphanumeric characters.
    """
    s = s.lower()
    # Split accented characters.
    s = unicodedata.normalize('NFKD', s)
    # Keep space, a to z
    s = re.sub(r"[^ a-z\s]+", r"", s) 
    s = re.sub(r'\s\s+', ' ', s).strip() # Remove multiple spaces
    s = s.translate(str.maketrans('', '', string.punctuation)) 
    return s.strip()


def extract_email_content(email_data):
    # Parse the email
    email_bytes = email_data.encode('utf-8')
    msg = BytesParser(policy=policy.default).parsebytes(email_bytes)
    
    content_dict = {}  # Dictionary to hold content types and their text
    subject = msg['Subject'] if msg['Subject'] else "No Subject"
    
    if msg.is_multipart():
        for part in msg.walk():
            try:
                content_type = part.get_content_type()  # Get the content type
                content_disposition = part.get_content_disposition()
                charset = part.get_content_charset() or 'utf-8'

                if content_disposition != "attachment":  # Skip attachments
                    payload = part.get_payload(decode=True)
                    if payload:
                        text = payload.decode(charset, errors='replace')
                        content_dict[content_type] = text  # Store content type and its text
                else:
                    content_dict["Attachment"] = "Attachment"
                        
                    
            except LookupError:
                # If there's an unknown encoding, use chardet to detect it
                result = chardet.detect(part.get_payload().encode('utf-8'))
                encoding = result['encoding']
                if content_disposition != "attachment":
                    payload = part.get_payload(decode=True)
                    if payload:
                        text = payload.decode(encoding, errors='replace')
                        content_dict[content_type] = text  # Store content type and its text
                else:
                    content_dict["Attachment"] = "Attachment"

    else:
        try:
            content_type = msg.get_content_type() 
            charset = msg.get_content_charset() or 'utf-8'
            payload = msg.get_payload(decode=True)
            if payload:
                plain_text_body = payload.decode(charset, errors='replace')
                content_dict[msg.get_content_type()] = plain_text_body  # Store the single part content type and text
        except LookupError:
                result = chardet.detect(msg.get_payload().encode('utf-8'))
                encoding = result['encoding']
                plain_text_body = msg.get_payload(decode=True).decode(encoding, errors='replace')
                content_dict[msg.get_content_type()] = plain_text_body 


    
    return content_dict, subject





 

def extract_content_features(content_dict):
    content_types = {'text/plain': 0, 'text/html': 0, 'application/pgp-signature': 0, 
                     'image/gif': 0, 'image/jpeg': 0, 'Attachment': 0, 
                     'application/octet-stream': 0, 'text/plain charset=us-ascii': 0, 
                     'text/x-patch': 0, 'text/x-diff': 0, 'application/x-gzip': 0, 
                     'image/png': 0, 'application/pdf': 0, 'plain/text': 0,
                     'text/html': 0, 'text/plain': 0, 'image/gif': 0, 
                     'image/jpeg': 0, 'image/png': 0, 'Attachment': 0, 
                     'application/x-msdownload': 0, 'text/rfc822-headers': 0, 
                     'image/jpg': 0, 'multipart/alternative': 0}
    for key in content_dict.keys():
        if key in content_types:
            content_types[key] = 1  # Presence of this content type
    return list(content_types.values())