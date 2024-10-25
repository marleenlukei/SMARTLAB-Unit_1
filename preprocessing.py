import string
import zipfile



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

    
def clean_data(data):
    data = data.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('Subject', '')
    data = data.lower()
    # Remove punctuations
    data = data.translate(str.maketrans('', '', string.punctuation))  
    return data

 

